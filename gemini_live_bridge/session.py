import asyncio
import base64
import json
import logging
import time

import numpy as np
import websockets
import websockets.server

from config import (
    DEVICE_SPK_RATE,
    GEMINI_IN_RATE,
    GEMINI_MODEL,
    GEMINI_VOICE,
    GEMINI_WS_URL,
    TAG_AUDIO,
    TAG_END_TURN,
    TAG_INTERRUPT,
)
from audio import resample_gemini_to_device
from aec import EchoCanceler
from ha_client import HAClient
from tools import build_gemini_setup

log = logging.getLogger(__name__)

class GeminiSession:
    def __init__(self, device_ws: websockets.server.ServerConnection):
        self.device_ws  = device_ws
        self.gemini_ws  = None
        self._stop      = asyncio.Event()
        self._interrupt = asyncio.Event()
        self.out_queue  = asyncio.Queue()
        self.addr       = device_ws.remote_address
        self.ha         = HAClient()
        self.aec        = EchoCanceler()

    async def run(self):
        log.info("[%s] Device connected", self.addr)
        try:
            async with websockets.connect(
                GEMINI_WS_URL,
                ping_interval=20,
                ping_timeout=10,
                max_size=10 * 1024 * 1024,
            ) as gemini_ws:
                self.gemini_ws = gemini_ws

                # 1. Handshake
                await gemini_ws.send(json.dumps(build_gemini_setup()))
                raw = await gemini_ws.recv()
                parsed = json.loads(raw)
                if "setupComplete" not in parsed:
                    log.error("[%s] Unexpected setup response: %s", self.addr, parsed)
                    return
                log.info("[%s] Gemini session ready  model=%s  voice=%s", self.addr, GEMINI_MODEL, GEMINI_VOICE)

                # 2. Run all coroutines concurrently
                await asyncio.gather(
                    self._device_to_gemini(),
                    self._gemini_to_device(),
                    self._audio_pacing_loop(),
                    return_exceptions=True,
                )
        except websockets.exceptions.ConnectionClosed as e:
            log.info("[%s] Gemini connection closed: %s", self.addr, e)
        except Exception as e:
            log.exception("[%s] Session error: %s", self.addr, e)
        finally:
            await self.ha.close()
            log.info("[%s] Session ended", self.addr)

    # ------------------------------------------------------------------
    # Device → Gemini  (mic audio)
    # ------------------------------------------------------------------

    async def _device_to_gemini(self):
        """Convert 32-bit stereo I2S audio → 16-bit mono and forward to Gemini."""
        try:
            async for raw in self.device_ws:
                if self._stop.is_set():
                    break
                if not isinstance(raw, bytes) or len(raw) == 0:
                    continue

                # 32-bit stereo → 16-bit mono (left channel, upper 16 bits)
                samples_i32 = np.frombuffer(raw, dtype=np.int32)
                left        = samples_i32[0::2]
                mono_i16    = (left >> 16).astype(np.int16)
                pcm_16k     = mono_i16.tobytes()

                # Acoustic echo cancellation: subtract speaker reference.
                pcm_16k = self.aec.process_mic(pcm_16k)
                if not pcm_16k:
                    continue

                encoded = base64.b64encode(pcm_16k).decode("ascii")
                await self.gemini_ws.send(json.dumps({
                    "realtimeInput": {
                        "audio": {
                            "data": encoded,
                            "mimeType": f"audio/pcm;rate={GEMINI_IN_RATE}",
                        }
                    }
                }))
        except websockets.exceptions.ConnectionClosed:
            log.info("[%s] Device disconnected (mic)", self.addr)
        finally:
            self._stop.set()

    # ------------------------------------------------------------------
    # Gemini → Device  (responses + tool calls)
    # ------------------------------------------------------------------

    async def _gemini_to_device(self):
        try:
            async for raw_msg in self.gemini_ws:
                if self._stop.is_set():
                    break
                try:
                    msg = json.loads(raw_msg)
                except json.JSONDecodeError:
                    log.warning("[%s] Non-JSON from Gemini: %r", self.addr, raw_msg[:80])
                    continue

                await self._handle_gemini_message(msg)

        except websockets.exceptions.ConnectionClosed:
            log.info("[%s] Gemini disconnected (audio loop)", self.addr)
        finally:
            self._stop.set()
            await self._safe_send(TAG_END_TURN)

    async def _handle_gemini_message(self, msg: dict):

        # ---- Tool calls -------------------------------------------------
        tool_call = msg.get("toolCall")
        if tool_call:
            fn_calls = tool_call.get("functionCalls", [])
            responses = []
            for fn in fn_calls:
                call_id = fn.get("id", "")
                name    = fn.get("name", "")
                args    = fn.get("args", {})

                content_parts = await self.ha.dispatch_tool(name, args)
                responses.append({
                    "id": call_id,
                    "name": name,
                    "response": {"output": content_parts},
                })
                log.info("[%s] Tool call response: %s", self.addr, content_parts)

            await self.gemini_ws.send(json.dumps({
                "toolResponse": {"functionResponses": responses}
            }))
            return

        # ---- Barge-in ---------------------------------------------------
        if msg.get("serverContent", {}).get("interrupted"):
            log.debug("[%s] Barge-in", self.addr)
            while not self.out_queue.empty():
                try:
                    self.out_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            self._interrupt.set()
            self.aec.reset_reference()
            await self._safe_send(TAG_INTERRUPT)
            return

        # ---- Turn complete ----------------------------------------------
        if msg.get("serverContent", {}).get("turnComplete"):
            log.debug("[%s] Turn complete", self.addr)
            await self.out_queue.put(TAG_END_TURN)
            return

        # ---- Audio parts ------------------------------------------------
        parts = (
            msg.get("serverContent", {})
               .get("modelTurn", {})
               .get("parts", [])
        )
        for part in parts:
            inline   = part.get("inlineData", {})
            mime     = inline.get("mimeType", "")
            data_b64 = inline.get("data", "")

            if data_b64 and mime.startswith("audio/"):
                pcm_24k     = base64.b64decode(data_b64)
                pcm_device  = resample_gemini_to_device(pcm_24k)
                await self.out_queue.put(TAG_AUDIO + pcm_device)

    # ------------------------------------------------------------------
    # Audio pacing loop
    # ------------------------------------------------------------------

    async def _audio_pacing_loop(self):
        """Drains the out_queue and sends audio to the device just slightly
        ahead of real time to maintain a small playback buffer on the ESP32."""
        CHUNK_SIZE = 4096
        try:
            next_send = time.monotonic()
            while not self._stop.is_set():
                self._interrupt.clear()
                try:
                    frame = await asyncio.wait_for(self.out_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                if frame == TAG_END_TURN:
                    await self._safe_send(TAG_END_TURN)
                    next_send = time.monotonic()
                    continue

                pcm = frame[1:]
                for i in range(0, len(pcm), CHUNK_SIZE):
                    if self._stop.is_set() or self._interrupt.is_set():
                        break

                    now = time.monotonic()
                    if now > next_send:
                        next_send = now

                    chunk    = pcm[i : i + CHUNK_SIZE]
                    duration = len(chunk) / (DEVICE_SPK_RATE * 2.0)

                    await self._safe_send(TAG_AUDIO + chunk)
                    self.aec.push_reference(chunk)
                    next_send += duration

                    # 50 ms look-ahead buffer for ESP32 jitter absorption
                    sleep_t = next_send - time.monotonic() - 0.050
                    if sleep_t > 0:
                        await asyncio.sleep(sleep_t)

        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _safe_send(self, data: bytes):
        try:
            await self.device_ws.send(data)
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            log.warning("[%s] Send error: %s", self.addr, e)
