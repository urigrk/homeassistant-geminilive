#!/usr/bin/env python3
"""
Gemini Live Bridge — Home Assistant Add-on
==========================================
Listens for WebSocket connections from Voice PE devices, relays audio to the
Gemini Live API in real time, and streams Gemini's audio responses back.

Gemini has full access to Home Assistant via function-calling tools:
  • get_entities        – list/filter entities with state & attributes
  • get_entity_state    – fetch a single entity's state + attributes
  • control_device      – call any HA service (turn on/off, set temp, etc.)
  • get_camera_image    – capture a still from a camera entity (base64 JPEG)
  • send_notification   – push a persistent notification to the HA dashboard
  • get_history         – fetch recent state history for an entity
  • run_script          – trigger a HA script by entity_id
  • activate_scene      – activate a HA scene

Wire protocol (binary WebSocket frames):

  Device → Bridge:  raw PCM bytes (int32, stereo, 16 kHz I2S)
  Bridge → Device:
    0x01 + PCM bytes  →  audio  (int16, mono, configurable kHz, LE)
    0x02              →  interrupt  (barge-in: stop playback)
    0x03              →  end-of-turn  (Gemini finished speaking)

Configuration is read from environment variables set by run.sh (which reads
/data/options.json via bashio).
"""

import asyncio
import base64
import collections
import json
import logging
import os
import signal
import textwrap
import time
from math import gcd
from typing import Optional

import aiohttp
import numpy as np
import websockets
import websockets.server
from scipy.signal import resample_poly

# ---------------------------------------------------------------------------
# Configuration  (populated from env vars set by run.sh / bashio)
# ---------------------------------------------------------------------------

BRIDGE_HOST   = "0.0.0.0"
BRIDGE_PORT   = int(os.environ.get("BRIDGE_PORT", 8765))

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-live-001")
GEMINI_VOICE   = os.environ.get("GEMINI_VOICE", "Aoede")
GEMINI_WS_URL  = (
    "wss://generativelanguage.googleapis.com"
    "/ws/google.ai.generativelanguage.v1beta"
    ".GenerativeService.BidiGenerateContent"
    f"?key={GEMINI_API_KEY}"
)

LANGUAGE              = os.environ.get("LANGUAGE", "en-US")
USER_SYSTEM_PROMPT    = os.environ.get("SYSTEM_PROMPT", "").strip()
ENABLE_DEVICE_CONTROL = os.environ.get("ENABLE_DEVICE_CONTROL", "true").lower() == "true"
ENABLE_CAMERA_ACCESS  = os.environ.get("ENABLE_CAMERA_ACCESS", "true").lower() == "true"
ENABLE_NOTIFICATIONS  = os.environ.get("ENABLE_NOTIFICATIONS", "true").lower() == "true"
LOG_LEVEL             = os.environ.get("LOG_LEVEL", "info").upper()

HA_BASE_URL = os.environ.get("HA_BASE_URL", "http://supervisor/core")
HA_TOKEN    = os.environ.get("HA_TOKEN", "")
HA_HEADERS  = {"Authorization": f"Bearer {HA_TOKEN}", "Content-Type": "application/json"}

# Audio constants
DEVICE_MIC_RATE   = 16_000
DEVICE_SPK_RATE   = int(os.environ.get("AUDIO_OUTPUT_RATE", 48_000))
GEMINI_IN_RATE    = 16_000
GEMINI_OUT_RATE   = 24_000   # Gemini always outputs 24 kHz

# Wire tags
TAG_AUDIO     = bytes([0x01])
TAG_INTERRUPT = bytes([0x02])
TAG_END_TURN  = bytes([0x03])

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
)
log = logging.getLogger("bridge")

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = textwrap.dedent("""
    You are a smart, friendly voice assistant built into a Home Assistant smart home system.
    You have real-time access to the home's devices, sensors, and cameras.

    ## Personality & style
    - Speak naturally and conversationally — you are heard through a speaker, not read on screen.
    - Keep answers brief and to the point. Do NOT use bullet points, lists, markdown, or headers.
    - If listing multiple items, speak them naturally: "You have three lights on: the kitchen, the living room, and the bedroom."
    - Be warm but efficient. Confirm actions concisely: "Done, turning off the kitchen lights."
    - If unsure about which device the user means, ask one clarifying question.
    - Never say "I cannot access real-time data" — you have live tools for that. Use them.
    - Proactively give useful context: if asked to turn on the AC, mention the current indoor temperature.

    ## Home control guidelines
    - When the user asks to control something, call the appropriate tool immediately — don't ask for confirmation unless the action is irreversible or unusual.
    - If a device name is ambiguous (e.g. "the light"), use get_entities to find candidates and clarify.
    - After controlling a device, confirm what you did: "Okay, the thermostat is now set to 22 degrees."
    - For scenes and scripts, activate them directly when requested.
    - When asked about cameras, get the image and describe what you see clearly and concisely.
    - Use get_history when the user asks about past events: "Was the front door opened today?"

    ## Safety & awareness
    - Never unlock doors, disable alarms, or perform security-sensitive actions without clear explicit user intent.
    - If a sensor shows a potential issue (smoke, CO, flood), mention it proactively and suggest action.
    - If a requested device is unavailable or offline, say so clearly and suggest alternatives.

    ## What you can do
    - Check and control lights, switches, covers (blinds/garage), fans, climate (thermostat/AC/heat), media players, locks, alarms.
    - Read temperature, humidity, motion, door/window, power, and other sensors.
    - View camera feeds and describe what's visible.
    - Activate scenes and run automations/scripts.
    - Send notifications to the Home Assistant dashboard.
    - Report on historical state: when something last changed, trends, etc.
""").strip()

SYSTEM_PROMPT = (
    USER_SYSTEM_PROMPT if USER_SYSTEM_PROMPT else DEFAULT_SYSTEM_PROMPT
)

# ---------------------------------------------------------------------------
# Gemini tool definitions
# ---------------------------------------------------------------------------

def build_tools() -> list:
    """Build the Gemini function declarations based on enabled features."""
    tools = [
        {
            "name": "get_entities",
            "description": (
                "Search Home Assistant entities by domain and/or a keyword in their "
                "name or entity_id. Returns entity_id, friendly name, state, and key attributes. "
                "Use this to discover which devices exist or to find an entity_id before controlling it."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "domain": {
                        "type": "STRING",
                        "description": (
                            "Optional HA domain to filter by, e.g. 'light', 'switch', 'climate', "
                            "'sensor', 'binary_sensor', 'cover', 'media_player', 'lock', 'scene', 'script'."
                        ),
                    },
                    "search": {
                        "type": "STRING",
                        "description": "Optional keyword to search in entity_id or friendly_name (case-insensitive).",
                    },
                    "area": {
                        "type": "STRING",
                        "description": "Optional area/room name to filter entities by location.",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "get_entity_state",
            "description": "Get the current state and all attributes of a single entity.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "entity_id": {
                        "type": "STRING",
                        "description": "The full entity_id, e.g. 'light.kitchen' or 'sensor.outdoor_temperature'.",
                    }
                },
                "required": ["entity_id"],
            },
        },
        {
            "name": "get_history",
            "description": (
                "Get the state history of an entity for the past N hours. "
                "Use to answer questions like 'was the door open today?' or 'what was the temperature this morning?'."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "entity_id": {"type": "STRING", "description": "The entity_id to query."},
                    "hours": {
                        "type": "NUMBER",
                        "description": "How many hours of history to retrieve (default: 24, max: 168).",
                    },
                },
                "required": ["entity_id"],
            },
        },
    ]

    if ENABLE_DEVICE_CONTROL:
        tools += [
            {
                "name": "control_device",
                "description": (
                    "Call a Home Assistant service to control a device or set of devices. "
                    "Examples: turn lights on/off, set thermostat temperature, open/close covers, "
                    "set media volume, lock/unlock doors, set light brightness or color."
                ),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "domain": {
                            "type": "STRING",
                            "description": "Service domain, e.g. 'light', 'switch', 'climate', 'cover', 'media_player', 'lock'.",
                        },
                        "service": {
                            "type": "STRING",
                            "description": (
                                "Service name, e.g. 'turn_on', 'turn_off', 'toggle', "
                                "'set_temperature', 'set_hvac_mode', 'open_cover', 'close_cover', "
                                "'media_play', 'media_pause', 'volume_set'."
                            ),
                        },
                        "entity_id": {
                            "type": "STRING",
                            "description": "Single entity_id or comma-separated list to target.",
                        },
                        "service_data": {
                            "type": "OBJECT",
                            "description": (
                                "Optional extra service data. Examples: "
                                "{\"brightness_pct\": 80} for lights, "
                                "{\"temperature\": 22} for climate, "
                                "{\"volume_level\": 0.5} for media_player, "
                                "{\"rgb_color\": [255, 100, 0]} for colored lights."
                            ),
                        },
                    },
                    "required": ["domain", "service", "entity_id"],
                },
            },
            {
                "name": "activate_scene",
                "description": "Activate a Home Assistant scene by entity_id.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "entity_id": {
                            "type": "STRING",
                            "description": "The scene entity_id, e.g. 'scene.movie_night'.",
                        }
                    },
                    "required": ["entity_id"],
                },
            },
            {
                "name": "run_script",
                "description": "Trigger a Home Assistant script by entity_id.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "entity_id": {
                            "type": "STRING",
                            "description": "The script entity_id, e.g. 'script.good_morning'.",
                        }
                    },
                    "required": ["entity_id"],
                },
            },
        ]

    if ENABLE_CAMERA_ACCESS:
        tools.append({
            "name": "get_camera_image",
            "description": (
                "Capture a current still image from a Home Assistant camera entity. "
                "Returns a JPEG image so you can describe what you see. "
                "Use when the user asks to check a camera, see who's at the door, etc."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "entity_id": {
                        "type": "STRING",
                        "description": "The camera entity_id, e.g. 'camera.front_door'.",
                    }
                },
                "required": ["entity_id"],
            },
        })

    if ENABLE_NOTIFICATIONS:
        tools.append({
            "name": "send_notification",
            "description": "Send a persistent notification to the Home Assistant dashboard.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "title": {"type": "STRING", "description": "Notification title."},
                    "message": {"type": "STRING", "description": "Notification body text."},
                },
                "required": ["title", "message"],
            },
        })

    return tools


def build_gemini_setup() -> dict:
    return {
        "setup": {
            "model": f"models/{GEMINI_MODEL}",
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {"voiceName": GEMINI_VOICE}
                    },
                    "languageCode": LANGUAGE,
                },
            },
            "systemInstruction": {
                "parts": [{"text": SYSTEM_PROMPT}]
            },
            "tools": [{"functionDeclarations": build_tools()}],
        }
    }


# ---------------------------------------------------------------------------
# Home Assistant API client
# ---------------------------------------------------------------------------

class HAClient:
    """Thin async wrapper around the Home Assistant REST API."""

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=HA_HEADERS)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def get(self, path: str) -> dict | list:
        session = await self._get_session()
        async with session.get(f"{HA_BASE_URL}/api{path}") as resp:
            resp.raise_for_status()
            return await resp.json()

    async def get_raw(self, path: str) -> bytes:
        session = await self._get_session()
        async with session.get(f"{HA_BASE_URL}/api{path}") as resp:
            resp.raise_for_status()
            return await resp.read()

    async def post(self, path: str, data: dict = None) -> dict:
        session = await self._get_session()
        async with session.post(f"{HA_BASE_URL}/api{path}", json=data or {}) as resp:
            resp.raise_for_status()
            try:
                return await resp.json()
            except Exception:
                return {}

    # ---- Tool implementations -----------------------------------------------

    async def get_entities(self, domain: str = None, search: str = None, area: str = None) -> str:
        states = await self.get("/states")
        results = []
        for s in states:
            eid = s.get("entity_id", "")
            name = s.get("attributes", {}).get("friendly_name", eid)
            state = s.get("state", "")
            attrs = s.get("attributes", {})

            # Domain filter
            if domain and not eid.startswith(f"{domain}."):
                continue
            # Keyword filter
            if search and search.lower() not in eid.lower() and search.lower() not in name.lower():
                continue
            # Area filter (best-effort, attr may not be present)
            if area:
                entity_area = attrs.get("area_id", "") or ""
                if area.lower() not in entity_area.lower() and area.lower() not in name.lower():
                    continue

            # Include a useful subset of attributes
            useful_attrs = {}
            keep = [
                "friendly_name", "unit_of_measurement", "device_class",
                "brightness", "color_temp", "rgb_color", "temperature",
                "current_temperature", "hvac_mode", "hvac_modes",
                "volume_level", "media_title", "is_volume_muted",
                "battery_level", "last_triggered",
            ]
            for k in keep:
                if k in attrs:
                    useful_attrs[k] = attrs[k]

            results.append({
                "entity_id": eid,
                "name": name,
                "state": state,
                "attributes": useful_attrs,
            })

        if not results:
            return json.dumps({"result": "no matching entities found"})
        return json.dumps({"entities": results[:50]})  # cap at 50

    async def get_entity_state(self, entity_id: str) -> str:
        try:
            state = await self.get(f"/states/{entity_id}")
            return json.dumps({
                "entity_id": state["entity_id"],
                "state": state["state"],
                "attributes": state.get("attributes", {}),
                "last_changed": state.get("last_changed"),
            })
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def get_history(self, entity_id: str, hours: float = 24) -> str:
        import datetime
        hours = min(float(hours), 168)
        start = (datetime.datetime.utcnow() - datetime.timedelta(hours=hours)).isoformat() + "Z"
        try:
            history = await self.get(f"/history/period/{start}?filter_entity_id={entity_id}&minimal_response=true")
            if not history or not history[0]:
                return json.dumps({"result": "no history found"})
            entries = history[0]
            # Summarize: keep state transitions (deduplicate consecutive same-state)
            summary = []
            prev_state = None
            for entry in entries:
                s = entry.get("state")
                t = entry.get("last_changed", "")
                if s != prev_state:
                    summary.append({"state": s, "changed_at": t})
                    prev_state = s
            return json.dumps({"entity_id": entity_id, "hours": hours, "transitions": summary})
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def control_device(self, domain: str, service: str, entity_id: str, service_data: dict = None) -> str:
        data = {"entity_id": entity_id}
        if service_data:
            data.update(service_data)
        try:
            await self.post(f"/services/{domain}/{service}", data)
            return json.dumps({"result": "success", "domain": domain, "service": service, "entity_id": entity_id})
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def activate_scene(self, entity_id: str) -> str:
        return await self.control_device("scene", "turn_on", entity_id)

    async def run_script(self, entity_id: str) -> str:
        return await self.control_device("script", "turn_on", entity_id)

    async def get_camera_image(self, entity_id: str) -> dict:
        """Returns dict with mime_type and base64 data for inline Gemini image."""
        try:
            img_bytes = await self.get_raw(f"/camera_proxy/{entity_id}")
            b64 = base64.b64encode(img_bytes).decode("ascii")
            return {"mime_type": "image/jpeg", "data": b64}
        except Exception as e:
            return {"error": str(e)}

    async def send_notification(self, title: str, message: str) -> str:
        try:
            await self.post("/services/persistent_notification/create", {
                "title": title,
                "message": message,
            })
            return json.dumps({"result": "notification sent"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def dispatch_tool(self, name: str, args: dict) -> list:
        """Execute a tool call and return Gemini-compatible content parts."""
        log.info("Tool call: %s(%s)", name, args)

        try:
            if name == "get_entities":
                result_text = await self.get_entities(**args)
                return [{"text": result_text}]

            elif name == "get_entity_state":
                result_text = await self.get_entity_state(**args)
                return [{"text": result_text}]

            elif name == "get_history":
                result_text = await self.get_history(**args)
                return [{"text": result_text}]

            elif name == "control_device":
                result_text = await self.control_device(**args)
                return [{"text": result_text}]

            elif name == "activate_scene":
                result_text = await self.activate_scene(**args)
                return [{"text": result_text}]

            elif name == "run_script":
                result_text = await self.run_script(**args)
                return [{"text": result_text}]

            elif name == "get_camera_image":
                result = await self.get_camera_image(**args)
                if "error" in result:
                    return [{"text": json.dumps(result)}]
                # Return as inline image so Gemini can describe it
                return [
                    {"text": f"Camera snapshot from {args.get('entity_id')}:"},
                    {"inlineData": {"mimeType": result["mime_type"], "data": result["data"]}},
                ]

            elif name == "send_notification":
                result_text = await self.send_notification(**args)
                return [{"text": result_text}]

            else:
                return [{"text": json.dumps({"error": f"Unknown tool: {name}"})}]

        except Exception as e:
            log.exception("Tool %s raised an error", name)
            return [{"text": json.dumps({"error": str(e)})}]


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def resample_gemini_to_device(pcm_bytes: bytes) -> bytes:
    """Resample Gemini's 24 kHz int16 PCM output to the device speaker rate."""
    if GEMINI_OUT_RATE == DEVICE_SPK_RATE:
        return pcm_bytes
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    g    = gcd(DEVICE_SPK_RATE, GEMINI_OUT_RATE)
    up   = DEVICE_SPK_RATE // g
    down = GEMINI_OUT_RATE  // g
    resampled = resample_poly(samples, up=up, down=down)
    return resampled.astype(np.int16).tobytes()


def resample_device_to_mic(pcm_bytes: bytes) -> bytes:
    """Resample device speaker-rate int16 PCM down to the Gemini mic rate (16 kHz).

    Used to build the AEC reference signal from what is actually sent to the
    device speaker so it can be compared against what the microphone captures.
    """
    if DEVICE_SPK_RATE == GEMINI_IN_RATE:
        return pcm_bytes
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    g    = gcd(GEMINI_IN_RATE, DEVICE_SPK_RATE)
    up   = GEMINI_IN_RATE  // g
    down = DEVICE_SPK_RATE // g
    resampled = resample_poly(samples, up=up, down=down)
    return resampled.astype(np.int16).tobytes()


# ---------------------------------------------------------------------------
# Acoustic Echo Canceller
# ---------------------------------------------------------------------------

class EchoCanceller:
    """
    Reference-power-driven acoustic echo suppressor.

    When Gemini's audio is being played through the device speaker, the
    microphone inevitably picks up some of that audio as echo and sends it
    back to Gemini — causing Gemini to "hear itself".  This class suppresses
    that feedback.

    Strategy
    --------
    1.  The pacing loop calls ``feed_reference()`` each time a PCM chunk is
        actually sent to the device (already resampled to DEVICE_SPK_RATE).
        We immediately down-sample it to GEMINI_IN_RATE (16 kHz) and push it
        into a sliding ring-buffer that represents the reference signal.

    2.  ``_device_to_gemini`` calls ``process()`` for every mic frame before
        forwarding it to Gemini.  We look up the reference window that was
        played ~ACOUSTIC_DELAY_S seconds before this mic frame and compare
        energy levels:

            echo-to-mic ratio  =  ref_power / mic_power

        A gain in [min_gain, 1.0] is computed from the ratio and applied to
        the mic signal with a fast-attack / slow-release envelope so there
        are no audible clicks.

    3.  Barge-in still works: the gain degrades gracefully as the user's
        voice raises the mic_power significantly above the reference echo.

    Parameters (tunable via env vars)
    ----------------------------------
    AEC_SUPPRESS_DB   – peak suppression in dB         (default 20 dB)
    AEC_DELAY_MS      – acoustic delay estimate in ms  (default 80 ms)
    AEC_ATTACK_MS     – gain attack time in ms         (default 8 ms)
    AEC_RELEASE_MS    – gain release time in ms        (default 150 ms)
    """

    SUPPRESS_DB  = float(os.environ.get("AEC_SUPPRESS_DB",  "20"))
    DELAY_MS     = float(os.environ.get("AEC_DELAY_MS",     "80"))
    ATTACK_MS    = float(os.environ.get("AEC_ATTACK_MS",    "8"))
    RELEASE_MS   = float(os.environ.get("AEC_RELEASE_MS",   "150"))
    RATE         = GEMINI_IN_RATE                         # 16 000 Hz
    REF_BUF_SECS = 1.0                                   # reference ring-buffer length

    def __init__(self):
        buf_len = int(self.REF_BUF_SECS * self.RATE)
        # Ring-buffer holding the most recent REF_BUF_SECS of speaker audio
        # at GEMINI_IN_RATE.  deque(maxlen=N) automatically drops oldest entries.
        self._ref: collections.deque = collections.deque(maxlen=buf_len)

        self._delay   = int(self.DELAY_MS   / 1000.0 * self.RATE)
        self._att     = np.exp(-1.0 / (self.ATTACK_MS  / 1000.0 * self.RATE))
        self._rel     = np.exp(-1.0 / (self.RELEASE_MS / 1000.0 * self.RATE))
        self._min_gain = 10.0 ** (-self.SUPPRESS_DB / 20.0)
        self._gain    = 1.0          # current smoothed gain
        self._playing = False        # True while Gemini audio is being sent

        log.info(
            "AEC enabled  suppress=%.0f dB  delay=%.0f ms  "
            "attack=%.0f ms  release=%.0f ms",
            self.SUPPRESS_DB, self.DELAY_MS, self.ATTACK_MS, self.RELEASE_MS,
        )

    # ------------------------------------------------------------------
    # Feed reference (called from the pacing loop)
    # ------------------------------------------------------------------

    def feed_reference(self, pcm_at_spk_rate: bytes) -> None:
        """Add a chunk of speaker output to the reference ring-buffer.

        ``pcm_at_spk_rate`` is int16 LE mono PCM at DEVICE_SPK_RATE.
        It is resampled to GEMINI_IN_RATE before being stored.
        """
        pcm_16k = resample_device_to_mic(pcm_at_spk_rate)
        samples  = np.frombuffer(pcm_16k, dtype=np.int16)
        self._ref.extend(samples.tolist())
        self._playing = True

    def notify_silent(self) -> None:
        """Inform the AEC that playback has paused/ended."""
        self._playing = False

    # ------------------------------------------------------------------
    # Process microphone audio
    # ------------------------------------------------------------------

    def process(self, mic_bytes: bytes) -> bytes:
        """Apply echo suppression to a mic frame.

        ``mic_bytes`` is int16 LE mono PCM at GEMINI_IN_RATE.
        Returns suppressed audio of the same length.
        """
        mic = np.frombuffer(mic_bytes, dtype=np.int16).astype(np.float32)
        N   = len(mic)
        avail = len(self._ref)

        # Need at least (delay + N) reference samples to have a valid window
        if avail < self._delay + N:
            return mic_bytes

        # Reference block: the N samples that were played ~DELAY ms ago
        # self._ref is ordered oldest→newest; -1 is most recent.
        # We want the window [-(delay+N) : -delay] (or [-N:] when delay==0).
        ref_list = list(self._ref)
        if self._delay > 0:
            ref_block = np.array(ref_list[-(self._delay + N) : -self._delay],
                                 dtype=np.float32)
        else:
            ref_block = np.array(ref_list[-N:], dtype=np.float32)

        if len(ref_block) < N:
            return mic_bytes   # safety guard

        # ----- Gain computation -----
        mic_pwr = float(np.mean(mic      ** 2)) + 1e-9
        ref_pwr = float(np.mean(ref_block ** 2)) + 1e-9

        # Echo-to-mic power ratio; high → strong echo → suppress more
        etr = ref_pwr / mic_pwr

        # Sigmoid-like mapping: gain → min_gain when echo dominates mic
        target = max(self._min_gain, 1.0 / (1.0 + 6.0 * etr))

        # ----- Smooth gain ramp over the block (avoids clicks) -----
        alpha = self._att if target < self._gain else self._rel
        # Exponential decay from current gain to target over N samples
        final = float(alpha ** N) * self._gain + (1.0 - float(alpha ** N)) * target
        ramp  = np.linspace(self._gain, final, N, dtype=np.float32)
        self._gain = final

        out = mic * ramp
        return np.clip(out, -32768.0, 32767.0).astype(np.int16).tobytes()


# ---------------------------------------------------------------------------
# Session: one per connected device
# ---------------------------------------------------------------------------

class GeminiSession:
    def __init__(self, device_ws: websockets.server.ServerConnection):
        self.device_ws  = device_ws
        self.gemini_ws  = None
        self._stop      = asyncio.Event()
        self._interrupt = asyncio.Event()
        self.out_queue  = asyncio.Queue()
        self.addr       = device_ws.remote_address
        self.ha         = HAClient()
        self.aec        = EchoCanceller()

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

                # Echo cancellation: suppress Gemini's own voice from the mic
                pcm_16k = self.aec.process(pcm_16k)

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
            self.aec.notify_silent()
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
                    self.aec.notify_silent()
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

                    # Feed the reference BEFORE sending so the AEC buffer
                    # leads the mic by at least the network round-trip time.
                    self.aec.feed_reference(chunk)
                    await self._safe_send(TAG_AUDIO + chunk)
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


# ---------------------------------------------------------------------------
# WebSocket server
# ---------------------------------------------------------------------------

async def handle_device(ws: websockets.server.ServerConnection, path: str = "/"):
    session = GeminiSession(ws)
    await session.run()


async def main():
    log.info(
        "Gemini Live Bridge  model=%s  voice=%s  port=%d  ha=%s",
        GEMINI_MODEL, GEMINI_VOICE, BRIDGE_PORT, HA_BASE_URL,
    )
    log.info("Device control=%s  Camera=%s  Notifications=%s",
             ENABLE_DEVICE_CONTROL, ENABLE_CAMERA_ACCESS, ENABLE_NOTIFICATIONS)

    loop = asyncio.get_running_loop()
    stop = loop.create_future()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set_result, None)

    async with websockets.serve(
        handle_device,
        BRIDGE_HOST,
        BRIDGE_PORT,
        max_size=2 * 1024 * 1024,
        ping_interval=30,
        ping_timeout=10,
    ):
        log.info("Bridge ready – waiting for Voice PE devices on :%d", BRIDGE_PORT)
        await stop

    log.info("Bridge shut down cleanly")


if __name__ == "__main__":
    asyncio.run(main())