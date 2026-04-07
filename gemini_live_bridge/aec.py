"""
Acoustic Echo Cancellation using speexdsp.

The device's microphone picks up its own speaker output (Gemini's voice),
which causes Gemini to hear itself. This module subtracts a reference of
what we sent to the speaker from the mic stream before forwarding to Gemini.

Both mic and reference are processed at 16 kHz mono int16 in fixed-size
frames. A small initial delay is inserted into the reference queue to
compensate for the network + ESP32 playback look-ahead buffer.
"""

import logging
import os
from collections import deque

import numpy as np
from scipy.signal import resample_poly

from config import DEVICE_SPK_RATE, GEMINI_IN_RATE

log = logging.getLogger(__name__)

# 10 ms frames at 16 kHz
FRAME_SAMPLES = 160
FRAME_BYTES   = FRAME_SAMPLES * 2
# ~200 ms tail covers typical room + ESP32 buffering
FILTER_LENGTH = 3200

AEC_ENABLED  = os.environ.get("AEC_ENABLED", "true").lower() == "true"
AEC_DELAY_MS = int(os.environ.get("AEC_DELAY_MS", "80"))


class EchoCanceler:
    def __init__(self):
        self._ec = None
        self._mic_buf = bytearray()
        self._ref_buf = bytearray()  # 16 kHz mono int16
        self._ref_lock_primed = False

        if not AEC_ENABLED:
            log.info("AEC disabled via AEC_ENABLED=false")
            return

        try:
            from speexdsp import EchoCanceller  # type: ignore
        except Exception as e:
            log.warning("speexdsp not available, AEC disabled: %s", e)
            return

        self._ec = EchoCanceller.create(FRAME_SAMPLES, FILTER_LENGTH, GEMINI_IN_RATE)
        log.info("AEC enabled (frame=%d filter=%d delay=%dms)",
                 FRAME_SAMPLES, FILTER_LENGTH, AEC_DELAY_MS)

    # ------------------------------------------------------------------
    # Reference (speaker) side
    # ------------------------------------------------------------------

    def push_reference(self, pcm_device: bytes):
        """Feed audio that was just sent to the device speaker.

        `pcm_device` is int16 mono PCM at DEVICE_SPK_RATE.
        """
        if self._ec is None or not pcm_device:
            return

        # Downsample to 16 kHz to match mic rate.
        if DEVICE_SPK_RATE != GEMINI_IN_RATE:
            samples = np.frombuffer(pcm_device, dtype=np.int16).astype(np.float32)
            from math import gcd
            g = gcd(DEVICE_SPK_RATE, GEMINI_IN_RATE)
            up   = GEMINI_IN_RATE // g
            down = DEVICE_SPK_RATE // g
            resampled = resample_poly(samples, up=up, down=down)
            pcm_16k = np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()
        else:
            pcm_16k = pcm_device

        if not self._ref_lock_primed:
            # Pad reference with silence to compensate for playback latency.
            pad_samples = int(GEMINI_IN_RATE * AEC_DELAY_MS / 1000)
            self._ref_buf.extend(b"\x00\x00" * pad_samples)
            self._ref_lock_primed = True

        self._ref_buf.extend(pcm_16k)

    def reset_reference(self):
        """Drop pending reference audio (e.g. on barge-in / interrupt)."""
        self._ref_buf.clear()
        self._ref_lock_primed = False

    # ------------------------------------------------------------------
    # Mic side
    # ------------------------------------------------------------------

    def process_mic(self, pcm_mic_16k: bytes) -> bytes:
        """Run mic audio through the echo canceller, returning cleaned PCM.

        Input/output: 16 kHz mono int16. Pass-through if AEC is disabled.
        """
        if self._ec is None:
            return pcm_mic_16k

        self._mic_buf.extend(pcm_mic_16k)
        out = bytearray()

        while len(self._mic_buf) >= FRAME_BYTES:
            mic_frame = bytes(self._mic_buf[:FRAME_BYTES])
            del self._mic_buf[:FRAME_BYTES]

            if len(self._ref_buf) >= FRAME_BYTES:
                ref_frame = bytes(self._ref_buf[:FRAME_BYTES])
                del self._ref_buf[:FRAME_BYTES]
            else:
                ref_frame = b"\x00\x00" * FRAME_SAMPLES

            try:
                cleaned = self._ec.process(mic_frame, ref_frame)
            except Exception as e:
                log.warning("AEC process error: %s", e)
                cleaned = mic_frame
            out.extend(cleaned)

        return bytes(out)
