"""
Acoustic Echo Cancellation using webrtc-audio-processing.

The device's microphone picks up its own speaker output (Gemini's voice),
which causes Gemini to hear itself. This module feeds a reference of what
we sent to the speaker into WebRTC's APM, which then cancels the echo from
the mic stream before forwarding to Gemini.

Both mic and reference are processed at 16 kHz mono int16 in fixed-size
10 ms frames. A small initial delay is inserted into the reference queue
to compensate for the network + ESP32 playback look-ahead buffer.
"""

import logging
import os

import numpy as np
from scipy.signal import resample_poly

from config import DEVICE_SPK_RATE, GEMINI_IN_RATE

log = logging.getLogger(__name__)

# WebRTC APM requires 10 ms frames
FRAME_SAMPLES = 160
FRAME_BYTES   = FRAME_SAMPLES * 2

AEC_ENABLED  = os.environ.get("AEC_ENABLED", "true").lower() == "true"
AEC_DELAY_MS = int(os.environ.get("AEC_DELAY_MS", "80"))


class EchoCanceler:
    def __init__(self):
        self._ap = None
        self._mic_buf = bytearray()
        self._ref_buf = bytearray()  # 16 kHz mono int16
        self._ref_lock_primed = False

        if not AEC_ENABLED:
            log.info("AEC disabled via AEC_ENABLED=false")
            return

        try:
            from webrtc_audio_processing import AudioProcessingModule as AP  # type: ignore
        except Exception as e:
            log.warning("webrtc-audio-processing not available, AEC disabled: %s", e)
            return

        # aec_type=2 selects the desktop AEC; enable_ns turns on noise suppression.
        ap = AP(2, True, 0, False)
        ap.set_stream_format(GEMINI_IN_RATE, 1)
        ap.set_reverse_stream_format(GEMINI_IN_RATE, 1)
        ap.set_system_delay(AEC_DELAY_MS)
        try:
            ap.set_ns_level(2)
            ap.set_aec_level(2)
        except Exception:
            pass
        self._ap = ap
        log.info("AEC enabled via webrtc-apm (frame=%d delay=%dms)",
                 FRAME_SAMPLES, AEC_DELAY_MS)

    # ------------------------------------------------------------------
    # Reference (speaker) side
    # ------------------------------------------------------------------

    def push_reference(self, pcm_device: bytes):
        """Feed audio that was just sent to the device speaker.

        `pcm_device` is int16 mono PCM at DEVICE_SPK_RATE.
        """
        if self._ap is None or not pcm_device:
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
        if self._ap is None:
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
                self._ap.process_reverse_stream(ref_frame)
                cleaned = self._ap.process_stream(mic_frame)
            except Exception as e:
                log.warning("AEC process error: %s", e)
                cleaned = mic_frame
            out.extend(cleaned)

        return bytes(out)
