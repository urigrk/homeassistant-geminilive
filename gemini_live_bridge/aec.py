"""
Acoustic Echo Cancellation — pure-Python frequency-domain adaptive filter.

The device's microphone picks up its own speaker output (Gemini's voice),
which causes Gemini to hear itself. This module subtracts an estimate of
the speaker signal (as it appears in the mic) from the mic stream before
forwarding to Gemini.

Algorithm: partitioned-block frequency-domain NLMS (PBFDAF), the same
family used inside Speex / WebRTC AEC. We model the room/echo path as a
linear FIR filter of length ~200 ms (3200 taps at 16 kHz), split into
20 blocks of 160 samples (10 ms each). On every 10 ms frame:

  1. Form the next 2N-sample reference block, FFT it, push it onto a
     ring of K=20 past spectra.
  2. Estimate the echo as the sum of W[k] * X[k] across partitions
     (frequency-domain convolution via overlap-save).
  3. Compute the error e = mic - echo_estimate.
  4. Update each W[k] by the normalized gradient
     conj(X[k]) * E / (sum_k |X[k]|^2 + delta).
  5. Constrain W[k] to be causal (zero its second half in the time
     domain) — this is the "constrained" PBFDAF and is essential for
     stability with long filters.

Double-talk handling: a simple energy-based detector freezes adaptation
(but NOT cancellation) when the near-end speaker dominates, so the
filter does not diverge while the user is talking.

Timing: a fixed AEC_DELAY_MS of silence is prepended to the reference
queue to compensate for the speaker's playback look-ahead. The adaptive
filter then absorbs the remaining (room + small jitter) delay, up to
the 200 ms tail length.

Both mic and reference are processed at 16 kHz mono int16 in fixed
10 ms frames. Pass-through if AEC_ENABLED=false.
"""

import logging
import os
from math import gcd

import numpy as np
from scipy.signal import resample_poly

from config import DEVICE_SPK_RATE, GEMINI_IN_RATE

log = logging.getLogger(__name__)

# 10 ms frames at 16 kHz
FRAME_SAMPLES = 160
FRAME_BYTES   = FRAME_SAMPLES * 2

# ~320 ms echo tail. 32 partitions of 10 ms each. Generous to absorb
# ESP32 playback jitter on top of the static AEC_DELAY_MS offset.
NUM_PARTITIONS = 32
FFT_SIZE       = 2 * FRAME_SAMPLES   # 320
NUM_BINS       = FRAME_SAMPLES + 1   # rfft length

# NLMS step size. Larger = faster convergence, less stable. 0.7 is the
# upper edge of "safe" for a constrained PBFDAF.
MU = 0.7

# Per-bin power-smoothing time constant for the NLMS denominator.
P_SMOOTH = 0.9

# Leakage on filter weights — very slow forgetting so the filter
# adapts to room changes over minutes but does not decay during a turn.
LEAK = 0.99995

# Adapt only when the reference (Gemini's voice) is energetic enough
# to actually drive the room/echo path. Below this, there is nothing
# to learn and adapting on noise would corrupt W.
REF_ACTIVE_POW = 1e-4   # ≈ -40 dBFS on a [-1, 1] frame

# Double-talk: if mic energy is much greater than reference energy,
# the near-end speaker is dominant and we freeze adaptation. The
# subtraction (cancellation) still runs every frame regardless.
DTD_MIC_TO_REF = 4.0

AEC_ENABLED  = os.environ.get("AEC_ENABLED", "true").lower() == "true"
AEC_DELAY_MS = int(os.environ.get("AEC_DELAY_MS", "80"))


class _FDAF:
    """Constrained partitioned-block frequency-domain NLMS."""

    def __init__(self):
        self.W = np.zeros((NUM_PARTITIONS, NUM_BINS), dtype=np.complex64)
        self.X = np.zeros((NUM_PARTITIONS, NUM_BINS), dtype=np.complex64)
        self.ref_prev = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        self.P = np.full(NUM_BINS, 1e-3, dtype=np.float64)
        # Running ERLE estimate (mic_pow / err_pow, dB) for logging.
        self._erle_mic = 1e-9
        self._erle_err = 1e-9
        self._erle_log_count = 0

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Process one 10 ms frame. Inputs are float32 in [-1, 1]."""
        # 1) FFT the new 2N reference block (overlap-save).
        x_block = np.concatenate((self.ref_prev, ref))
        self.ref_prev = ref
        X_new = np.fft.rfft(x_block).astype(np.complex64)

        # Shift partition ring so X[0] is the most recent block.
        self.X = np.roll(self.X, 1, axis=0)
        self.X[0] = X_new

        # 2) Echo estimate: sum_k W[k] * X[k] (frequency domain).
        Y = np.sum(self.W * self.X, axis=0)
        y_full = np.fft.irfft(Y, n=FFT_SIZE).astype(np.float32)
        echo = y_full[FRAME_SAMPLES:]  # overlap-save: keep last N

        # 3) Error in time domain (cancellation — always runs).
        e = mic - echo

        # ERLE accounting for periodic logging.
        mic_pow = float(np.dot(mic, mic))
        err_pow = float(np.dot(e, e))
        ref_pow = float(np.dot(ref, ref))
        self._erle_mic = 0.95 * self._erle_mic + 0.05 * mic_pow
        self._erle_err = 0.95 * self._erle_err + 0.05 * err_pow
        self._erle_log_count += 1
        if self._erle_log_count >= 200:  # ~2 s
            self._erle_log_count = 0
            erle_db = 10.0 * np.log10((self._erle_mic + 1e-12) /
                                      (self._erle_err + 1e-12))
            log.debug("AEC ERLE ~%.1f dB (mic=%.4f err=%.4f)",
                      erle_db, self._erle_mic, self._erle_err)

        # 4) Adapt only when the reference is active and the mic is not
        #    dominated by near-end speech. Cancellation always runs.
        if ref_pow < REF_ACTIVE_POW:
            return e
        if mic_pow > DTD_MIC_TO_REF * ref_pow:
            return e

        # 5) Per-bin power normalization across all partitions.
        Px = np.sum((self.X.real ** 2 + self.X.imag ** 2), axis=0)
        self.P = P_SMOOTH * self.P + (1.0 - P_SMOOTH) * Px
        norm = (1.0 / (self.P + 1e-3)).astype(np.float32)

        # 6) Zero-padded error spectrum (overlap-save gradient: zeros in
        #    the FIRST half, error in the SECOND).
        e_padded = np.empty(FFT_SIZE, dtype=np.float32)
        e_padded[:FRAME_SAMPLES] = 0.0
        e_padded[FRAME_SAMPLES:] = e
        E = np.fft.rfft(e_padded).astype(np.complex64)

        # 7) Vectorized NLMS update across all partitions.
        grad = np.conj(self.X) * (MU * norm * E)[None, :]
        W_new = self.W + grad

        # 8) Causal constraint: in time domain, zero the SECOND half of
        #    each partition's impulse response, then re-FFT. This is
        #    what makes long-tail PBFDAF stable.
        w_time = np.fft.irfft(W_new, n=FFT_SIZE, axis=1)
        w_time[:, FRAME_SAMPLES:] = 0.0
        self.W = (LEAK * np.fft.rfft(w_time, axis=1)).astype(np.complex64)

        return e


class EchoCanceler:
    def __init__(self):
        self._fdaf = None
        self._mic_buf = bytearray()
        self._ref_buf = bytearray()  # 16 kHz mono int16
        self._ref_primed = False

        if not AEC_ENABLED:
            log.info("AEC disabled via AEC_ENABLED=false")
            return

        self._fdaf = _FDAF()
        log.info("AEC enabled (PBFDAF, taps=%d, delay=%dms)",
                 NUM_PARTITIONS * FRAME_SAMPLES, AEC_DELAY_MS)

    # ------------------------------------------------------------------
    # Reference (speaker) side
    # ------------------------------------------------------------------

    def push_reference(self, pcm_device: bytes):
        """Feed audio that was just sent to the device speaker.

        `pcm_device` is int16 mono PCM at DEVICE_SPK_RATE.
        """
        if self._fdaf is None or not pcm_device:
            return

        # Downsample to 16 kHz to match mic rate.
        if DEVICE_SPK_RATE != GEMINI_IN_RATE:
            samples = np.frombuffer(pcm_device, dtype=np.int16).astype(np.float32)
            g = gcd(DEVICE_SPK_RATE, GEMINI_IN_RATE)
            up   = GEMINI_IN_RATE // g
            down = DEVICE_SPK_RATE // g
            resampled = resample_poly(samples, up=up, down=down)
            pcm_16k = np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()
        else:
            pcm_16k = pcm_device

        if not self._ref_primed:
            # Prepend silence so the reference lines up with mic playback.
            pad_samples = int(GEMINI_IN_RATE * AEC_DELAY_MS / 1000)
            self._ref_buf.extend(b"\x00\x00" * pad_samples)
            self._ref_primed = True

        self._ref_buf.extend(pcm_16k)

    def reset_reference(self):
        """Drop pending reference audio (e.g. on barge-in / interrupt)."""
        self._ref_buf.clear()
        self._ref_primed = False

    # ------------------------------------------------------------------
    # Mic side
    # ------------------------------------------------------------------

    def process_mic(self, pcm_mic_16k: bytes) -> bytes:
        """Run mic audio through the echo canceller, returning cleaned PCM.

        Input/output: 16 kHz mono int16. Pass-through if AEC is disabled.
        """
        if self._fdaf is None:
            return pcm_mic_16k

        self._mic_buf.extend(pcm_mic_16k)
        out = bytearray()

        while len(self._mic_buf) >= FRAME_BYTES:
            mic_bytes = bytes(self._mic_buf[:FRAME_BYTES])
            del self._mic_buf[:FRAME_BYTES]

            if len(self._ref_buf) >= FRAME_BYTES:
                ref_bytes = bytes(self._ref_buf[:FRAME_BYTES])
                del self._ref_buf[:FRAME_BYTES]
            else:
                ref_bytes = b"\x00" * FRAME_BYTES

            mic = np.frombuffer(mic_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            ref = np.frombuffer(ref_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            try:
                cleaned = self._fdaf.process(mic, ref)
            except Exception as e:
                log.warning("AEC process error: %s", e)
                cleaned = mic

            cleaned_i16 = np.clip(cleaned * 32768.0, -32768, 32767).astype(np.int16)
            out.extend(cleaned_i16.tobytes())

        return bytes(out)
