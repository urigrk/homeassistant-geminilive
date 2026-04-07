import numpy as np
from scipy.signal import resample_poly

from config import DEVICE_SPK_RATE, GEMINI_OUT_RATE

def resample_gemini_to_device(pcm_bytes: bytes) -> bytes:
    """Resample Gemini's 24 kHz int16 PCM output to the device speaker rate."""
    if GEMINI_OUT_RATE == DEVICE_SPK_RATE:
        return pcm_bytes
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    # Compute GCD-reduced up/down ratio
    from math import gcd
    g = gcd(DEVICE_SPK_RATE, GEMINI_OUT_RATE)
    up   = DEVICE_SPK_RATE // g
    down = GEMINI_OUT_RATE  // g
    resampled = resample_poly(samples, up=up, down=down)
    return resampled.astype(np.int16).tobytes()
