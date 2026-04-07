import os

BRIDGE_HOST   = "0.0.0.0"
BRIDGE_PORT   = int(os.environ.get("BRIDGE_PORT", 8765))

# Optional locally during dev, required in production.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
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
