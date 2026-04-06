#!/usr/bin/with-contenv bashio
set -e

bashio::log.info "Starting Gemini Live Bridge..."

# Validate required config
if ! bashio::config.has_value "gemini_api_key"; then
    bashio::log.fatal "gemini_api_key is required — add it in the add-on options."
    exit 1
fi

# Export config values as environment variables consumed by bridge.py
export GEMINI_API_KEY="$(bashio::config 'gemini_api_key')"
export GEMINI_MODEL="$(bashio::config 'model')"
export GEMINI_VOICE="$(bashio::config 'voice')"
export BRIDGE_PORT="$(bashio::config 'bridge_port')"
export SYSTEM_PROMPT="$(bashio::config 'system_prompt')"
export LANGUAGE="$(bashio::config 'language')"
export ENABLE_DEVICE_CONTROL="$(bashio::config 'enable_device_control')"
export ENABLE_CAMERA_ACCESS="$(bashio::config 'enable_camera_access')"
export ENABLE_NOTIFICATIONS="$(bashio::config 'enable_notifications')"
export AUDIO_OUTPUT_RATE="$(bashio::config 'audio_output_rate')"
export LOG_LEVEL="$(bashio::config 'log_level')"

# HA Supervisor provides these automatically
export HA_BASE_URL="http://supervisor/core"
export HA_TOKEN="${SUPERVISOR_TOKEN}"

bashio::log.info "Model: ${GEMINI_MODEL} | Voice: ${GEMINI_VOICE} | Port: ${BRIDGE_PORT}"

exec python3 /app/bridge.py
