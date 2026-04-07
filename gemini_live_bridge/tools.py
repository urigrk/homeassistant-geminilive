from config import (
    ENABLE_DEVICE_CONTROL,
    ENABLE_CAMERA_ACCESS,
    ENABLE_NOTIFICATIONS,
    GEMINI_MODEL,
    GEMINI_VOICE,
    LANGUAGE,
)
from prompts import SYSTEM_PROMPT

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
