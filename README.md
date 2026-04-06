# Gemini Live Bridge — Home Assistant Add-on

Connects a **Home Assistant Voice PE** (or any ESP32-based voice device) to the
**Gemini Live API** with full smart-home integration: device control, sensor
reading, camera access, and more.

---

## Installation

1. In Home Assistant, go to **Settings → Add-ons → Add-on Store**.
2. Click the menu (⋮) in the top right and select **Repositories**.
3. Add the URL of this repository and click **Add**.
4. Find **Gemini Live Bridge** and click **Install**.

---

## Configuration

| Option | Default | Description |
|---|---|---|
| `gemini_api_key` | *(required)* | Your Google AI / Gemini API key |
| `model` | `gemini-2.0-flash-live-001` | Gemini Live model to use |
| `voice` | `Aoede` | Voice: Aoede, Charon, Fenrir, Kore, Puck |
| `bridge_port` | `8765` | WebSocket port Voice PE connects to |
| `system_prompt` | *(built-in)* | Override the assistant's personality & instructions |
| `language` | `en-US` | Language/locale for speech recognition |
| `enable_device_control` | `true` | Allow Gemini to control HA devices |
| `enable_camera_access` | `true` | Allow Gemini to view camera snapshots |
| `enable_notifications` | `true` | Allow Gemini to create HA notifications |
| `audio_output_rate` | `48000` | Speaker sample rate (match your device) |
| `log_level` | `info` | Logging verbosity |

---

## Voice PE device setup

Point your Voice PE firmware at this add-on's WebSocket server:

```
ws://<your-ha-ip>:8765
```

The wire protocol is unchanged from the original bridge — no firmware changes needed.

---

## What Gemini can do

| Capability | Example utterances |
|---|---|
| **Read sensors** | "What's the temperature outside?" / "Is the front door open?" |
| **Control lights** | "Turn off all the lights" / "Set the bedroom to 50% brightness" |
| **Climate control** | "Set the thermostat to 22 degrees" / "Turn on the AC" |
| **Covers & locks** | "Close the garage" / "Lock the front door" |
| **Media players** | "Pause the kitchen speaker" / "Set volume to 30%" |
| **Cameras** | "Check the front door camera" / "Is anyone in the backyard?" |
| **Scenes & scripts** | "Activate movie night" / "Run the good morning routine" |
| **History** | "Was the washing machine running today?" |
| **Notifications** | "Remind me on the dashboard to take out the bins" |

---

## Security notes

- The add-on uses the **Supervisor token** for HA API access — no long-lived token needed.
- Camera access and device control can be individually disabled in options.
- The bridge only accepts connections on your local network (no external exposure unless you configure it).

---

## Customising the system prompt

Leave `system_prompt` blank to use the built-in personality. To customise, paste
your prompt into the option — keep it concise (voice output only, no markdown).
