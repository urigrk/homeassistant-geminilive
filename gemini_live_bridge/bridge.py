#!/usr/bin/env python3
"""
Gemini Live Bridge — Home Assistant Add-on
==========================================
Listens for WebSocket connections from Voice PE devices, relays audio to the
Gemini Live API in real time, and streams Gemini's audio responses back.
"""

import asyncio
import logging
import signal

import websockets
import websockets.server

import config
from session import GeminiSession

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
)
log = logging.getLogger("bridge")

# ---------------------------------------------------------------------------
# WebSocket server
# ---------------------------------------------------------------------------

async def handle_device(ws: websockets.server.ServerConnection, path: str = "/"):
    session = GeminiSession(ws)
    await session.run()

async def main():
    log.info(
        "Gemini Live Bridge  model=%s  voice=%s  port=%d  ha=%s",
        config.GEMINI_MODEL, config.GEMINI_VOICE, config.BRIDGE_PORT, config.HA_BASE_URL,
    )
    log.info("Device control=%s  Camera=%s  Notifications=%s",
             config.ENABLE_DEVICE_CONTROL, config.ENABLE_CAMERA_ACCESS, config.ENABLE_NOTIFICATIONS)

    loop = asyncio.get_running_loop()
    stop = loop.create_future()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set_result, None)

    async with websockets.serve(
        handle_device,
        config.BRIDGE_HOST,
        config.BRIDGE_PORT,
        max_size=2 * 1024 * 1024,
        ping_interval=30,
        ping_timeout=10,
    ):
        log.info("V1.0 - Bridge ready - waiting for Voice PE devices on :%d", config.BRIDGE_PORT)
        await stop

    log.info("Bridge shut down cleanly")

if __name__ == "__main__":
    asyncio.run(main())
