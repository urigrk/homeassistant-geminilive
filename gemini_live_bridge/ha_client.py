import base64
import json
import logging
from typing import Optional

import aiohttp

from config import HA_BASE_URL, HA_HEADERS

log = logging.getLogger(__name__)

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
