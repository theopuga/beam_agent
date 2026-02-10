import asyncio
import hashlib
import hmac
import json
import logging
import platform
import time
import uuid
from typing import Any, Dict, Optional

import aiohttp

from .config import CONTROL_PLANE_URL, HEARTBEAT_INTERVAL_SEC
from .petals_wrapper import PetalsWrapper

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("BeamNodeAgent")


class BeamNodeAgent:
    def __init__(self):
        self.node_id: Optional[str] = None
        self.node_secret: Optional[str] = None
        self.petals = PetalsWrapper(port=31337)
        self.session: Optional[aiohttp.ClientSession] = None
        self.current_assignment: Optional[Dict] = None

    async def start(self):
        self.session = aiohttp.ClientSession()

        # 1. Register
        await self._register()

        # 2. Main Loop
        try:
            while True:
                await self._heartbeat()
                await self._check_assignment()
                await asyncio.sleep(HEARTBEAT_INTERVAL_SEC)
        except asyncio.CancelledError:
            log.info("Agent stopping...")
        finally:
            self.petals.stop()
            await self.session.close()

    async def _register(self):
        log.info(f"Registering with Control Plane at {CONTROL_PLANE_URL}...")

        fingerprint = hashlib.sha256(platform.node().encode()).hexdigest()

        payload = {
            "machine_fingerprint": fingerprint,
            "gpu": {
                "name": "Generic GPU",  # Placeholder, would use pynvml
                "vram_gb": 24.0,
                "count": 1,
            },
            "software": {"node_agent_version": "0.1.0", "petals_version": "2.3.0"},
            "transports": ["http"],
            "capabilities": {},
        }

        try:
            async with self.session.post(
                f"{CONTROL_PLANE_URL}/api/v1/nodes/register", json=payload
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    log.error(f"Registration failed: {resp.status} - {text}")
                    raise Exception("Registration failed")

                data = await resp.json()
                self.node_id = data["node_id"]
                self.node_secret = data["node_secret"]
                log.info(f"Registered successfully. Node ID: {self.node_id}")

                # Handle initial assignment
                if "assignment" in data:
                    await self._apply_assignment(data["assignment"])

        except Exception as e:
            log.error(f"Failed to connect to Control Plane: {e}")
            raise

    async def _heartbeat(self):
        if not self.node_id or not self.node_secret:
            return

        timestamp = int(time.time())
        status = "healthy" if self.petals.is_running() else "idle"

        payload = {
            "protocol_version": "1.0",
            "node_id": self.node_id,
            "timestamp": timestamp,
            "status": status,
            "metrics": {"gpu_util": 0.5},  # Stub
            "active_jobs": [],
            "current_assignment": self.current_assignment or {},
        }

        # Sign Request
        body_json = json.dumps(payload)
        # Canonical hash of body (simple sha256 of json string for v0)
        # Ideally should satisfy exact byte matching
        body_sha256 = hashlib.sha256(body_json.encode("utf-8")).hexdigest()

        canonical_string = f"{timestamp}\n{body_sha256}"
        signature = hmac.new(
            self.node_secret.encode("utf-8"),
            canonical_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        headers = {
            "X-Node-Id": self.node_id,
            "X-Timestamp": str(timestamp),
            "X-Body-SHA256": body_sha256,
            "X-Signature": signature,
            "Content-Type": "application/json",
        }

        try:
            async with self.session.post(
                f"{CONTROL_PLANE_URL}/api/v1/nodes/heartbeat",
                data=body_json,
                headers=headers,
            ) as resp:
                if resp.status != 200:
                    log.warning(f"Heartbeat rejected: {resp.status}")
                else:
                    log.debug("Heartbeat ack")
        except Exception as e:
            log.error(f"Heartbeat failed: {e}")

    async def _check_assignment(self):
        # In v0, assignment updates usually come via heartbeat response or separate polling
        # We implemented a dedicated endpoint GET /assignment in M1
        pass

    async def _apply_assignment(self, assignment: Dict):
        self.current_assignment = assignment
        model_id = assignment.get("model_id")
        block_range = assignment.get("block_range")

        if model_id and block_range:
            # range is list [start, end]
            range_str = f"{block_range[0]}:{block_range[1]}"
            # Start Petals
            self.petals.start(model_id, range_str)


if __name__ == "__main__":
    agent = BeamNodeAgent()
    try:
        asyncio.run(agent.start())
    except KeyboardInterrupt:
        pass
