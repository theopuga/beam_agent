import hashlib
import hmac
import json
import logging
import os
import platform
import threading
import time
import uuid
from typing import Optional, Tuple

log = logging.getLogger(__name__)


class NodeIdentity:
    def __init__(self, state_file: str):
        self.state_file = state_file
        self.node_id: Optional[str] = None
        self.node_secret: Optional[str] = None
        self.machine_fingerprint: str = self._generate_fingerprint()
        self._last_timestamp = 0
        self._ts_lock = threading.Lock()

        self._load_state()

    def _generate_fingerprint(self) -> str:
        """
        Generates a semi-stable identifier for this machine.
        For v0, we rely on hostname + platform info + mac address hash.
        """
        node_name = platform.node()
        system = platform.system()
        machine = platform.machine()
        # uuid.getnode() returns mac address as int
        mac_addr = str(uuid.getnode())

        raw = f"{node_name}|{system}|{machine}|{mac_addr}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    self.node_id = data.get("node_id")
                    self.node_secret = data.get("node_secret")
                    log.info(f"Loaded node identity: {self.node_id}")
            except Exception as e:
                log.error(f"Failed to load state file {self.state_file}: {e}")

    def save_state(self, node_id: str, node_secret: str):
        self.node_id = node_id
        self.node_secret = node_secret
        parent_dir = os.path.dirname(self.state_file)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(
                {
                    "node_id": node_id,
                    "node_secret": node_secret,
                    "fingerprint": self.machine_fingerprint,
                },
                f,
                indent=2,
            )
        log.info("Persisted node identity.")

    def sign_request(self, timestamp: int, body_json: str) -> dict:
        """
        Returns headers for HMAC authentication.
        """
        if not self.node_id or not self.node_secret:
            raise ValueError("Cannot sign request: Identity not established")

        body_sha256 = hashlib.sha256(body_json.encode("utf-8")).hexdigest()
        canonical_string = f"{timestamp}\n{body_sha256}"

        signature = hmac.new(
            self.node_secret.encode("utf-8"),
            canonical_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return {
            "X-Node-Id": self.node_id,
            "X-Timestamp": str(timestamp),
            "X-Body-SHA256": body_sha256,
            "X-Signature": signature,
        }

    def next_timestamp(self) -> int:
        """
        Return a monotonically increasing timestamp (seconds) to avoid replay detection.
        """
        now = int(time.time())
        with self._ts_lock:
            if now <= self._last_timestamp:
                now = self._last_timestamp + 1
            self._last_timestamp = now
        return now
