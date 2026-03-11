import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from typing import List, Optional

log = logging.getLogger(__name__)


class PetalsWrapper:
    """
    Manages the lifecycle of the Petals server process.
    """

    def __init__(
        self,
        port: int,
        public_ip: Optional[str] = None,
        gpu_vram_limit: float = 0.9,
        device: Optional[str] = None,
    ):
        self.port = port
        self.public_ip = public_ip
        self.gpu_vram_limit = gpu_vram_limit
        self.device = device  # e.g. "cuda:0", "cuda:1"
        self.process: Optional[subprocess.Popen] = None
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._watcher_thread: Optional[threading.Thread] = None
        self._log_buffer: deque[str] = deque(maxlen=200)
        self._last_start_time: Optional[int] = None
        self._last_exit_code: Optional[int] = None
        self._last_exit_time: Optional[int] = None
        self._local_p2p_addrs: List[str] = []

    def _stream_logs(self, stream, label: str):
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                message = line.rstrip()
                self._log_buffer.append(f"[{label}] {message}")
                log.info("Petals %s: %s", label, message)
                # Extract local P2P multiaddrs from "Running a server on [...]"
                if "Running a server on" in message and not self._local_p2p_addrs:
                    # Extract all multiaddr strings (start with /) from the log line
                    addrs = re.findall(r"(/ip[46]/[^\s'\",\]]+)", message)
                    if addrs:
                        self._local_p2p_addrs = addrs
                        log.info("Extracted local P2P addrs: %s", addrs)
        except Exception as exc:
            log.warning("Petals %s log stream error: %s", label, exc)

    def _watch_process(self):
        if not self.process:
            return
        exit_code = self.process.wait()
        self._last_exit_code = exit_code
        self._last_exit_time = int(time.time())
        log.warning("Petals process exited with code %s", exit_code)

    def start(self, model_id: str, block_range: str, initial_peers: Optional[list] = None):
        """
        Starts the Petals server subprocess.
        block_range expected format: "start:end" (e.g., "0:4")
        initial_peers: list of multiaddr strings for beam's DHT bootstrap nodes.
                       If None or empty the petals server uses PUBLIC_INITIAL_PEERS.
        """
        if self.is_running():
            log.warning(
                "Petals process already running. Restarting not implemented yet."
            )
            return

        # Basic CLI construction
        python_exec = os.environ.get("BEAM_PETALS_PYTHON") or sys.executable
        cmd = [
            python_exec,
            "-m",
            "petals.cli.run_server",
            model_id,
            "--port",
            str(self.port),
            "--block_indices",
            block_range,
            "--torch_dtype",
            "float16",  # Default optimization
        ]

        # If the backend provided explicit DHT bootstrap peers, pass them so the
        # petals server joins the beam private swarm instead of the public one.
        if initial_peers:
            cmd.extend(["--initial_peers"] + list(initial_peers))

        # Pin to a specific GPU (e.g. "cuda:1") for multi-GPU machines
        if self.device:
            cmd.extend(["--device", self.device])

        # Add public IP if configured (crucial for p2p)
        if self.public_ip:
            cmd.extend(["--public_ip", self.public_ip])

        # VRAM limit isn't a direct CLI arg for petals usually, it manages its own cache.
        # But for v0 we just pass what we can.
        # (Actually petals manages memory via `num_blocks` or `token_capacity` but let's stick to basics)

        log.info("Starting Petals (python=%s): %s", python_exec, " ".join(cmd))

        try:
            # We redirect stdout/stderr to PIPE to prevent messy console output,
            # but in a real agent we might want to thread-read them and log to file.
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            self._last_start_time = int(time.time())
            self._last_exit_code = None
            self._last_exit_time = None
            if self.process.stdout is not None:
                self._stdout_thread = threading.Thread(
                    target=self._stream_logs, args=(self.process.stdout, "stdout"), daemon=True
                )
                self._stdout_thread.start()
            if self.process.stderr is not None:
                self._stderr_thread = threading.Thread(
                    target=self._stream_logs, args=(self.process.stderr, "stderr"), daemon=True
                )
                self._stderr_thread.start()
            self._watcher_thread = threading.Thread(
                target=self._watch_process, daemon=True
            )
            self._watcher_thread.start()

            time.sleep(0.5)
            if self.process.poll() is not None:
                log.error("Petals exited early with code %s", self.process.returncode)
        except Exception as e:
            log.error(f"Failed to start Petals process: {e}")
            raise

    def stop(self):
        if self.is_running():
            log.info("Stopping Petals process...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                log.warning("Petals process unresponsive, killing...")
                self.process.kill()
        self.process = None
        self._local_p2p_addrs = []

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def local_p2p_addrs(self) -> List[str]:
        """Return the local Petals server's P2P multiaddrs (extracted from logs).

        Loopback addresses (127.x.x.x, ::1) are excluded since they are
        not useful for peer discovery across machines.
        """
        return [
            a for a in self._local_p2p_addrs
            if not a.startswith("/ip4/127.") and not a.startswith("/ip6/::1/")
        ]

    def get_logs(self) -> str:
        # TODO: Implement log tailing from the PIPE or log file
        return ""

    def status_snapshot(self) -> dict:
        now = int(time.time())
        running = self.is_running()
        uptime_sec = None
        if running and self._last_start_time is not None:
            uptime_sec = max(0, now - self._last_start_time)
        return {
            "running": running,
            "started_at": self._last_start_time,
            "uptime_sec": uptime_sec,
            "last_exit_code": self._last_exit_code,
            "last_exit_at": self._last_exit_time,
        }

    def recent_logs(self, limit: int = 50) -> list[str]:
        if limit <= 0:
            return []
        return list(self._log_buffer)[-limit:]
