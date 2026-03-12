"""Tor hidden service manager for Beam node agents.

Manages a local Tor process to:
1. Expose the node's pairing server as a .onion hidden service
2. Provide a SOCKS5 proxy for outbound connections (gateway WebSocket)

Requires the ``tor`` binary to be installed on the system.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import stat
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class TorStartupError(RuntimeError):
    """Raised when the Tor process fails to bootstrap."""


class TorManager:
    """Lifecycle manager for a local Tor daemon with a hidden service."""

    def __init__(
        self,
        data_dir: str = "tor_data",
        socks_port: int = 9050,
        hidden_service_port: int = 51337,
        tor_binary: Optional[str] = None,
        bootstrap_timeout: float = 90.0,
    ) -> None:
        self._data_dir = os.path.abspath(data_dir)
        self._socks_port = socks_port
        self._hs_port = hidden_service_port
        self._tor_binary = tor_binary or self._find_tor_binary()
        self._bootstrap_timeout = bootstrap_timeout
        self._process: Optional[asyncio.subprocess.Process] = None
        self._onion_address: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def onion_address(self) -> Optional[str]:
        """The .onion hostname, available after ``start()`` completes."""
        return self._onion_address

    @property
    def socks_port(self) -> int:
        return self._socks_port

    def socks_proxy_url(self) -> str:
        """SOCKS5 proxy URL suitable for aiohttp-socks."""
        return f"socks5://127.0.0.1:{self._socks_port}"

    async def start(self) -> str:
        """Start the Tor daemon and return the .onion address.

        Blocks until Tor has fully bootstrapped or *bootstrap_timeout* elapses.
        """
        if not self._tor_binary:
            raise TorStartupError(
                "Tor binary not found. Install Tor: "
                "Linux: apt install tor | macOS: brew install tor | "
                "Windows: download Tor Expert Bundle"
            )

        self._prepare_data_dir()
        torrc_path = self._write_torrc()

        log.info(
            "Starting Tor (binary=%s, socks=%d, hs_port=%d) ...",
            self._tor_binary,
            self._socks_port,
            self._hs_port,
        )

        self._process = await asyncio.create_subprocess_exec(
            self._tor_binary,
            "-f",
            torrc_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        await self._wait_for_bootstrap()

        self._onion_address = self._read_onion_hostname()
        log.info("Tor hidden service ready: %s", self._onion_address)
        return self._onion_address

    async def stop(self) -> None:
        """Gracefully terminate the Tor process."""
        if self._process and self._process.returncode is None:
            log.info("Stopping Tor daemon ...")
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=10)
            except asyncio.TimeoutError:
                log.warning("Tor did not exit gracefully, killing")
                self._process.kill()
                await self._process.wait()
        self._process = None

    async def health_check(self) -> bool:
        """Return True if the Tor process is still running."""
        return self._process is not None and self._process.returncode is None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_tor_binary() -> Optional[str]:
        return shutil.which("tor")

    def _prepare_data_dir(self) -> None:
        hs_dir = os.path.join(self._data_dir, "hidden_service")
        os.makedirs(hs_dir, exist_ok=True)
        # Tor requires 0700 on the hidden service directory (Unix only)
        if os.name != "nt":
            os.chmod(hs_dir, stat.S_IRWXU)

    def _write_torrc(self) -> str:
        hs_dir = os.path.join(self._data_dir, "hidden_service")
        torrc_path = os.path.join(self._data_dir, "torrc")

        lines = [
            f"SocksPort {self._socks_port}",
            f"DataDirectory {self._data_dir}",
            f"HiddenServiceDir {hs_dir}",
            f"HiddenServicePort 80 127.0.0.1:{self._hs_port}",
            # Suppress warning about missing control port
            "ControlPort 0",
            "Log notice stdout",
        ]

        with open(torrc_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        return torrc_path

    async def _wait_for_bootstrap(self) -> None:
        """Read Tor stdout until 'Bootstrapped 100%' or timeout."""
        assert self._process is not None
        assert self._process.stdout is not None

        try:
            async with asyncio.timeout(self._bootstrap_timeout):
                while True:
                    line_bytes = await self._process.stdout.readline()
                    if not line_bytes:
                        rc = await self._process.wait()
                        raise TorStartupError(
                            f"Tor process exited prematurely (exit code {rc})"
                        )
                    line = line_bytes.decode("utf-8", errors="replace").strip()
                    log.debug("tor: %s", line)
                    if "Bootstrapped 100%" in line:
                        # Start a background task to drain remaining stdout
                        # so the pipe buffer doesn't fill up and block Tor.
                        asyncio.create_task(self._drain_stdout())
                        return
        except TimeoutError:
            await self.stop()
            raise TorStartupError(
                f"Tor failed to bootstrap within {self._bootstrap_timeout}s"
            )

    async def _drain_stdout(self) -> None:
        """Keep reading Tor stdout to prevent pipe blocking."""
        try:
            assert self._process is not None and self._process.stdout is not None
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").strip()
                if decoded:
                    log.debug("tor: %s", decoded)
        except Exception:
            pass

    def _read_onion_hostname(self) -> str:
        hostname_path = os.path.join(
            self._data_dir, "hidden_service", "hostname"
        )
        try:
            hostname = Path(hostname_path).read_text().strip()
        except FileNotFoundError:
            raise TorStartupError(
                f"Tor did not create hostname file at {hostname_path}. "
                "Check Tor logs for errors."
            )
        if not hostname.endswith(".onion"):
            raise TorStartupError(f"Unexpected hostname format: {hostname}")
        return hostname
