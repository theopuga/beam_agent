"""Live agent dashboard — launched after setup or when config already exists."""
from __future__ import annotations

import asyncio
import os
import signal
import sys
from datetime import datetime, timezone
from typing import Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, RichLog, Static

from beam_node_agent.tui.log_parser import EventKind, parse

_MAX_LOG_LINES = 500


# ---------------------------------------------------------------------------
# Messages posted from the subprocess reader task to the screen
# ---------------------------------------------------------------------------
class LogLine(Message):
    def __init__(self, line: str) -> None:
        super().__init__()
        self.line = line


class AgentExited(Message):
    def __init__(self, return_code: int) -> None:
        super().__init__()
        self.return_code = return_code


# ---------------------------------------------------------------------------
# Status pill widget
# ---------------------------------------------------------------------------
class StatusPill(Static):
    """A coloured status indicator."""

    _STYLES = {
        "waiting":    ("dim", "Waiting…"),
        "pairing":    ("yellow", "Awaiting pairing"),
        "connected":  ("green", "Connected"),
        "error":      ("red", "Error"),
        "exited":     ("dim", "Agent exited"),
    }

    status: reactive[str] = reactive("waiting")

    def render(self) -> str:
        colour, label = self._STYLES.get(self.status, ("dim", self.status))
        return f"[{colour}]●[/{colour}] {label}"


# ---------------------------------------------------------------------------
# Dashboard screen
# ---------------------------------------------------------------------------
class DashboardScreen(Screen):
    BINDINGS = [
        Binding("q", "quit_agent", "Quit"),
        Binding("r", "restart_agent", "Restart"),
    ]

    CSS = """
    DashboardScreen {
        layout: vertical;
    }

    #top-row {
        height: auto;
        layout: horizontal;
    }

    #status-panel {
        width: 1fr;
        height: auto;
        border: round $primary;
        padding: 1;
        margin: 0 1 0 0;
    }

    #assignment-panel {
        width: 2fr;
        height: auto;
        border: round $primary;
        padding: 1;
    }

    #pairing-panel {
        height: auto;
        border: round $warning;
        padding: 1;
        margin: 1 0;
        text-align: center;
        display: none;
    }

    #pairing-panel.visible {
        display: block;
    }

    #pairing-code-label {
        text-align: center;
        text-style: bold;
        color: $warning;
        font-size: 2;
    }

    #pairing-hint {
        text-align: center;
        color: $text-muted;
    }

    #stats-row {
        height: auto;
        layout: horizontal;
        margin-bottom: 1;
    }

    .stat-box {
        width: 1fr;
        border: round $panel;
        padding: 0 1;
        height: auto;
        margin-right: 1;
    }

    .stat-box:last-child {
        margin-right: 0;
    }

    #log-panel {
        height: 1fr;
        border: round $panel;
        padding: 0;
    }

    #log-title {
        background: $panel;
        padding: 0 1;
        color: $text-muted;
    }
    """

    # reactive state driven by log parsing
    status: reactive[str] = reactive("waiting")
    pairing_code: reactive[str] = reactive("")
    pairing_expires: reactive[str] = reactive("")
    assignment_model: reactive[str] = reactive("—")
    assignment_blocks: reactive[str] = reactive("—")
    jobs_active: reactive[int] = reactive(0)
    jobs_done: reactive[int] = reactive(0)
    petals_status: reactive[str] = reactive("—")
    _active_job_ids: set[str]

    def __init__(self, config_path: str) -> None:
        super().__init__()
        self._config_path = config_path
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._active_job_ids = set()
        self._pairing_expire_dt: Optional[datetime] = None
        self._countdown_handle = None

    # ------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="top-row"):
            with Vertical(id="status-panel"):
                yield Label("[b]Status[/b]")
                yield StatusPill(id="status-pill")
                yield Label("", id="petals-label")
            with Vertical(id="assignment-panel"):
                yield Label("[b]Assignment[/b]")
                yield Label("", id="model-label")
                yield Label("", id="blocks-label")

        with Vertical(id="pairing-panel"):
            yield Label("Enter this code in the Rent Panel to link your machine", id="pairing-hint")
            yield Label("", id="pairing-code-label")
            yield Label("", id="pairing-expires-label")

        with Horizontal(id="stats-row"):
            with Vertical(classes="stat-box"):
                yield Label("[b]Active jobs[/b]")
                yield Label("0", id="jobs-active-label")
            with Vertical(classes="stat-box"):
                yield Label("[b]Completed jobs[/b]")
                yield Label("0", id="jobs-done-label")

        with Vertical(id="log-panel"):
            yield Label(" LOGS ", id="log-title")
            yield RichLog(id="log-view", highlight=True, markup=True, wrap=False)

        yield Footer()

    def on_mount(self) -> None:
        self.set_interval(1, self._tick_pairing_countdown)
        self.call_after_refresh(self._start_agent)

    # ------------------------------------------------------------------
    # Subprocess management
    # ------------------------------------------------------------------
    async def _start_agent(self) -> None:
        cmd = _build_agent_cmd(self._config_path)
        self._append_log(f"[dim]Starting: {' '.join(cmd)}[/dim]")
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ},
            )
        except Exception as exc:
            self._append_log(f"[red]Failed to start agent: {exc}[/red]")
            self.status = "error"
            return

        self._reader_task = asyncio.create_task(self._read_output())

    async def _read_output(self) -> None:
        assert self._proc and self._proc.stdout
        while True:
            raw = await self._proc.stdout.readline()
            if not raw:
                break
            line = raw.decode(errors="replace").rstrip()
            self.post_message(LogLine(line))
        rc = await self._proc.wait()
        self.post_message(AgentExited(rc))

    async def _stop_agent(self) -> None:
        if self._proc and self._proc.returncode is None:
            try:
                if sys.platform == "win32":
                    self._proc.terminate()
                else:
                    self._proc.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
        if self._reader_task:
            self._reader_task.cancel()

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------
    def on_log_line(self, message: LogLine) -> None:
        line = message.line
        self._append_log(line)
        event = parse(line)

        if event.kind == EventKind.PAIRING_CODE:
            self.pairing_code = event.pairing_code or ""
            self.pairing_expires = event.pairing_expires or ""
            self._try_parse_expiry(self.pairing_expires)
            self.status = "pairing"
            self._show_pairing_panel(True)

        elif event.kind == EventKind.PAIRING_EXPIRED:
            self._show_pairing_panel(False)

        elif event.kind == EventKind.CONNECTED:
            self.status = "connected"
            self._show_pairing_panel(False)

        elif event.kind == EventKind.DISCONNECTED:
            if self.status == "connected":
                self.status = "waiting"

        elif event.kind == EventKind.ASSIGNMENT:
            self.assignment_model = event.assignment_model or "—"
            self.assignment_blocks = event.assignment_blocks or "—"

        elif event.kind == EventKind.JOB_START:
            if event.job_id:
                self._active_job_ids.add(event.job_id)
            self.jobs_active = len(self._active_job_ids)

        elif event.kind == EventKind.JOB_DONE:
            if event.job_id:
                self._active_job_ids.discard(event.job_id)
            self.jobs_active = len(self._active_job_ids)
            self.jobs_done += 1

        elif event.kind == EventKind.PETALS_READY:
            self.petals_status = "Ready"

        elif event.kind == EventKind.PETALS_EXIT:
            self.petals_status = f"Exited ({event.petals_exit_code})"

        elif event.kind == EventKind.FATAL:
            self.status = "error"

    def on_agent_exited(self, message: AgentExited) -> None:
        self.status = "exited"
        self._append_log(
            f"[yellow]Agent process exited with code {message.return_code}[/yellow]"
        )

    # ------------------------------------------------------------------
    # Reactive watchers — keep widgets in sync
    # ------------------------------------------------------------------
    def watch_status(self, value: str) -> None:
        pill = self.query_one("#status-pill", StatusPill)
        pill.status = value

    def watch_pairing_code(self, value: str) -> None:
        spaced = "  ".join(value) if value else ""
        self.query_one("#pairing-code-label", Label).update(
            f"[bold yellow]{spaced}[/bold yellow]"
        )

    def watch_pairing_expires(self, value: str) -> None:
        self.query_one("#pairing-expires-label", Label).update(
            f"[dim]Expires: {value}[/dim]" if value else ""
        )

    def watch_assignment_model(self, value: str) -> None:
        self.query_one("#model-label", Label).update(f"[b]Model:[/b] {value}")

    def watch_assignment_blocks(self, value: str) -> None:
        self.query_one("#blocks-label", Label).update(f"[b]Blocks:[/b] {value}")

    def watch_jobs_active(self, value: int) -> None:
        self.query_one("#jobs-active-label", Label).update(str(value))

    def watch_jobs_done(self, value: int) -> None:
        self.query_one("#jobs-done-label", Label).update(str(value))

    def watch_petals_status(self, value: str) -> None:
        self.query_one("#petals-label", Label).update(f"Petals: {value}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _append_log(self, line: str) -> None:
        log_view = self.query_one("#log-view", RichLog)
        log_view.write(line)

    def _show_pairing_panel(self, visible: bool) -> None:
        panel = self.query_one("#pairing-panel")
        if visible:
            panel.add_class("visible")
        else:
            panel.remove_class("visible")

    def _try_parse_expiry(self, expires_str: str) -> None:
        """Parse the expiry string into a datetime for countdown."""
        try:
            self._pairing_expire_dt = datetime.fromisoformat(expires_str)
        except ValueError:
            self._pairing_expire_dt = None

    def _tick_pairing_countdown(self) -> None:
        if not self._pairing_expire_dt:
            return
        now = datetime.now(tz=timezone.utc)
        exp = self._pairing_expire_dt
        if exp.tzinfo is None:
            from datetime import timezone as tz
            exp = exp.replace(tzinfo=tz.utc)
        remaining = exp - now
        secs = int(remaining.total_seconds())
        if secs <= 0:
            self.query_one("#pairing-expires-label", Label).update("[dim]Expired[/dim]")
            self._pairing_expire_dt = None
        else:
            m, s = divmod(secs, 60)
            self.query_one("#pairing-expires-label", Label).update(
                f"[dim]Expires in: {m}:{s:02d}[/dim]"
            )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    async def action_quit_agent(self) -> None:
        await self._stop_agent()
        self.app.exit()

    async def action_restart_agent(self) -> None:
        self._append_log("[dim]--- Restarting agent ---[/dim]")
        await self._stop_agent()
        self.status = "waiting"
        self._active_job_ids.clear()
        self.jobs_active = 0
        self.petals_status = "—"
        self._pairing_expire_dt = None
        self._show_pairing_panel(False)
        await self._start_agent()

    async def on_unmount(self) -> None:
        await self._stop_agent()


# ---------------------------------------------------------------------------
# Build the agent command to run
# ---------------------------------------------------------------------------
def _build_agent_cmd(config_path: str) -> list[str]:
    """Return the command list to launch the agent.

    When BEAM_AGENT_BIN is set (e.g. by install.sh for the pre-built binary),
    that binary is used directly.  Otherwise the agent is run from the current
    Python interpreter (source / editable install).
    """
    binary = os.environ.get("BEAM_AGENT_BIN", "").strip()
    if binary:
        return [binary, "--config", config_path]
    return [sys.executable, "-m", "beam_node_agent.main", "--config", config_path]
