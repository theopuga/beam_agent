"""Main Textual app entry point for the Beam Node Agent TUI."""
from __future__ import annotations

import os

from textual.app import App, ComposeResult
from textual.message import Message

_DEFAULT_CONFIG = "config.yaml"


# ---------------------------------------------------------------------------
# App-level messages
# ---------------------------------------------------------------------------
class StartDashboard(Message):
    """Posted by SetupScreen when config has been written and the agent should start."""

    def __init__(self, config_path: str) -> None:
        super().__init__()
        self.config_path = config_path


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
class BeamTuiApp(App):
    """Beam Node Agent terminal UI."""

    TITLE = "Beam Node Agent"
    SUB_TITLE = "Decentralized GPU Inference"

    CSS = """
    Screen {
        background: $background;
    }
    """

    def __init__(self, config_path: str = _DEFAULT_CONFIG) -> None:
        super().__init__()
        self._config_path = config_path

    def on_mount(self) -> None:
        if os.path.exists(self._config_path):
            self._open_dashboard(self._config_path)
        else:
            self._open_setup()

    # ------------------------------------------------------------------
    def _open_setup(self) -> None:
        from beam_node_agent.tui.setup_screen import SetupScreen

        self.push_screen(SetupScreen(config_path=self._config_path))

    def _open_dashboard(self, config_path: str) -> None:
        from beam_node_agent.tui.dashboard_screen import DashboardScreen

        self.push_screen(DashboardScreen(config_path=config_path))

    # ------------------------------------------------------------------
    def on_start_dashboard(self, message: StartDashboard) -> None:
        # Pop the setup screen, then open the dashboard
        self.pop_screen()
        self._open_dashboard(message.config_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Beam Node Agent TUI")
    parser.add_argument(
        "--config",
        default=os.environ.get("BEAM_CONFIG_PATH", _DEFAULT_CONFIG),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--agent-bin",
        default=os.environ.get("BEAM_AGENT_BIN", ""),
        help="Path to the pre-built agent binary (sets BEAM_AGENT_BIN if given)",
    )
    args = parser.parse_args()

    if args.agent_bin:
        os.environ["BEAM_AGENT_BIN"] = args.agent_bin

    BeamTuiApp(config_path=args.config).run()


if __name__ == "__main__":
    main()
