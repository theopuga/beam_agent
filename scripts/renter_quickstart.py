#!/usr/bin/env python3
import os
import stat
import sys
import subprocess
import urllib.request
from pathlib import Path


def prompt(text: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{text}{suffix}: ").strip()
    return value or (default or "")


def confirm(text: str, default: bool = False) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    value = input(f"{text}{suffix}: ").strip().lower()
    if not value:
        return default
    return value in {"y", "yes"}


def detect_asset_name() -> str:
    if sys.platform.startswith("linux"):
        return "beam-node-agent-linux"
    if sys.platform == "darwin":
        return "beam-node-agent-macos"
    return "beam-node-agent-windows.exe"


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        data = response.read()
    dest.write_bytes(data)


def write_config(path: Path, control_plane_url: str) -> None:
    content = f"""control_plane:
  url: "{control_plane_url}"

petals:
  port: 31337
  gpu_vram_limit: 0.9

agent:
  heartbeat_interval_sec: 15
  state_file: "node_state.json"
  transports:
    - "fast"
  pairing_host: "127.0.0.1"
  pairing_ports:
    - 51337
    - 51338
    - 51339
    - 51340
  capabilities:
    supports_heavy_middle_layers: true
    max_concurrent_jobs: 1
"""
    path.write_text(content, encoding="utf-8")


def main() -> int:
    print("Beam Node Agent Quickstart")
    print("This will download the agent, write a config, and start pairing.")
    print("Keep this terminal open to see the pairing code.")
    print()

    if not confirm("Continue?", default=True):
        print("Aborted.")
        return 1

    control_plane_url = os.environ.get(
        "BEAM_CONTROL_PLANE_URL", "https://beam-production-f317.up.railway.app"
    ).rstrip("/")
    print(f"Control plane URL: {control_plane_url}")

    asset_name = detect_asset_name()
    default_base = "https://github.com/theopuga/beam_agent/releases/latest/download"
    release_base = prompt("Release base URL", default=default_base).rstrip("/")
    download_url = f"{release_base}/{asset_name}"
    binary_path = Path(asset_name)

    if binary_path.exists():
        if confirm(f"{binary_path} exists. Redownload?", default=False):
            download_file(download_url, binary_path)
            print(f"Downloaded {binary_path}")
        else:
            print(f"Using existing {binary_path}")
    else:
        download_file(download_url, binary_path)
        print(f"Downloaded {binary_path}")

    if os.name != "nt":
        binary_path.chmod(binary_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    config_path = prompt("Config path (press Enter for default)", default="config.yaml")
    config_file = Path(config_path).expanduser()
    if config_path.strip().lower() in {"y", "yes", "n", "no"}:
        print("Using default config.yaml")
        config_file = Path("config.yaml")

    if config_file.exists():
        if confirm(f"{config_file} exists. Overwrite?", default=False):
            write_config(config_file, control_plane_url)
            print(f"Wrote {config_file}")
        else:
            print(f"Using existing {config_file}")
    else:
        write_config(config_file, control_plane_url)
        print(f"Wrote {config_file}")

    print()
    print("Next: the agent will start and print a 6-digit pair code.")
    print("Open the Rent Panel and enter the code to link this machine.")
    print()

    if not confirm("Start the agent now?", default=True):
        print("You can start it later with:")
        print(f"{binary_path} --config {config_file}")
        return 0

    os.environ["BEAM_CONTROL_PLANE_URL"] = control_plane_url
    cmd = [str(binary_path), "--config", str(config_file)]
    print(f"Running: {' '.join(cmd)}")
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
