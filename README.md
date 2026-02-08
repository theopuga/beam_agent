# Beam Node Agent (Public Downloads)

This repo hosts the **public installer scripts** and **release binaries** for Beam node agents.
Source code lives in the private core repo.

## Quick Start (Linux/macOS)

```bash
curl -fsSL https://raw.githubusercontent.com/theopuga/beam_agent/main/install.sh -o install.sh
bash install.sh
Quick Start (Windows PowerShell)
powershell

Invoke-WebRequest -Uri https://raw.githubusercontent.com/theopuga/beam_agent/main/install.ps1 -OutFile install.ps1
powershell -ExecutionPolicy Bypass -File .\install.ps1
What This Does
Downloads the correct node agent binary for your OS
Writes config.yaml
Starts the agent and prints a 6‑digit pair code
You enter that code in the Rent Panel to link your machine
Releases
Binaries are published under:
https://github.com/theopuga/beam_agent/releases/latest

Expected asset names:

beam-node-agent-linux
beam-node-agent-macos
beam-node-agent-windows.exe
Support
If you get stuck, contact the Beam team.