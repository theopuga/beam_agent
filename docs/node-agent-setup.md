# Node Agent Setup

*Last updated: March 11, 2026*

---

## Overview

The Beam Node Agent is the provider-side daemon that connects your GPU to the Beam inference network. It handles registration, heartbeat reporting, assignment management, and inference execution via Ollama. Each node runs the full model locally — no multi-node coordination or distributed block chains are involved.

---

## Quick Start

Run the one-liner installer on any Linux GPU machine (RunPod, Lambda, bare metal, etc.):

```bash
curl -fsSL https://raw.githubusercontent.com/beam-open-node/beam_agent/main/setup_runpod.sh | bash
```

### What the Installer Does

1. Installs Ollama and pulls the active model (~16.5 GB, stored on `/workspace` if available)
2. Clones this repository and sets up the Python environment
3. Writes `config.yaml` and `start_agent.sh`
4. Starts the agent and prints a **6-digit pair code**
5. You enter that code in the **Rent Panel** at [openbeam.me](https://www.openbeam.me) to link your machine

---

## Configuration

The agent reads configuration from `config.yaml`. Here is a reference:

```yaml
control_plane:
  url: "https://www.openbeam.me"

ollama:
  base_url: "http://localhost:11434"
  model_tag: "qwen3.5:35b-a3b"

agent:
  heartbeat_interval_sec: 15
  state_file: "node_state.json"
  # mock_inference: true      # For testing without real GPU
  # pairing_token: "paste-pairing-token-here"
  pairing_host: "127.0.0.1"
  pairing_ports:
    - 51337
    - 51338
    - 51339
    - 51340
  capabilities:
    max_concurrent_jobs: 1
    max_model_class: "B"
    preferred_model_id: "Qwen/Qwen3.5-35B-A3B-Ollama"
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `BEAM_CONTROL_PLANE_URL` | Control plane server URL | `https://www.openbeam.me` |
| `BEAM_OLLAMA_URL` | Ollama API base URL | `http://localhost:11434` |
| `BEAM_OLLAMA_MODEL` | Ollama model tag to serve | `qwen3.5:35b-a3b` |
| `BEAM_PAIRING_TOKEN` | Pre-set pairing token (skips interactive pairing) | — |
| `BEAM_PAIRING_PORT` | Single pairing port override | — |
| `BEAM_PAIRING_PORTS` | Comma-separated pairing port list | `51337,51338,51339,51340` |
| `BEAM_MOCK_INFERENCE` | Set `true` to run mock inference (no GPU needed) | `false` |
| `BEAM_GPU_NAME` | Override GPU name detection | Auto-detect |
| `BEAM_GPU_VRAM_GB` | Override GPU VRAM detection (in GB) | Auto-detect |
| `BEAM_GPU_COUNT` | Override GPU count detection | Auto-detect |
| `BEAM_TOR_ENABLED` | Enable Tor onion routing | `true` |
| `BEAM_TOR_SOCKS_PORT` | Tor SOCKS5 proxy port | `9050` |
| `BEAM_TOR_DATA_DIR` | Tor state data directory | `tor_data` |
| `BEAM_TOR_BINARY` | Custom Tor binary path | `tor` |
| `BEAM_E2E_ENCRYPTION` | Enable E2E encryption (`true` or `1`) | `false` |

---

## Pairing Process

1. Start the node agent — it will print a 6-digit pair code
2. Open the Beam web app at [openbeam.me](https://www.openbeam.me) and go to the Rent Panel
3. Enter the pair code to link your machine to your account
4. The agent will begin receiving assignments and serving inference

---

## Multi-GPU Support

If your machine has multiple NVIDIA GPUs, Ollama will automatically use all available GPUs for inference. The agent detects and reports each GPU to the control plane. Total VRAM is summed across all devices.

---

## Restarting the Agent

After a pod reboot or machine restart:

```bash
cd beam_agent && bash start_agent.sh
```

This will restart the Ollama daemon (if not already running) and launch the agent.

---

## Support

If you encounter issues, see the [Troubleshooting Guide](./troubleshooting.md) or contact the Beam team.
