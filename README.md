# Beam Node Agent

Public installer scripts and release binaries for the Beam Node Agent — the provider-side daemon that connects your GPU to the Beam decentralized inference network.

---

## Prerequisites

| Requirement | Details |
|---|---|
| **OS** | Linux, macOS, or Windows |
| **GPU** | NVIDIA GPU with up-to-date drivers |
| **CUDA** | CUDA toolkit (installed automatically by the Linux/macOS installer) |
| **Network** | Outbound internet access to reach the control plane |

---

## Quick Start

### Single-Node Mode (Ollama) — RunPod / any Linux GPU box

Runs the full model on one machine via Ollama. Requires a GPU with enough VRAM for the model (e.g. RTX 4090 24GB for Qwen3.5-35B-A3B).

```bash
curl -fsSL https://raw.githubusercontent.com/beam-open-node/beam_agent/feat/ollama-single-node/setup_runpod.sh | bash
```

This will:
1. Install Ollama and pull the model (~16.5GB, stored on `/workspace` if available)
2. Clone this branch and set up the Python environment
3. Write `config.yaml` and `start_agent.sh`
4. Print a **6-digit pair code** — enter it in the Rent Panel at [openbeam.me](https://www.openbeam.me)

To restart after a pod reboot:
```bash
cd beam_agent && bash start_agent.sh
```

---

### Multi-Node Mode (Petals) — Linux / macOS

```bash
curl -fsSL https://raw.githubusercontent.com/beam-open-node/beam_agent/main/install.sh -o install.sh
bash install.sh
```

### Windows (PowerShell)

```powershell
Invoke-WebRequest -Uri https://raw.githubusercontent.com/beam-open-node/beam_agent/main/install.ps1 -OutFile install.ps1
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

### What the Installer Does

1. Downloads the correct node-agent binary for your OS
2. Writes `config.yaml` with default settings
3. Sets up the Petals runtime environment (Linux/macOS)
4. Starts the agent and prints a **6-digit pair code**
5. You enter that code in the **Rent Panel** to link your machine

---

## Configuration

The agent reads its settings from `config.yaml`, which the installer creates for you. Key options include the control-plane URL, GPU VRAM limit, transport mode, and pairing ports.

See the full reference → [docs/configuration.md](docs/configuration.md)

---

## Environment Variables

Override any config value with an environment variable. The most common ones:

| Variable | Description | Default |
|---|---|---|
| `BEAM_CONTROL_PLANE_URL` | Control plane server URL | `https://www.openbeam.me` |
| `BEAM_OLLAMA_URL` | Ollama API base URL (single-node mode) | `http://localhost:11434` |
| `BEAM_OLLAMA_MODEL` | Ollama model tag to serve | `qwen3.5:35b-a3b` |
| `BEAM_GPU_NAME` | GPU name reported to control plane (auto-detected via nvidia-smi) | — |
| `BEAM_GPU_VRAM_GB` | GPU VRAM in GB (auto-detected) | — |
| `BEAM_GPU_COUNT` | Number of GPUs (auto-detected) | — |
| `BEAM_PETALS_PYTHON` | Python interpreter with Petals installed | System Python |
| `BEAM_PAIRING_TOKEN` | Pre-set pairing token (skips interactive pairing) | — |
| `BEAM_MOCK_INFERENCE` | `true` to run without a real GPU | `false` |
| `BEAM_MAX_BLOCKS` | Max transformer blocks to serve | — |

Full list → [docs/node-agent-setup.md](docs/node-agent-setup.md#environment-variables)

---

## Releases

Pre-built binaries are published at:

<https://github.com/beam-open-node/beam_agent/releases/latest>

Expected assets:

- `beam-node-agent-linux`
- `beam-node-agent-macos`
- `beam-node-agent-windows.exe`

---

## Documentation

| Doc | Description |
|---|---|
| [Node Agent Setup](docs/node-agent-setup.md) | Full setup guide, environment variables, transport modes |
| [Configuration Reference](docs/configuration.md) | Complete `config.yaml` field reference |
| [Troubleshooting](docs/troubleshooting.md) | Known issues and debugging tips |
| [API Reference](docs/api-reference.md) | Control-plane API endpoints |

---

## Support

If you get stuck, see the [Troubleshooting Guide](docs/troubleshooting.md) or contact the Beam team.