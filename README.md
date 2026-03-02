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

### Linux / macOS

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
| `BEAM_PETALS_PYTHON` | Python interpreter with Petals installed | System Python |
| `BEAM_PAIRING_TOKEN` | Pre-set pairing token (skips interactive pairing) | — |
| `BEAM_MOCK_INFERENCE` | `true` to run without a real GPU | `false` |
| `BEAM_SINGLE_NODE` | Enable single-node mode | `false` |
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