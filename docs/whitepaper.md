# Beam Whitepaper

*Last updated: March 11, 2026*

---

## Decentralized Intelligence — A Censorship-Resistant Inference Network

**Ollama-powered · MoE-optimized · Progressively decentralized**

---

## Abstract

Beam is a decentralized inference network that connects GPU providers to users who need access to open-weight AI models. Each node runs the full model locally via Ollama on dedicated GPU hardware. The network is coordinated by a central control plane that handles node registration, model assignment, and request routing.

**Beam is purpose-built for:**

- Open-weight models served on community-contributed GPUs
- Censorship-resistant access to AI inference
- Efficient MoE (Mixture-of-Experts) models that maximize capability per VRAM dollar
- Progressive decentralization with tokenized incentives

---

## 1. Core Architecture

### Components

- **Gateway API**: OpenAI-compatible user entrypoint
- **Scheduler / Router**: Node selection and request routing
- **Control Plane**: Node registry, health index, assignment engine, accounting
- **Node Agent**: Provider-side daemon handling registration, heartbeat, and inference
- **Ollama Runtime**: Local model serving engine with GPU acceleration

### System Flow

```
User / Developer
      |
  Gateway API (OpenAI-compatible)
      |
  Scheduler / Router
      |
  Node Agent (selected node)
      |
  Ollama (local GPU inference)
```

Each inference request is routed to a single node that runs the full model. The node handles the entire inference locally using Ollama, which automatically utilizes all available GPUs on the machine.

---

## 2. Model Classes

### Class A — Current

- Full model runs on one machine via Ollama
- Suited for efficient MoE architectures (e.g. Qwen 3.5 35B-A3B: 35B total params, ~3B active)
- Requires a GPU with sufficient VRAM (e.g. 24 GB for current models)
- No multi-node coordination overhead

| Model | Total Params | Active Params | Min VRAM | Status |
|---|---|---|---|---|
| Qwen 3.5 35B-A3B | 35 B (MoE) | ~3 B | 24 GB | **Active** |

### Class B — Coming Soon

- Larger models requiring more VRAM or multi-GPU setups
- Single-node deployment via Ollama

| Model | Description | Status |
|---|---|---|
| Kimi K2.5 | Moonshot AI reasoning model | Coming soon |
| GLM-5 | Zhipu AI general-purpose model | Coming soon |
| DeepSeek V3 | DeepSeek MoE model | Coming soon |

---

## 3. Node Hardware Tiers

| Tier | VRAM | Description |
|---|---|---|
| T1 | 6-8 GB | Future lightweight models |
| T2 | 10-16 GB | Mid-range models |
| T3 | 24+ GB | Current models (recommended) |

Multi-GPU machines are supported. Ollama automatically utilizes all available GPUs for inference, and total VRAM is summed across devices.

---

## 4. Privacy & Transport

All traffic between users, the gateway, and nodes travels over standard HTTPS.

Future versions may introduce additional transport options:

| Mode | Transport | Intended Use | Latency |
|---|---|---|---|
| Standard | HTTPS | Default usage | Lowest |
| Secure | TLS + pinned certs | Privacy-aware users | Medium |
| Onion | Tor (.onion) only | Censorship-resistant | Higher |

---

## 5. Protocol

### Authentication

All node communications are authenticated via HMAC-SHA256 signatures. Each registered node receives a `node_secret` used to sign heartbeats and API requests.

### Validation Rules

- Reject if absolute clock skew > 60 seconds
- Reject replayed `(node_id, timestamp, body_sha256)` tuples within a 5-minute window

### Node Lifecycle

```
joining -> running -> degraded -> draining -> offline
```

### Job Lifecycle

```
accepted -> routed -> running -> completed | failed | expired
```

---

## 6. Tokenomics

### Token Overview

- **Token name**: DI (placeholder)
- **Type**: Utility accounting unit (off-chain initially; on-chain later)
- Users interact with internal credits (not crypto)
- Nodes earn internal reward units initially, crypto DI tokens later

### Inference Pricing

```
inference_cost = base_model_cost * token_count * priority
```

| Class | Relative Cost |
|---|---|
| A (current) | 1x |
| B (coming soon) | 2-3x |

### Node Rewards

```
reward = base_rate * uptime_factor * reliability_factor
```

- Uptime factor: proportion of time the node is online and responsive [0, 1]
- Reliability factor: penalizes inference failures and timeouts

### Anti-Ponzi Guarantees

- No guaranteed returns
- Rewards strictly tied to measurable work
- No referral-for-yield loops
- No fixed emission schedules
- User credits cannot be traded or speculated on

---

## 7. Security & Threat Model

### Known Realities

- Nodes can see prompts
- Nodes can log data
- The control plane is centralized

### Controls

- HMAC-signed heartbeats
- Canary inference checks
- Rate limits and anomaly detection

### Threat Categories

| Threat | Mitigation |
|---|---|
| Malicious nodes returning bad outputs | Canary checks, redundant routing, reward decay, bans |
| Prompt logging by nodes | Clear disclosure; future: encrypted inference |
| Sybil attacks | Hardware benchmarks, VRAM thresholds, admission controls |
| Gateway API abuse | Rate limiting, token pricing, anomaly detection |

### Explicit Non-Guarantees

The system does **not** guarantee:
- Prompt confidentiality
- Output correctness
- Resistance to state-level adversaries
- Trustless inference

---

## 8. Proof-of-Inference (PoInf)

### What It Proves

- Node participated in inference
- Duration and tokens served

### What It Does NOT Prove

- Output correctness
- Prompt secrecy

### Eligibility Rules

A node is reward-eligible when:
- Job ID exists and signature is valid
- Node is the assigned node for that job
- Heartbeat freshness is within tolerance
- No disqualifying error flags

---

## 9. Roadmap

| Milestone | Description |
|---|---|
| M0 | Single-node Ollama inference (Class A) — **current** |
| M1 | Node agent + registry + pairing |
| M2 | Gateway API operational with Qwen 3.5 35B-A3B |
| M3 | Additional MoE models (Kimi K2.5, GLM-5) |
| M4 | Off-chain token ledger |
| M5 | On-chain settlement (optional) |
| M6 | Transport privacy options (secure, onion) |

---

## 10. Summary

> Beam is a decentralized inference network that connects GPU providers to users who need access to open-weight AI models. Each node runs the full model locally via Ollama, optimized for efficient MoE architectures. Progressive decentralization via tokenized incentives ensures the network grows sustainably.

Users pay in stable internal credits. Nodes earn crypto tokens for work. Platform revenue comes from credit sales, fees, and treasury tokens. Value and rewards are tied to actual usage, not speculation.
