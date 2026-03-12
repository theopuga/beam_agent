# API Reference

*Last updated: March 11, 2026*

---

## Overview

The Beam Gateway API provides an **OpenAI-compatible** interface for submitting inference requests to the network. All API endpoints use JSON for request and response bodies.

**Base URL**: `https://www.openbeam.me/api/v1`

---

## Authentication

All API requests require authentication via API key or session token.

```
Authorization: Bearer <your-api-key>
```

---

## Inference

### Chat Completions

`POST /chat/completions`

Submit a chat-style inference request.

**Request Body:**

```json
{
  "model": "Qwen/Qwen3.5-35B-A3B-Ollama",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "Explain quantum computing." }
  ],
  "max_tokens": 256,
  "temperature": 0.7,
  "stream": false
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `model` | string | Yes | Model identifier (e.g. `Qwen/Qwen3.5-35B-A3B-Ollama`) |
| `messages` | array | Yes | Array of message objects with `role` and `content` |
| `max_tokens` | int | No | Maximum tokens to generate (default: 256) |
| `temperature` | float | No | Sampling temperature 0.0–2.0 (default: 1.0) |
| `stream` | bool | No | Enable streaming response (default: false) |

**Response:**

```json
{
  "id": "chat-abc123",
  "object": "chat.completion",
  "created": 1709251200,
  "model": "Qwen/Qwen3.5-35B-A3B-Ollama",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Quantum computing uses quantum bits..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 24,
    "completion_tokens": 128,
    "total_tokens": 152
  }
}
```

---

## Node Registration

### Register Node

`POST /nodes/register`

Register a new GPU node with the network.

**Request Body:**

```json
{
  "protocol_version": "1.0",
  "machine_fingerprint": "sha256_hex_string",
  "gpu": {
    "name": "RTX 4090",
    "vram_gb": 24,
    "count": 1
  },
  "software": {
    "node_agent_version": "0.2.0",
    "ollama_version": "0.9.0"
  },
  "capabilities": {
    "max_concurrent_jobs": 1,
    "max_model_class": "A"
  }
}
```

**Response:**

```json
{
  "protocol_version": "1.0",
  "node_id": "uuid",
  "node_secret": "base64_string",
  "assignment": {
    "model_id": "Qwen/Qwen3.5-35B-A3B-Ollama",
    "model_tag": "qwen3.5:35b-a3b"
  },
  "heartbeat_interval_sec": 15
}
```

---

## Heartbeat

### Send Heartbeat

`POST /nodes/heartbeat`

**Signed request required.** Nodes must include HMAC authentication headers.

**Required Headers:**

| Header | Description |
|---|---|
| `X-Node-Id` | The node's unique identifier |
| `X-Timestamp` | Unix timestamp (seconds) |
| `X-Body-SHA256` | SHA-256 hash of the request body |
| `X-Signature` | HMAC-SHA256 signature |

**Signature Computation:**

```
canonical_string = timestamp + "\n" + sha256(body)
signature = HMAC_SHA256(node_secret, canonical_string)
```

**Request Body:**

```json
{
  "protocol_version": "1.0",
  "node_id": "uuid",
  "timestamp": 1709251200,
  "status": "running",
  "metrics": {
    "uptime_sec": 3600,
    "tokens_processed": 128000,
    "req_ok": 512,
    "req_err": 3,
    "p50_latency_ms": 180,
    "p95_latency_ms": 540
  },
  "active_jobs": [],
  "current_assignment": {
    "model_id": "Qwen/Qwen3.5-35B-A3B-Ollama",
    "model_tag": "qwen3.5:35b-a3b"
  }
}
```

---

## Assignment

### Fetch Assignment

`GET /nodes/{node_id}/assignment`

**Signed request required.**

**Response:**

```json
{
  "protocol_version": "1.0",
  "model_id": "Qwen/Qwen3.5-35B-A3B-Ollama",
  "model_tag": "qwen3.5:35b-a3b",
  "assignment_epoch": 42,
  "effective_at": 1709251200
}
```

---

## Error Codes

All error responses follow this format:

```json
{
  "error": {
    "code": "ERROR_CODE_STRING",
    "message": "Human readable description",
    "request_id": "req_uuid",
    "details": {}
  }
}
```

### Authentication & Identification Errors

| Error Code | HTTP Status | Description |
|---|---|---|
| `MISSING_HEADERS` | 400 | Required headers (X-Node-Id, X-Signature, etc.) are missing |
| `INVALID_SIGNATURE` | 401 | HMAC signature verification failed |
| `CLOCK_SKEW` | 401 | Request timestamp is outside the allowed window (>60s) |
| `REPLAY_DETECTED` | 401 | This signed request has already been processed |
| `UNKNOWN_NODE` | 403 | Node ID is not found in the registry |
| `NODE_BANNED` | 403 | Node is permanently banned from the network |

### Business Logic Errors

| Error Code | HTTP Status | Description |
|---|---|---|
| `INVALID_SCHEMA` | 400 | Request body does not match the JSON schema |
| `UNSUPPORTED_VERSION` | 400 | Protocol version is not supported |
| `NODE_FINGERPRINT_MISMATCH` | 409 | Machine fingerprint does not match the registered node |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests from this node/user |
| `NO_CAPACITY` | 503 | No nodes are available to serve the requested model |
| `JOB_NOT_FOUND` | 404 | Job ID referenced does not exist |
| `PAIRING_TOKEN_INVALID` | 400 | Pairing token is invalid or already used |
| `PAIRING_TOKEN_EXPIRED` | 400 | Pairing token has expired |
| `NODE_ALREADY_CLAIMED` | 409 | Node is already linked to another user |
| `NODE_NOT_ELIGIBLE` | 409 | Node does not meet eligibility requirements |

### Internal Errors

| Error Code | HTTP Status | Description |
|---|---|---|
| `INTERNAL_ERROR` | 500 | Unexpected server-side failure |
| `UPSTREAM_TIMEOUT` | 504 | Timeout communicating with upstream services |

---

## Model Classes

| Class | Description | Status | Example |
|---|---|---|---|
| S (Single-node) | Full model served via Ollama on one machine | **Active** | Qwen 3.5 35B-A3B |
| P (Phi) | Lightweight model served via Ollama on one machine | **Active** | Phi-4 Mini |
| A (Light) | 7-8B parameters, single-node or short chain | Reserved / Future | — |
| B (Large) | 13-30B parameters, multi-node chain | Reserved / Future | — |
| C (Heavy) | 30-100B+ parameters, distributed inference | Reserved / Future | DeepSeek V3, large MoE |

---

## Rate Limits

- Default: 60 requests per minute per API key
- Burst: Up to 10 concurrent requests
- Rate limit headers are included in responses:
  - `X-RateLimit-Limit`
  - `X-RateLimit-Remaining`
  - `X-RateLimit-Reset`
