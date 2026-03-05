# Petals Inference Debugging — Session Notes

## Context

RunPod GPU pod (RTX 3090, 24 GB VRAM) serving `Qwen/Qwen3-8B` via `beam-node-agent-linux`.
SSH: `root@213.192.2.115 -p 40195 -i ~/.ssh/id_ed25519`

---

## Bug 1 (FIXED): p2pd Unix socket multiaddr encoding — stream handler never fires

### Symptom

Client calls `session.step(embeds)`, `stream_open` returns type=0 (OK), but the server's
`_handler` callback in Python **never fires**. Inference hangs indefinitely; client gets
`TimeoutError` after 3 minutes.

### Root Cause

`Multiaddr('/unix/tmp/hivemind-p2pclient-XYZ.sock').to_bytes()` encodes the path as a
**relative** path — `tmp/hivemind-p2pclient-XYZ.sock` (no leading `/`):

```
9003 16 746d702f...   ← length=0x16=22, no leading 0x2f='/'
```

Python's `value_for_protocol(P_UNIX)` adds the `/` back when decoding, so the mismatch
is invisible on the Python side. But **Go's `manet.Dial`** reads the raw bytes and gets
the relative path, which doesn't exist — it fails silently (only logged at `Debug` level).

### Fix

`_make_abs_unix_maddr(socket_path)` in `hivemind/p2p/p2p_daemon.py` constructs the multiaddr
bytes manually with the full absolute path (including the leading `/`):

```python
def _make_abs_unix_maddr(socket_path: str):
    p_unix_varint = bytes([0x90, 0x03])   # P_UNIX = 400 varint-encoded
    path_bytes = socket_path.encode()     # e.g. b'/tmp/hivemind-p2pclient-XYZ.sock'
    length = len(path_bytes)
    ...
    return Multiaddr(p_unix_varint + len_bytes + path_bytes)
```

Both occurrences of `self._client_listen_maddr = Multiaddr(cls._UNIX_SOCKET_PREFIX + ...)` in
`P2P.create()` and `P2P.replicate()` are replaced with `_make_abs_unix_maddr(...)`.

### Applied

- Manually patched on the RunPod server: `/root/beam-petals-venv/lib/python3.11/site-packages/hivemind/p2p/p2p_daemon.py`
- Added `patch_hivemind_p2pd()` to `scripts/patch_petals_models.py` (runs automatically on `install.sh`)
- Same function inlined in `install.sh`'s embedded patcher

### Verified

Test script `test_p2p_nob.py` confirmed `[SERVER] HANDLER FIRED!` and `[RESULT] SUCCESS`
after the fix. Agent log subsequently shows `[DEBUG] p2pclient _handler called` and
`[DEBUG] rpc_inference called` on every inference attempt.

---

## Bug 2 (IN PROGRESS): `wait_for_alloc` hangs on subsequent inference sessions

### Symptom

After the first inference request, all subsequent `rpc_inference` calls get stuck at
`wait_for_alloc` and never reach `alloc_done`. The server logs:

```
rpc_inference.wait_for_alloc(size=0.00 GiB), already used 0.00/2.25 GiB (0.1%)
```

The client times out (`TimeoutError`) after ~3 minutes. The `0.1%` cache usage (~2.4 MB)
is likely a leak from the first failed request.

### First Request History

The very first request failed because the test script didn't use `torch.no_grad()`,
causing `RuntimeError: Cowardly refusing to serialize non-leaf tensor which requires_grad`.
The server did reach `alloc_done` for this request and `rpc_inference.close` followed quickly.
All subsequent requests hang at `wait_for_alloc`.

### Suspected Causes (not yet confirmed)

1. **`_lock_acquire_memory` deadlock** — `_wait_for_free_memory` acquires an `mp.Lock` via
   `enter_asynchronously`. With `alloc_timeout=0` (the client default), no asyncio timeout
   is applied to the lock acquisition. If the lock is stuck from a previous session, every
   subsequent request waits indefinitely.

2. **Leaked KV cache from failed first session** — The requires_grad error may have left
   `current_size_bytes` non-zero in the `MemoryCache`. Even 0.1% used shouldn't block
   allocation with 99.9% free; but if the lock is also stuck, it manifests as a permanent hang.

3. **Pipe backlog** — `_pipe_send.send()` in `_schedule_alloc` sends allocation messages
   that the runtime reads via `use_cache()`. If the runtime never ran (first request failed
   before forward pass), the pipe might have unread messages that eventually block.

### Debug Instrumentation Added (temporary, on server only)

Added `[MEM]` debug prints to `memory_cache.py`'s `_wait_for_free_memory` to log:
- Before lock acquisition attempt
- When lock is acquired/released
- Which OOM path is taken

### Next Steps

1. **Restart the agent cleanly** — kill all stale test client processes (PIDs 114334, 120846,
   271204, 272297 etc.) that are stuck since the previous session, then restart beam_agent.
   Watch for `[MEM]` lines in the log to see where allocation actually stalls.

2. **Check if `_lock_acquire_memory` is stuck** — add `block=False` probe:
   ```python
   acquired = self._lock_acquire_memory.acquire(block=False)
   if acquired:
       self._lock_acquire_memory.release()
       print("Lock is FREE")
   else:
       print("Lock is HELD — deadlock!")
   ```

3. **Try `alloc_timeout > 0`** — if the client sends `alloc_timeout=30` in metadata,
   `async_timeout.timeout(30)` would be applied, and a stuck lock would surface as
   `AllocationFailed` (with a timeout error) rather than a silent hang. If we see
   `AllocationFailed` within 30s, the lock IS stuck.

4. **Investigate the pipe backlog** — check how many messages are in `_pipe_recv` using
   `poll()` before and after a new request arrives.

5. **Consider resetting `MemoryCache` state on session start** — if `current_size_bytes`
   or `_lock_acquire_memory` can get into a bad state after a failed request, reset them.

---

## Other Fixes Applied During This Session

### transformers 5.x compatibility (Qwen3 / Qwen3.5-MoE blocks)

Added `WrappedQwen3Block` and related model classes to `patch_petals_models.py`. These handle:
- `position_embeddings` required by transformers≥5.x attention (RoPE computed upfront)
- `DynamicCache` instead of legacy `past_key_values` tuple
- KV tensor format conversion between beam/petals format and transformers format

### `SO_REUSEPORT` removal

`hivemind/utils/networking.py` used `SO_REUSEPORT` which isn't available on all kernels.
Patched to use `SO_REUSEADDR` instead (or skip the option).

### `p2pd` stale DHT entries

Added `identity_path=/root/.petals_p2p_identity` to `run_server.py` so the server uses
a persistent peer ID across restarts, avoiding stale DHT entries.

### `balanced=True` for multi-worker connection handlers

`hivemind/moe/server/connection_handler.py` must use `balanced=True` when registering
protocol handlers. With `balanced=False`, only one of the 8 worker processes can register;
the other 7 crash with `P2PDaemonError: handler for protocol ... already set`.

---

## Current Server State (as of 2026-03-05)

- `beam-node-agent-linux` is running (PID 269278/269279)
- The agent is in a layer-detection loop (no actual petals server subprocess running)
- GPU shows only 4 MiB used — model is not loaded
- Several stale test client processes from previous sessions are still running (PIDs 114334, 120846, 271204, 272297)
- `_make_abs_unix_maddr` fix is applied to the venv's `p2p_daemon.py`
- `memory_cache.py` has temporary `[MEM]` debug prints

### Clean Restart Procedure

```bash
# 1. Kill stale test processes
kill 114334 114470 120846 121047 271204 271404 272297 272433 2>/dev/null || true
kill $(pgrep -f "hivemind-p2pd" | grep -v $(pgrep -f beam-node-agent)) 2>/dev/null || true

# 2. Restart agent (it will re-launch the petals server)
cd /root && pkill -f beam-node-agent-linux && sleep 2
nohup ./beam-node-agent-linux --config config.yaml >> /tmp/agent.log 2>&1 &

# 3. Watch logs
tail -f /tmp/agent.log | grep -E "rpc_inference|alloc|MEM|handler"
```
