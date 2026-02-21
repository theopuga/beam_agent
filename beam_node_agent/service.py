import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, AsyncIterator, Dict, Iterator, Optional

import aiohttp
from aiohttp import web

from .config import BeamConfig
from .node_identity import NodeIdentity
from .petals_wrapper import PetalsWrapper

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Inference worker script
# ---------------------------------------------------------------------------
# This script is written to a temp file and launched as a subprocess using
# BEAM_PETALS_PYTHON (the real petals-venv Python, not the PyInstaller binary).
# Running as a separate process means torch / hivemind / petals can be imported
# normally — no sys.path surgery, no C-extension conflicts.
#
# Protocol (newline-delimited JSON on stdin/stdout):
#   stdin  ← {"job_id":…, "model_id":…, "prompt":…,
#               "max_new_tokens":…, "temperature":…}
#   stdout → {"type":"ready"}                          (once, at startup)
#           → {"type":"token",  "job_id":…, "token":…} (per token)
#           → {"type":"done",   "job_id":…}            (end of job)
#           → {"type":"error",  "job_id":…, "message":…, "traceback":…}
# ---------------------------------------------------------------------------
_INFERENCE_WORKER_SCRIPT = r"""
import json
import os
import sys
import time


def _emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _load_model(model_id, peers, dtype_map, float16):
    # Load the distributed model and tokenizer; returns (model, tokenizer).
    from petals import AutoDistributedModelForCausalLM
    from transformers import AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    if tokenizer.pad_token is None and tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoDistributedModelForCausalLM.from_pretrained(
        model_id,
        initial_peers=peers,
        torch_dtype=dtype_map.get("float16", float16),
    )
    model.eval()
    return model, tokenizer


def main():
    model = None
    tokenizer = None
    current_model_id = None

    # initial_peers can be provided via env var (JSON-encoded list).
    # If not set, petals falls back to PUBLIC_INITIAL_PEERS.
    _peers_env = os.environ.get("BEAM_INFERENCE_INITIAL_PEERS", "")
    try:
        _initial_peers = json.loads(_peers_env) if _peers_env else None
    except Exception:
        _initial_peers = None

    # Maximum seconds to wait for the first token before aborting.
    _inference_timeout = int(os.environ.get("BEAM_INFERENCE_TIMEOUT", "120"))

    # Pre-warm: if BEAM_INFERENCE_WARMUP_MODEL is set, load the model immediately
    # so the first user request doesn't have to wait for disk I/O.
    _warmup_model_id = os.environ.get("BEAM_INFERENCE_WARMUP_MODEL", "")

    _emit({"type": "ready"})

    if _warmup_model_id:
        try:
            from petals.constants import DTYPE_MAP, PUBLIC_INITIAL_PEERS
            import torch
            peers = _initial_peers if _initial_peers else PUBLIC_INITIAL_PEERS
            model, tokenizer = _load_model(_warmup_model_id, peers, DTYPE_MAP, torch.float16)
            current_model_id = _warmup_model_id
            _emit({"type": "warmup_done", "model_id": _warmup_model_id})
        except Exception as exc:
            import traceback as _tb
            _emit({"type": "warmup_error", "model_id": _warmup_model_id,
                   "message": str(exc), "traceback": _tb.format_exc()})

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except Exception:
            continue

        job_id = req.get("job_id", "")
        model_id = req.get("model_id", "")
        prompt = req.get("prompt", "")
        max_new_tokens = int(req.get("max_new_tokens") or 256)
        temperature = float(req.get("temperature") or 1.0)
        do_sample = temperature > 0.0

        try:
            if model is None or current_model_id != model_id:
                from petals.constants import DTYPE_MAP, PUBLIC_INITIAL_PEERS
                import torch
                peers = _initial_peers if _initial_peers else PUBLIC_INITIAL_PEERS
                model, tokenizer = _load_model(model_id, peers, DTYPE_MAP, torch.float16)
                current_model_id = model_id

            import torch

            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"]

            gen_kwargs = dict(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )
            if do_sample:
                gen_kwargs["temperature"] = temperature

            # Run generation synchronously.
            # NOTE: TextIteratorStreamer does not work reliably with
            # AutoDistributedModelForCausalLM because the distributed
            # generate() implementation does not call streamer.put() per token.
            with torch.no_grad():
                output_ids = model.generate(**gen_kwargs)

            # Decode only the newly generated tokens (skip the prompt).
            new_ids = output_ids[0, input_ids.shape[-1]:]
            text = tokenizer.decode(new_ids, skip_special_tokens=True)

            # Emit one chunk at a time so the frontend streams word-by-word.
            words = text.split(" ")
            for i, word in enumerate(words):
                chunk = word if i == 0 else " " + word
                if chunk:
                    _emit({"type": "token", "job_id": job_id, "token": chunk})

            _emit({"type": "done", "job_id": job_id})

        except Exception as exc:
            import traceback as _tb
            # Reset the cached model if it may be in a bad state after an error.
            model = None
            current_model_id = None
            _emit({
                "type": "error",
                "job_id": job_id,
                "message": str(exc),
                "traceback": _tb.format_exc(),
            })


if __name__ == "__main__":
    main()
"""


class _InferenceSubprocess:
    """
    Manages a persistent Python subprocess (launched via BEAM_PETALS_PYTHON)
    that keeps the petals distributed model loaded and serves inference
    requests via a newline-delimited JSON stdin/stdout protocol.

    Because it runs as a separate OS process using the real petals-venv
    Python interpreter, it has full access to torch, hivemind, petals,
    and all C-extension stdlib modules (cmath, etc.) without conflicting
    with the PyInstaller binary's embedded runtime.
    """

    def __init__(
        self,
        petals_python: str,
        script_path: str,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> None:
        self._python = petals_python
        self._script = script_path
        self._extra_env = extra_env or {}
        self._proc: Optional[subprocess.Popen] = None
        self._ready = False
        # Serialise access: only one job at a time (mirrors _inference_lock above)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_started(self) -> None:
        """Start (or restart) the worker subprocess if it is not alive."""
        if self._proc is not None and self._proc.poll() is None:
            return  # still running

        log.info(
            "Starting inference worker subprocess (python=%s script=%s)",
            self._python, self._script,
        )
        env = os.environ.copy()
        env.update(self._extra_env)  # overlay beam-specific vars (peers, timeout, etc.)
        self._proc = subprocess.Popen(
            [self._python, self._script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self._ready = False

        # Drain stderr in a background thread so it doesn't block.
        threading.Thread(target=self._drain_stderr, daemon=True).start()

        # Block until the worker signals "ready" (or the process dies).
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except Exception:
                log.warning("InferenceWorker unexpected startup output: %s", line)
                continue
            if msg.get("type") == "ready":
                self._ready = True
                log.info("Inference worker subprocess ready.")
                break
            if msg.get("type") == "error":
                log.error("Inference worker startup error: %s", msg.get("message"))
                break
        else:
            # stdout closed without a ready signal — process likely died.
            log.error("Inference worker subprocess exited before signalling ready.")

    def _drain_stderr(self) -> None:
        if self._proc and self._proc.stderr:
            for line in self._proc.stderr:
                log.info("InferenceWorker stderr: %s", line.rstrip())

    def stop(self) -> None:
        if self._proc:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = None
        self._ready = False

    def is_alive(self) -> bool:
        return self._ready and self._proc is not None and self._proc.poll() is None

    # ------------------------------------------------------------------
    # Job execution
    # ------------------------------------------------------------------

    def run_job(
        self,
        job_id: str,
        model_id: str,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> Iterator[dict]:
        """
        Send one inference job to the worker and yield response dicts.
        Blocks the calling thread until the job completes (or errors).
        Caller must hold the inference lock so only one job runs at a time.
        """
        with self._lock:
            self._ensure_started()
            if not self._ready or self._proc is None:
                yield {
                    "type": "error",
                    "job_id": job_id,
                    "message": "inference worker failed to start",
                }
                return

            req = json.dumps({
                "job_id": job_id,
                "model_id": model_id,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            })
            assert self._proc.stdin is not None
            try:
                self._proc.stdin.write(req + "\n")
                self._proc.stdin.flush()
            except Exception as exc:
                yield {
                    "type": "error",
                    "job_id": job_id,
                    "message": f"failed to send job to worker: {exc}",
                }
                return

            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except Exception:
                    log.warning("InferenceWorker non-JSON output: %s", line)
                    continue
                mtype = msg.get("type")
                # Absorb warm-up notifications silently (they are not job results).
                if mtype == "warmup_done":
                    log.info("InferenceWorker warmup complete: model=%s", msg.get("model_id"))
                    continue
                if mtype == "warmup_error":
                    log.warning("InferenceWorker warmup error: %s", msg.get("message"))
                    continue
                yield msg
                if mtype in ("done", "error"):
                    break


class NodeAgent:
    def __init__(self, config: BeamConfig):
        self.config = config
        self.identity = NodeIdentity(config.agent.state_file)
        self.petals = PetalsWrapper(
            port=config.petals.port,
            public_ip=config.petals.public_ip,
            gpu_vram_limit=config.petals.gpu_vram_limit,
        )
        self._gpu_spec = self._resolve_gpu_spec()
        self.session: Optional[aiohttp.ClientSession] = None
        self.current_assignment: Optional[Dict] = None
        self.started_at = int(time.time())
        self._pairing_runner: Optional[web.AppRunner] = None
        self._pairing_site: Optional[web.BaseSite] = None
        self._pairing_port: Optional[int] = None
        self._pairing_session_id: Optional[str] = None
        self._pairing_session_secret: Optional[str] = None
        self._awaiting_remote_pairing = False
        self._register_lock = asyncio.Lock()
        self._ws_task: Optional[asyncio.Task] = None
        self._ws_send_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()
        self._model_lock = asyncio.Lock()  # kept for compatibility; no longer guards model loading
        self._petals_check_task: Optional[asyncio.Task] = None
        # Persistent inference worker subprocess (uses BEAM_PETALS_PYTHON)
        self._inference_worker: Optional[_InferenceSubprocess] = None
        self._inference_worker_script: Optional[str] = None

    async def start(self):
        log.info("Starting Beam Node Agent...")
        log.info("Control plane URL: %s", self.config.control_plane.url)
        self.session = aiohttp.ClientSession()
        self._ws_task = asyncio.create_task(self._run_gateway_ws())
        await self._start_pairing_server()
        await self._start_remote_pairing_session()

        try:
            # 1. Registration / Linking Phase
            if self.identity.node_id:
                if self.config.agent.pairing_token:
                    linked, message, status = await self._link_node_with_token(
                        self.config.agent.pairing_token
                    )
                    if linked:
                        log.info("Linked node to renter: %s", message)
                    else:
                        log.warning(
                            "Failed to link node with pairing token (%s): %s",
                            status,
                            message,
                        )
                log.info(f"Using existing identity: {self.identity.node_id}")
            else:
                if self.config.agent.pairing_token or not self._awaiting_remote_pairing:
                    await self._register()
                else:
                    log.info("Waiting for remote pairing before registering.")

            # 2. Assignment Check (Initial)
            # If we restarted, we should re-fetch assignment or use cached if persisted.
            # ideally fetch from CP.
            await self._refresh_assignment()

            # 3. Main Heartbeat Loop
            await self._loop()

        except asyncio.CancelledError:
            log.info("Agent stopping...")
        except Exception as e:
            log.error(f"Fatal error in agent: {e}", exc_info=True)
        finally:
            await self.stop()

    async def stop(self):
        log.info("Shutting down...")
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
            self._ws_task = None
        if self._petals_check_task:
            self._petals_check_task.cancel()
            self._petals_check_task = None
        await self._stop_pairing_server()
        self.petals.stop()
        if self.session:
            await self.session.close()

    def _pairing_ports(self) -> list[int]:
        ports = self.config.agent.pairing_ports or []
        return ports if ports else [51337, 51338, 51339, 51340]

    def _cors_headers(self) -> dict:
        return {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "POST, OPTIONS, GET",
        }

    async def _start_pairing_server(self):
        app = web.Application()

        async def options_handler(request):
            return web.Response(status=204, headers=self._cors_headers())

        async def health_handler(request):
            petals_status = self.petals.status_snapshot()
            payload = {
                "status": "ok",
                "node_id": self.identity.node_id,
                "registered": bool(self.identity.node_id),
                "port": self._pairing_port,
                "petals": petals_status,
            }
            return web.json_response(payload, headers=self._cors_headers())

        async def pair_handler(request):
            try:
                payload = await request.json()
            except Exception:
                payload = {}

            token = payload.get("token") or payload.get("pairing_token")
            if not token:
                return web.json_response(
                    {"status": "error", "message": "pairing token required"},
                    status=400,
                    headers=self._cors_headers(),
                )

            try:
                if self.identity.node_id:
                    return await self._link_existing_node(token)

                await self._register(pairing_token_override=token)
                if self.identity.node_id:
                    return web.json_response(
                        {
                            "status": "registered",
                            "node_id": self.identity.node_id,
                            "message": "Node registered with pairing token.",
                        },
                        headers=self._cors_headers(),
                    )
                return web.json_response(
                    {"status": "error", "message": "Registration failed"},
                    status=500,
                    headers=self._cors_headers(),
                )
            except Exception as exc:
                return web.json_response(
                    {"status": "error", "message": str(exc)},
                    status=500,
                    headers=self._cors_headers(),
                )

        app.router.add_route("OPTIONS", "/pair", options_handler)
        app.router.add_route("OPTIONS", "/health", options_handler)
        app.router.add_get("/health", health_handler)
        app.router.add_post("/pair", pair_handler)

        runner = web.AppRunner(app)
        await runner.setup()

        host = self.config.agent.pairing_host
        for port in self._pairing_ports():
            try:
                site = web.TCPSite(runner, host=host, port=port)
                await site.start()
                self._pairing_runner = runner
                self._pairing_site = site
                self._pairing_port = port
                log.info("Local pairing server listening on %s:%s", host, port)
                return
            except OSError as exc:
                log.warning("Pairing port %s unavailable: %s", port, exc)
                continue

        await runner.cleanup()
        log.warning("Failed to start local pairing server on any port.")

    async def _stop_pairing_server(self):
        if self._pairing_runner:
            await self._pairing_runner.cleanup()
            self._pairing_runner = None
            self._pairing_site = None
            self._pairing_port = None

    async def _start_remote_pairing_session(self):
        if self.config.agent.pairing_token:
            return

        url = f"{self.config.control_plane.url}/api/v1/beam/pairing/sessions"
        payload = {"machine_fingerprint": self.identity.machine_fingerprint}

        try:
            async with self.session.post(url, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    log.warning("Failed to create pairing session: %s", text)
                    return

                data = await resp.json()
                self._pairing_session_id = data.get("session_id")
                self._pairing_session_secret = data.get("session_secret")
                self._awaiting_remote_pairing = not self.identity.node_id
                pair_code = data.get("pair_code")
                expires_at = data.get("expires_at")

                if pair_code:
                    log.info("Pairing code: %s (expires %s)", pair_code, expires_at)

                asyncio.create_task(self._poll_remote_pairing_session())
        except Exception as exc:
            log.warning("Failed to start remote pairing session: %s", exc)

    async def _poll_remote_pairing_session(self):
        if not self._pairing_session_id or not self._pairing_session_secret:
            return

        url = f"{self.config.control_plane.url}/api/v1/beam/pairing/sessions/{self._pairing_session_id}"
        headers = {"x-session-secret": self._pairing_session_secret}

        while not self.identity.node_id:
            try:
                async with self.session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        await asyncio.sleep(3)
                        continue

                    data = await resp.json()
                    status = data.get("status")
                    if status == "claimed" and data.get("token"):
                        token = data.get("token")
                        try:
                            if self.identity.node_id:
                                linked, message, status_code = await self._link_node_with_token(
                                    token
                                )
                                if linked:
                                    log.info("Remote pairing linked node: %s", message)
                                else:
                                    log.warning(
                                        "Remote pairing link failed (%s): %s",
                                        status_code,
                                        message,
                                    )
                                self._awaiting_remote_pairing = False
                                return
                            await self._register(pairing_token_override=token)
                            self._awaiting_remote_pairing = False
                            return
                        except Exception as exc:
                            log.warning("Remote pairing registration failed: %s", exc)
                            await asyncio.sleep(3)
                            continue
                    if status == "expired":
                        log.warning("Pairing session expired")
                        self._awaiting_remote_pairing = False
                        await self._start_remote_pairing_session()
                        return
            except Exception as exc:
                log.warning("Remote pairing poll error: %s", exc)
            await asyncio.sleep(3)

        await asyncio.sleep(3)

    def _resolve_gpu_spec(self) -> Dict[str, Any]:
        env_name = os.environ.get("BEAM_GPU_NAME")
        env_vram = os.environ.get("BEAM_GPU_VRAM_GB")
        env_count = os.environ.get("BEAM_GPU_COUNT")

        name = env_name.strip() if env_name else None
        vram_gb = self._coerce_float(env_vram) if env_vram else None
        count = self._coerce_int(env_count) if env_count else None

        if name and vram_gb and count:
            return {"name": name, "vram_gb": vram_gb, "count": count}

        python_exec = os.environ.get("BEAM_PETALS_PYTHON") or sys.executable
        try:
            # Detect GPU using a subprocess to avoid importing torch in this process
            script = "import torch; print(f'{torch.cuda.device_count()}|{torch.cuda.get_device_properties(0).name}|{torch.cuda.get_device_properties(0).total_memory}') if torch.cuda.is_available() else print('0||0')"
            result = (
                subprocess.check_output([python_exec, "-c", script], text=True)
                .strip()
                .split("|")
            )
            if len(result) == 3 and int(result[0]) > 0:
                device_count = int(result[0])
                total_mem = int(result[2])
                detected_vram = round(float(total_mem) / (1024**3), 2)
                detected_name = result[1] or "GPU"
                return {
                    "name": detected_name,
                    "vram_gb": detected_vram if detected_vram > 0 else 24,
                    "count": device_count,
                }
        except Exception as exc:
            log.warning("GPU detection failed, using defaults: %s", exc)

        return {
            "name": name or "Generic GPU",
            "vram_gb": vram_gb if vram_gb else 24,
            "count": count if count and count > 0 else 1,
        }

    def _normalized_capabilities(self) -> Dict[str, Any]:
        caps = dict(self.config.agent.capabilities or {})
        max_blocks = self._coerce_int(caps.get("max_blocks"))
        if max_blocks is not None and max_blocks > 0:
            caps["max_blocks"] = max_blocks
        else:
            caps.pop("max_blocks", None)

        max_model_class = caps.get("max_model_class")
        if max_model_class:
            caps["max_model_class"] = str(max_model_class).upper()
        else:
            caps.pop("max_model_class", None)
        return caps

    def _gateway_ws_url(self) -> str:
        base = (self.config.control_plane.url or "").rstrip("/")
        if base.startswith("https://"):
            ws_base = "wss://" + base[len("https://") :]
        elif base.startswith("http://"):
            ws_base = "ws://" + base[len("http://") :]
        else:
            ws_base = base
        return f"{ws_base}/api/v1/beam/agents/ws"

    async def _run_gateway_ws(self):
        backoff = 1.0
        while True:
            if not self.session:
                await asyncio.sleep(1)
                continue

            if not self.identity.node_id or not self.identity.node_secret:
                await asyncio.sleep(1)
                continue

            url = self._gateway_ws_url()
            timestamp = self.identity.next_timestamp()
            headers = self.identity.sign_request(timestamp, "")

            try:
                async with self.session.ws_connect(url, headers=headers) as ws:
                    log.info("Connected to gateway websocket at %s", url)
                    backoff = 1.0
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await self._handle_ws_message(msg.data, ws)
                        elif msg.type in (
                            aiohttp.WSMsgType.ERROR,
                            aiohttp.WSMsgType.CLOSED,
                        ):
                            break
            except asyncio.CancelledError:
                return
            except Exception as exc:
                log.warning("Gateway websocket error: %s", exc)

            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 10)

    async def _handle_ws_message(self, raw: str, ws: aiohttp.ClientWebSocketResponse):
        try:
            payload = json.loads(raw)
        except Exception:
            log.warning("Invalid WS payload (not JSON)")
            return

        if payload.get("type") != "job":
            return

        asyncio.create_task(self._handle_job(payload, ws))

    async def _handle_job(self, payload: Dict[str, Any], ws: aiohttp.ClientWebSocketResponse):
        job_id = payload.get("job_id")
        model_id = payload.get("model_id")
        messages = payload.get("messages") or []
        if not isinstance(messages, list):
            messages = []
        block_range = payload.get("block_range")
        max_tokens = self._coerce_int(payload.get("max_tokens"))
        temperature = self._coerce_float(payload.get("temperature"))

        if not job_id or not model_id:
            await self._send_ws(
                ws,
                {
                    "type": "error",
                    "job_id": job_id or "unknown",
                    "message": "invalid_job_request",
                },
            )
            return

        needs_refresh = (
            not self.current_assignment
            or self.current_assignment.get("model_id") != model_id
            or (
                block_range
                and self.current_assignment.get("block_range") != block_range
            )
        )
        if needs_refresh:
            await self._refresh_assignment()

        if not self.current_assignment:
            await self._send_ws(
                ws,
                {
                    "type": "error",
                    "job_id": job_id,
                    "message": "no_assignment",
                },
            )
            return

        if self.current_assignment.get("model_id") != model_id:
            await self._send_ws(
                ws,
                {
                    "type": "error",
                    "job_id": job_id,
                    "message": "assignment_model_mismatch",
                },
            )
            return

        if block_range and self.current_assignment.get("block_range") != block_range:
            await self._send_ws(
                ws,
                {
                    "type": "error",
                    "job_id": job_id,
                    "message": "assignment_block_mismatch",
                },
            )
            return

        if self.config.agent.mock_inference:
            async with self._inference_lock:
                async for token in self._run_mock_inference(messages):
                    await self._send_ws(
                        ws, {"type": "token", "job_id": job_id, "token": token}
                    )
                await self._send_ws(
                    ws,
                    {"type": "done", "job_id": job_id, "finish_reason": "stop"},
                )
            return

        if not self.petals.is_running():
            await self._send_ws(
                ws,
                {
                    "type": "error",
                    "job_id": job_id,
                    "message": "petals_not_running",
                },
            )
            return

        log.info("Starting inference: job_id=%s model_id=%s", job_id, model_id)
        async with self._inference_lock:
            try:
                token_count = 0
                async for token in self._run_inference(
                    job_id=job_id,
                    model_id=model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ):
                    token_count += 1
                    await self._send_ws(
                        ws, {"type": "token", "job_id": job_id, "token": token}
                    )
                log.info(
                    "Inference done: job_id=%s tokens_sent=%d ws_closed=%s",
                    job_id, token_count, ws.closed,
                )
                await self._send_ws(
                    ws,
                    {"type": "done", "job_id": job_id, "finish_reason": "stop"},
                )
            except Exception as exc:
                log.exception("Inference job failed: job_id=%s model_id=%s", job_id, model_id)
                await self._send_ws(
                    ws,
                    {
                        "type": "error",
                        "job_id": job_id,
                        "message": str(exc),
                    },
                )

    async def _run_mock_inference(self, messages: list[str]) -> AsyncIterator[str]:
        user_text = ""
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_text = msg.get("content", "") or ""
                break
        response = (
            "Mock response (no Petals). "
            "Your last message was: "
            f"{user_text[:120]}"
        )
        for token in response.split():
            await asyncio.sleep(0.02)
            yield token + " "

    async def _send_ws(
        self, ws: aiohttp.ClientWebSocketResponse, payload: Dict[str, Any]
    ) -> None:
        async with self._ws_send_lock:
            if ws.closed:
                return
            await ws.send_json(payload)

    # ------------------------------------------------------------------
    # Inference worker helpers
    # ------------------------------------------------------------------

    def _get_or_write_worker_script(self) -> str:
        """Write the inference worker script to a temp file (once) and return its path."""
        if self._inference_worker_script and os.path.exists(self._inference_worker_script):
            return self._inference_worker_script
        fd, path = tempfile.mkstemp(suffix=".py", prefix="beam_inference_worker_")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(_INFERENCE_WORKER_SCRIPT)
        except Exception:
            os.close(fd)
            raise
        log.info("Inference worker script written to: %s", path)
        self._inference_worker_script = path
        return path

    def _ensure_inference_worker(self) -> _InferenceSubprocess:
        """Return the inference worker, creating it if necessary."""
        if self._inference_worker is None:
            petals_python = os.environ.get("BEAM_PETALS_PYTHON") or sys.executable
            script_path = self._get_or_write_worker_script()

            # Build extra env vars for the worker subprocess.
            extra_env: Dict[str, str] = {}

            # Pass beam DHT initial_peers (from current assignment) so the worker
            # connects to the beam private swarm rather than the public Petals swarm.
            assignment_peers = (
                self.current_assignment.get("initial_peers")
                if self.current_assignment
                else None
            )
            if assignment_peers:
                extra_env["BEAM_INFERENCE_INITIAL_PEERS"] = json.dumps(assignment_peers)
                log.info("Inference worker will use %d beam DHT peer(s)", len(assignment_peers))
            else:
                log.info(
                    "No beam DHT peers in assignment — inference worker will use "
                    "PUBLIC_INITIAL_PEERS (public Petals swarm)"
                )

            # Pre-warm: instruct the worker to load the model immediately at startup
            # so the first real inference request doesn't have to wait for disk I/O.
            if self.current_assignment:
                warmup_model = self.current_assignment.get("model_id", "")
                if warmup_model:
                    extra_env["BEAM_INFERENCE_WARMUP_MODEL"] = warmup_model
                    log.info("Inference worker will pre-load model '%s' at startup.", warmup_model)

            self._inference_worker = _InferenceSubprocess(
                petals_python, script_path, extra_env=extra_env
            )
            log.info(
                "Created inference worker (python=%s, script=%s)",
                petals_python, script_path,
            )
        return self._inference_worker

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def _run_inference(
        self,
        *,
        job_id: str = "",
        model_id: str,
        messages: list,
        max_tokens: Optional[int],
        temperature: Optional[float],
    ) -> AsyncIterator[str]:
        """
        Run a single inference job via the persistent worker subprocess.

        The worker runs under BEAM_PETALS_PYTHON (the real petals-venv Python),
        so it has full access to torch, cmath, and all other C-extension stdlib
        modules — none of which are available inside the PyInstaller binary process.
        """
        prompt = self._format_messages(messages)
        max_new_tokens = max_tokens if max_tokens and max_tokens > 0 else 256
        temp_value = 1.0 if temperature is None else float(temperature)

        worker = self._ensure_inference_worker()
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _thread():
            try:
                for msg in worker.run_job(
                    job_id=job_id or "job",
                    model_id=model_id,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temp_value,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, msg)
            except Exception as exc:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"type": "error", "message": str(exc)},
                )

        threading.Thread(target=_thread, daemon=True).start()

        while True:
            msg = await queue.get()
            mtype = msg.get("type")
            if mtype == "token":
                yield msg.get("token", "")
            elif mtype == "done":
                return
            else:
                tb = msg.get("traceback", "")
                if tb:
                    log.error("InferenceWorker traceback:\n%s", tb)
                raise Exception(msg.get("message", "inference error"))

    def _format_messages(self, messages: list) -> str:
        lines = []
        for msg in messages or []:
            role = msg.get("role") if isinstance(msg, dict) else None
            content = msg.get("content") if isinstance(msg, dict) else str(msg)
            if content is None:
                continue
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        parts.append(str(part.get("text", "")))
                    else:
                        parts.append(str(part))
                content = " ".join(p for p in parts if p)
            else:
                content = str(content)
            role_label = (role or "user").capitalize()
            lines.append(f"{role_label}: {content}")
        lines.append("Assistant:")
        return "\n".join(lines).strip()

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    async def _loop(self):
        while True:
            start_time = time.time()

            await self._heartbeat()
            await self._refresh_assignment()  # Or rely on heartbeat response

            # Sleep remainder of interval
            elapsed = time.time() - start_time
            sleep_time = max(0, self.config.agent.heartbeat_interval_sec - elapsed)
            await asyncio.sleep(sleep_time)

    async def _register(self, pairing_token_override: Optional[str] = None):
        async with self._register_lock:
            url = f"{self.config.control_plane.url}/api/v1/nodes/register"
            log.info(f"Registering at {url}...")

            if (
                "onion" in self.config.agent.transports
                and not self.config.agent.onion_address
            ):
                raise ValueError(
                    "onion_address must be configured when using onion transport"
                )

            payload = {
                "protocol_version": "1.0",
                "machine_fingerprint": self.identity.machine_fingerprint,
                "gpu": self._gpu_spec,
                "software": {
                    "node_agent_version": "0.1.0",
                    "petals_version": "2.3.0",
                },
                "transports": self.config.agent.transports,
                "capabilities": self._normalized_capabilities(),
            }

            if self.config.agent.onion_address:
                payload["onion_address"] = self.config.agent.onion_address
            pairing_token = pairing_token_override or self.config.agent.pairing_token
            if pairing_token:
                payload["pairing_token"] = pairing_token

            async with self.session.post(url, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    log.error("Registration failed: %s - %s", resp.status, text)
                    raise Exception(f"Registration failed: {resp.status} - {text}")

                data = await resp.json()
                self.identity.save_state(data["node_id"], data["node_secret"])

                assignment = data.get("assignment")
                if isinstance(assignment, dict):
                    await self._apply_assignment(assignment)

    async def _link_node_with_token(self, pairing_token: str) -> tuple[bool, str, int]:
        if not self.identity.node_id:
            return False, "Node is not registered", 400
        if not self.session:
            return False, "Client session not available", 500

        url = f"{self.config.control_plane.url}/api/v1/beam/renters/link"
        payload = {"token": pairing_token, "node_id": self.identity.node_id}

        async with self._register_lock:
            async with self.session.post(url, json=payload) as resp:
                try:
                    data = await resp.json()
                except Exception:
                    data = {}

                if resp.status == 200:
                    return True, "Node linked to renter.", 200

                message = (
                    data.get("error", {}).get("message")
                    or data.get("detail")
                    or "Failed to link node"
                )
                return False, message, resp.status

    async def _link_existing_node(self, pairing_token: str) -> web.Response:
        linked, message, status = await self._link_node_with_token(pairing_token)
        if linked:
            return web.json_response(
                {
                    "status": "linked",
                    "node_id": self.identity.node_id,
                    "message": message,
                },
                headers=self._cors_headers(),
            )

        return web.json_response(
            {"status": "error", "message": message},
            status=status,
            headers=self._cors_headers(),
        )

    async def _heartbeat(self):
        if not self.identity.node_id:
            return

        timestamp = self.identity.next_timestamp()
        status = "running" if self.petals.is_running() else "degraded"
        uptime_sec = max(0, timestamp - self.started_at)
        petals_info = self.petals.status_snapshot()

        payload = {
            "protocol_version": "1.0",
            "node_id": self.identity.node_id,
            "timestamp": timestamp,
            "status": status,
            "metrics": {
                "uptime_sec": uptime_sec,
                "tokens_processed": 0,
                "req_ok": 0,
                "req_err": 0,
                "p50_latency_ms": 0,
                "p95_latency_ms": 0,
                "petals_status": "running" if petals_info.get("running") else "stopped",
                "petals_uptime_sec": petals_info.get("uptime_sec"),
                "petals_last_exit_code": petals_info.get("last_exit_code"),
                "petals_last_exit_at": petals_info.get("last_exit_at"),
            },
            "active_jobs": [],
            "current_assignment": self.current_assignment or None,
        }

        body_json = json.dumps(payload)
        headers = self.identity.sign_request(timestamp, body_json)
        headers["Content-Type"] = "application/json"

        url = f"{self.config.control_plane.url}/api/v1/nodes/heartbeat"
        try:
            async with self.session.post(url, data=body_json, headers=headers) as resp:
                if resp.status == 200:
                    log.debug("Heartbeat OK")
                elif resp.status in (401, 403):
                    # Auth failed - maybe node deleted?
                    log.error("Heartbeat auth failed. Identity might be invalid.")
                else:
                    log.warning(f"Heartbeat failed: {resp.status}")
        except Exception as e:
            log.error(f"Heartbeat connection error: {e}")

    async def _refresh_assignment(self):
        if not self.identity.node_id:
            return

        # Explicit GET for assignment
        url = f"{self.config.control_plane.url}/api/v1/nodes/{self.identity.node_id}/assignment"

        # GET requests need empty body signature
        timestamp = self.identity.next_timestamp()
        headers = self.identity.sign_request(timestamp, "")

        try:
            async with self.session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    await self._apply_assignment(data)
                else:
                    log.warning(f"Failed to fetch assignment: {resp.status}")
        except Exception as e:
            log.error(f"Assignment fetch error: {e}")

    async def _apply_assignment(self, assignment: Dict):
        new_mid = assignment.get("model_id")
        new_range = assignment.get("block_range")
        new_epoch = int(assignment.get("assignment_epoch", 0))

        if not new_mid or not isinstance(new_range, list) or len(new_range) != 2:
            return
        if new_range[0] < 0 or new_range[1] <= new_range[0]:
            return

        current_mid = (
            self.current_assignment.get("model_id") if self.current_assignment else None
        )
        current_range = (
            self.current_assignment.get("block_range")
            if self.current_assignment
            else None
        )
        current_epoch = (
            int(self.current_assignment.get("assignment_epoch", 0))
            if self.current_assignment
            else 0
        )

        changed = (
            new_mid != current_mid
            or new_range != current_range
            or new_epoch != current_epoch
        )
        if changed:
            # initial_peers: list of multiaddrs from the backend (beam DHT bootstrap).
            # Falls back to None which means the petals server / inference worker
            # will use PUBLIC_INITIAL_PEERS.
            new_peers: Optional[list] = assignment.get("initial_peers") or None
            if isinstance(new_peers, list) and not new_peers:
                new_peers = None  # empty list → treat as "not provided"

            log.info(f"New assignment received: {new_mid} blocks {new_range}")
            self.current_assignment = {
                "model_id": new_mid,
                "block_range": [int(new_range[0]), int(new_range[1])],
                "assignment_epoch": new_epoch,
                "initial_peers": new_peers,
            }

            if not self.config.agent.mock_inference:
                range_str = f"{new_range[0]}:{new_range[1]}"
                log.info(f"Applying new assignment: model={new_mid}, range={range_str}")
                self.petals.stop()
                try:
                    self.petals.start(new_mid, range_str, initial_peers=new_peers)
                except Exception as e:
                    log.error(f"Failed to start Petals for {new_mid} [{range_str}]: {e}")
                    return

                if self._petals_check_task:
                    self._petals_check_task.cancel()
                self._petals_check_task = asyncio.create_task(
                    self._check_petals_liveness(new_mid, range_str)
                )
                # Pre-warm the inference worker once petals is ready.
                asyncio.create_task(self._prewarm_inference_worker())
            # Stop the inference worker subprocess so it reloads the new model.
            if self._inference_worker is not None:
                log.info("Assignment changed — stopping inference worker subprocess.")
                self._inference_worker.stop()
                self._inference_worker = None

    async def _prewarm_inference_worker(self) -> None:
        """
        Wait until the petals server is running, then pre-start the inference worker
        so that the model is loaded from disk/cache before the first user request arrives.
        This eliminates the 30-60 second 'cold start' delay users would otherwise experience.
        """
        try:
            # Poll until petals is running (max 5 minutes).
            for _ in range(300):
                if self.petals.is_running():
                    break
                await asyncio.sleep(1)
            else:
                log.warning("Pre-warm: petals server did not become ready within 5 minutes.")
                return

            if self.config.agent.mock_inference:
                return

            log.info("Pre-warming inference worker (petals server is ready).")
            loop = asyncio.get_running_loop()
            # Run worker creation in a thread so it doesn't block the event loop.
            await loop.run_in_executor(None, self._ensure_inference_worker)
            log.info("Inference worker pre-warm complete — model is loading in the background.")
        except asyncio.CancelledError:
            return
        except Exception as exc:
            log.warning("Inference worker pre-warm failed: %s", exc)

    async def _check_petals_liveness(self, model_id: str, block_range: str) -> None:
        try:
            await asyncio.sleep(60)
            if (
                self.current_assignment
                and self.current_assignment.get("model_id") == model_id
                and self.petals.is_running() is False
            ):
                log.warning(
                    "Petals did not stay running for 60s (model=%s, blocks=%s)",
                    model_id,
                    block_range,
                )
        except asyncio.CancelledError:
            return
