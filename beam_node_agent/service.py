import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import time
from typing import Any, AsyncIterator, Dict, Optional

import aiohttp
from aiohttp import web

from .config import BeamConfig
from .node_identity import NodeIdentity
from .petals_wrapper import PetalsWrapper

log = logging.getLogger(__name__)


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
        self._model_lock = asyncio.Lock()
        self._inference_model_id: Optional[str] = None
        self._inference_model: Optional[object] = None
        self._inference_tokenizer: Optional[object] = None
        self._petals_check_task: Optional[asyncio.Task] = None
        # Set to True when a torch/petals C-extension import corruption is detected.
        # Once corrupted, the process must be restarted; we short-circuit further attempts.
        self._inference_import_failed: bool = False
        self._inference_import_error: Optional[str] = None

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

        async with self._inference_lock:
            try:
                async for token in self._run_inference(
                    model_id=model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ):
                    await self._send_ws(
                        ws, {"type": "token", "job_id": job_id, "token": token}
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

    async def _run_inference(
        self,
        *,
        model_id: str,
        messages: list,
        max_tokens: Optional[int],
        temperature: Optional[float],
    ) -> AsyncIterator[str]:
        prompt = self._format_messages(messages)
        max_new_tokens = max_tokens if max_tokens and max_tokens > 0 else 256
        temp_value = 1.0 if temperature is None else float(temperature)
        do_sample = temp_value > 0

        model, tokenizer = await self._get_inference_model(model_id)
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _worker():
            try:
                from transformers import TextIteratorStreamer

                inputs = tokenizer(prompt, return_tensors="pt")
                try:
                    inputs = inputs.to(model.device)
                except Exception:
                    pass

                streamer = TextIteratorStreamer(
                    tokenizer, skip_prompt=True, skip_special_tokens=True
                )

                gen_kwargs = {
                    **inputs,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "streamer": streamer,
                }
                if do_sample:
                    gen_kwargs["temperature"] = temp_value

                thread = threading.Thread(
                    target=model.generate, kwargs=gen_kwargs, daemon=True
                )
                thread.start()
                for token in streamer:
                    loop.call_soon_threadsafe(queue.put_nowait, ("token", token))
                thread.join()
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))

        threading.Thread(target=_worker, daemon=True).start()

        while True:
            kind, value = await queue.get()
            if kind == "token":
                yield value
            elif kind == "done":
                return
            else:
                raise Exception(value)

    async def _get_inference_model(self, model_id: str):
        # Short-circuit immediately if a prior attempt corrupted torch C-extensions.
        # Re-importing torch in the same process after this failure is impossible;
        # the node agent process must be restarted.
        if self._inference_import_failed:
            raise RuntimeError(
                f"Petals/torch import is permanently broken in this process "
                f"({self._inference_import_error}). "
                "Please restart the beam-node-agent to recover."
            )

        async with self._model_lock:
            if (
                self._inference_model_id == model_id
                and self._inference_model
                and self._inference_tokenizer
            ):
                return self._inference_model, self._inference_tokenizer

            loop = asyncio.get_running_loop()
            model, tokenizer = await loop.run_in_executor(
                None, self._load_model_sync, model_id
            )
            self._inference_model_id = model_id
            self._inference_model = model
            self._inference_tokenizer = tokenizer
            return model, tokenizer

    def _petals_python_sysconfig_paths(self) -> Dict[str, str]:
        petals_python = os.environ.get("BEAM_PETALS_PYTHON")
        if not petals_python:
            return {}
        try:
            raw = subprocess.check_output(
                [
                    petals_python,
                    "-c",
                    (
                        "import json, sysconfig; "
                        "paths = sysconfig.get_paths(); "
                        "print(json.dumps({k: paths.get(k, '') for k in "
                        "('purelib', 'platlib', 'stdlib', 'platstdlib')}))"
                    ),
                ],
                text=True,
                timeout=10,
            ).strip()
        except Exception:
            return {}
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        if not isinstance(parsed, dict):
            return {}

        normalized: Dict[str, str] = {}
        for key, value in parsed.items():
            if isinstance(value, str) and value.strip():
                normalized[str(key)] = value.strip()
        return normalized

    @staticmethod
    def _normalize_existing_paths(candidates: list[str]) -> list[str]:
        normalized: list[str] = []
        for path in candidates:
            resolved = os.path.realpath(path)
            if os.path.isdir(resolved) and resolved not in normalized:
                normalized.append(resolved)
        return normalized

    def _petals_site_package_candidates(self) -> list[str]:
        candidates: list[str] = []

        site_pkgs_env = os.environ.get("BEAM_PETALS_SITE_PACKAGES")
        if site_pkgs_env:
            candidates.extend(
                segment.strip()
                for segment in site_pkgs_env.split(os.pathsep)
                if segment.strip()
            )

        sysconfig_paths = self._petals_python_sysconfig_paths()
        for key in ("purelib", "platlib"):
            path = sysconfig_paths.get(key)
            if path:
                candidates.append(path)

        return self._normalize_existing_paths(candidates)

    def _petals_stdlib_candidates(self) -> list[str]:
        candidates: list[str] = []

        sysconfig_paths = self._petals_python_sysconfig_paths()
        for key in ("stdlib", "platstdlib"):
            path = sysconfig_paths.get(key)
            if path:
                candidates.append(path)

        expanded: list[str] = []
        for path in candidates:
            expanded.append(path)
            lib_dynload = os.path.join(path, "lib-dynload")
            if os.path.isdir(lib_dynload):
                expanded.append(lib_dynload)

        return self._normalize_existing_paths(expanded)

    def _prepare_inference_imports(self) -> None:
        # ------------------------------------------------------------------ #
        # PRE-LOAD critical stdlib C-extension modules BEFORE any sys.path    #
        # manipulation.  In a PyInstaller binary the normal built-in/frozen   #
        # importer resolves these via the embedded archive; once sys.path is  #
        # rewritten to point at the petals venv's lib-dynload the file-system #
        # importer takes over and may not find them (e.g. 'cmath', 'math').   #
        # Caching them in sys.modules first avoids the ModuleNotFoundError    #
        # that otherwise surfaces deep inside torch/testing/_comparison.py.   #
        # ------------------------------------------------------------------ #
        _stdlib_preload = (
            "cmath", "math", "statistics", "fractions", "decimal",
            "unicodedata", "_decimal", "_json", "json", "struct", "_struct",
            "array", "select", "ssl", "socket", "_socket", "binascii",
            "_codecs", "codecs", "io", "_io", "abc",
        )
        for _mod_name in _stdlib_preload:
            if _mod_name not in sys.modules:
                try:
                    __import__(_mod_name)
                    log.debug("Pre-loaded stdlib module: %s", _mod_name)
                except ImportError:
                    log.debug("Stdlib module not available for pre-load: %s", _mod_name)

        petals_paths = self._petals_site_package_candidates()
        stdlib_paths = self._petals_stdlib_candidates()
        inference_paths = petals_paths + stdlib_paths

        if inference_paths:
            # Ensure the Petals runtime paths win import resolution.
            for path in reversed(inference_paths):
                if path in sys.path:
                    sys.path.remove(path)
                sys.path.insert(0, path)

        # Guard against accidental ".../numpy" entries which trigger numpy source-tree errors.
        cleaned_path: list[str] = []
        removed: list[str] = []
        for entry in sys.path:
            resolved = os.path.realpath(entry) if entry else entry
            if resolved and os.path.basename(resolved.rstrip(os.sep)) == "numpy":
                removed.append(entry)
                continue
            cleaned_path.append(entry)
        if removed:
            log.warning("Removed suspicious sys.path entries before inference import: %s", removed)
        sys.path[:] = cleaned_path

        # Always evict core ML modules before inference imports to avoid partial-import residue.
        for root_name in (
            "numpy",
            "torch",
            "petals",
            "hivemind",
            "transformers",
            "accelerate",
            "huggingface_hub",
            "tokenizers",
            "pydantic",
            "pydantic_core",
            "annotated_types",
        ):
            for loaded_name in list(sys.modules):
                if loaded_name == root_name or loaded_name.startswith(f"{root_name}."):
                    sys.modules.pop(loaded_name, None)

    def _load_model_sync(self, model_id: str):
        self._prepare_inference_imports()

        try:
            import numpy as _numpy
        except Exception as exc:
            raise RuntimeError(
                "failed to import numpy from the Petals runtime; "
                f"BEAM_PETALS_SITE_PACKAGES={os.environ.get('BEAM_PETALS_SITE_PACKAGES', '')}"
            ) from exc

        log.info("Inference runtime numpy: %s (%s)", _numpy.__version__, getattr(_numpy, "__file__", "unknown"))

        try:
            from petals import AutoDistributedModelForCausalLM
            from petals.constants import DTYPE_MAP, PUBLIC_INITIAL_PEERS
            from transformers import AutoTokenizer
            import torch
        except Exception as exc:
            # Detect C-extension corruption that cannot be recovered in this process.
            # Symptoms:
            #   - "already has a docstring" – torch._C was partially initialised and
            #     cannot be re-imported after a previous failed attempt.
            #   - "No module named 'cmath'" – a stdlib C-extension was not pre-cached
            #     and is now unreachable through the rewritten sys.path.
            # In both cases, every subsequent import attempt in this process will
            # fail with the same (or a derived) error.  Mark the flag so that
            # _get_inference_model short-circuits further jobs and reports a clear
            # restart message instead of spamming identical tracebacks.
            exc_str = str(exc)
            is_permanent = (
                "already has a docstring" in exc_str
                or "No module named 'cmath'" in exc_str
                or "ModuleNotFoundError" in type(exc).__name__
            )
            if is_permanent:
                self._inference_import_failed = True
                self._inference_import_error = exc_str
                log.error(
                    "Petals/torch C-extension import is permanently broken in this "
                    "process.  Root cause: %s.  "
                    "The node agent process must be restarted to recover.",
                    exc_str,
                )
            raise RuntimeError(
                "failed to import Petals runtime dependencies in node agent process"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        if tokenizer.pad_token is None and tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token

        log.info(f"Loading Distributed Model: {model_id}...")
        model = AutoDistributedModelForCausalLM.from_pretrained(
            model_id,
            initial_peers=PUBLIC_INITIAL_PEERS,
            torch_dtype=DTYPE_MAP.get("float16", torch.float16),
        )
        log.info(f"Model {model_id} loaded successfully.")
        model.eval()
        return model, tokenizer

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
            log.info(f"New assignment received: {new_mid} blocks {new_range}")
            self.current_assignment = {
                "model_id": new_mid,
                "block_range": [int(new_range[0]), int(new_range[1])],
                "assignment_epoch": new_epoch,
            }

            if not self.config.agent.mock_inference:
                range_str = f"{new_range[0]}:{new_range[1]}"
                log.info(f"Applying new assignment: model={new_mid}, range={range_str}")
                self.petals.stop()
                try:
                    self.petals.start(new_mid, range_str)
                except Exception as e:
                    log.error(f"Failed to start Petals for {new_mid} [{range_str}]: {e}")
                    return

                if self._petals_check_task:
                    self._petals_check_task.cancel()
                self._petals_check_task = asyncio.create_task(
                    self._check_petals_liveness(new_mid, range_str)
                )
            if self._inference_model_id and self._inference_model_id != new_mid:
                self._inference_model_id = None
                self._inference_model = None
                self._inference_tokenizer = None

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
