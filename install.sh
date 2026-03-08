#!/usr/bin/env bash
set -euo pipefail

echo "Beam Node Agent Quickstart"
echo "This will download the agent, write a config, and start pairing."
echo "Keep this terminal open to see the pairing code."
echo

if [[ "${BEAM_ACCEPT_DEFAULTS:-}" == "true" ]]; then
  confirm="Y"
else
  read -r -p "Continue? [Y/n]: " confirm < /dev/tty || confirm="Y"
fi
confirm=${confirm:-Y}
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
  echo "Aborted."
  exit 1
fi

control_plane_url="${BEAM_CONTROL_PLANE_URL:-https://www.openbeam.me/}"
control_plane_url="${control_plane_url%/}"
echo "Control plane URL: $control_plane_url"

release_base_default="https://github.com/beam-open-node/beam_agent/releases/latest/download"
if [[ "${BEAM_ACCEPT_DEFAULTS:-}" == "true" ]]; then
  release_base=""
else
  read -r -p "Release base URL [$release_base_default]: " release_base < /dev/tty || release_base=""
fi
release_base=${release_base:-$release_base_default}
release_base="${release_base%/}"

os="$(uname -s)"
case "$os" in
  Linux*) asset_name="beam-node-agent-linux" ;;
  Darwin*) asset_name="beam-node-agent-macos" ;;
  *)
    echo "Unsupported OS: $os"
    exit 1
    ;;
esac

binary_path="./$asset_name"
download_url="$release_base/$asset_name"

# -----------------------------------------------------------------------
# BEAM_RUN_FROM_SOURCE=true  — run the agent from Python source instead
# of the pre-built binary.  Use this when a new binary hasn't been built
# yet, or when you want to test local changes without a full PyInstaller
# build.
#
# What it does:
#   1. Clones / updates the source from GitHub (or uses BEAM_SOURCE_DIR).
#   2. Creates a minimal "beam-agent-venv" with only the three packages
#      the agent itself needs (aiohttp, pyyaml, pydantic>=2).
#   3. Generates start_agent.sh that runs:
#        beam-agent-venv/bin/python -m beam_node_agent.main --config …
#
# The heavy ML work (torch / petals) is still done in the SEPARATE
# BEAM_PETALS_PYTHON subprocess, exactly as in the binary.
# -----------------------------------------------------------------------
if [[ "${BEAM_RUN_FROM_SOURCE:-false}" == "true" ]]; then
  source_repo="${BEAM_SOURCE_REPO:-https://github.com/beam-open-node/beam_agent}"
  source_dir="${BEAM_SOURCE_DIR:-./beam_agent_src}"

  if [[ -d "$source_dir/.git" ]]; then
    echo "Updating agent source in $source_dir …"
    git -C "$source_dir" pull --ff-only || git -C "$source_dir" fetch --all
  else
    echo "Cloning agent source from $source_repo into $source_dir …"
    git clone "$source_repo" "$source_dir"
  fi

  agent_venv="${BEAM_AGENT_VENV_DIR:-./beam-agent-venv}"
  if [[ ! -d "$agent_venv" ]]; then
    echo "Creating agent venv at $agent_venv …"
    python3 -m venv "$agent_venv"
    "$agent_venv/bin/pip" install --upgrade pip
    "$agent_venv/bin/pip" install aiohttp pyyaml "pydantic>=2.0"
  else
    echo "Using existing agent venv at $agent_venv"
  fi

  # Assemble start_agent.sh (source-based variant)
  {
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    [[ -n "${BEAM_PETALS_PYTHON:-}" ]] && echo "export BEAM_PETALS_PYTHON=\"${BEAM_PETALS_PYTHON}\""
    echo "export BEAM_CONTROL_PLANE_URL=\"$control_plane_url\""
    echo "export CONTROL_PLANE_URL=\"$control_plane_url\""
    [[ -n "${BEAM_HOP_COUNTS:-}" ]] && echo "export BEAM_HOP_COUNTS=\"${BEAM_HOP_COUNTS}\""
    [[ -n "${BEAM_SINGLE_NODE:-}" ]] && echo "export BEAM_SINGLE_NODE=\"${BEAM_SINGLE_NODE}\""
    [[ -n "${BEAM_MAX_BLOCKS:-}" ]] && echo "export BEAM_MAX_BLOCKS=\"${BEAM_MAX_BLOCKS}\""
    echo "cd \"$(realpath "$source_dir")\""
    echo "exec \"$(realpath "$agent_venv")/bin/python\" -m beam_node_agent.main --config \"$(realpath "$config_path")\""
  } > start_agent.sh
  chmod +x start_agent.sh

  echo
  echo "Source-based start script written to start_agent.sh"
  echo "Running: ./start_agent.sh"
  exec ./start_agent.sh
fi

# -----------------------------------------------------------------------
# Default path: download the pre-built binary from GitHub Releases.
# -----------------------------------------------------------------------
if [[ -f "$binary_path" ]]; then
  if [[ "${BEAM_ACCEPT_DEFAULTS:-}" == "true" ]]; then
    redownload="N"
  else
    read -r -p "$binary_path exists. Redownload? [y/N]: " redownload < /dev/tty || redownload="N"
  fi
  redownload=${redownload:-N}
  if [[ "$redownload" =~ ^[Yy]$ ]]; then
    curl -fL "$download_url" -o "$binary_path"
    echo "Downloaded $binary_path"
  else
    echo "Using existing $binary_path"
  fi
else
  curl -fL "$download_url" -o "$binary_path"
  echo "Downloaded $binary_path"
fi

chmod +x "$binary_path"

if [[ "${BEAM_ACCEPT_DEFAULTS:-}" == "true" ]]; then
  config_path="config.yaml"
else
  read -r -p "Config path [config.yaml] (press Enter for default): " config_path < /dev/tty || config_path=""
fi
case "${config_path,,}" in
  "" )
    config_path="config.yaml"
    ;;
  y|yes|n|no )
    echo "Using default config.yaml"
    config_path="config.yaml"
    ;;
esac

if [[ -f "$config_path" ]]; then
  if [[ "${BEAM_ACCEPT_DEFAULTS:-}" == "true" ]]; then
    overwrite="N"
  else
    read -r -p "$config_path exists. Overwrite? [y/N]: " overwrite < /dev/tty || overwrite="N"
  fi
  overwrite=${overwrite:-N}
  if [[ "$overwrite" =~ ^[Yy]$ ]]; then
    cat > "$config_path" <<EOF
control_plane:
  url: "$control_plane_url"

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
EOF
    echo "Wrote $config_path"
  else
    echo "Using existing $config_path"
  fi
else
  cat > "$config_path" <<EOF
control_plane:
  url: "$control_plane_url"

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
EOF
  echo "Wrote $config_path"
fi

echo
echo "Next: the agent will start and print a 6-digit pair code."
echo "Open the Rent Panel and enter the code to link this machine."
ensure_petals_runtime() {
  if [[ -n "${BEAM_PETALS_PYTHON:-}" ]]; then
    echo "Using BEAM_PETALS_PYTHON=$BEAM_PETALS_PYTHON"
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    if python3 - <<'PY' >/dev/null 2>&1
import petals  # noqa: F401
PY
    then
      BEAM_PETALS_PYTHON="$(command -v python3)"
      export BEAM_PETALS_PYTHON
      echo "Using system python for Petals: $BEAM_PETALS_PYTHON"
      return 0
    fi
  fi

  if [[ "${BEAM_SKIP_PETALS_SETUP:-}" == "true" ]]; then
    echo "Petals not found. Set BEAM_PETALS_PYTHON or install petals."
    return 1
  fi

  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 not found. Please install Python 3 to set up Petals."
    return 1
  fi

  petals_venv="${BEAM_PETALS_VENV_DIR:-./beam-petals-venv}"
  # Normalize to a clean absolute path (avoid leading '//' which can break pip installs).
  petals_venv="$(
    BEAM_PETALS_VENV_DIR_RESOLVE="$petals_venv" python3 - <<'PY'
import os
path = os.environ["BEAM_PETALS_VENV_DIR_RESOLVE"]
path = os.path.expanduser(path)
path = os.path.abspath(path)
if path.startswith("//"):
    path = "/" + path.lstrip("/")
print(path)
PY
  )"
  echo "Installing Petals runtime in $petals_venv"
  recreate_venv="${BEAM_PETALS_RECREATE_VENV:-true}"
  if [[ -d "$petals_venv" && "$recreate_venv" == "true" ]]; then
    echo "Recreating existing Petals runtime at $petals_venv"
    rm -rf "$petals_venv"
  fi
  python3 -m venv "$petals_venv"
  "$petals_venv/bin/python" -m pip install --upgrade pip "setuptools<70" wheel
  if ! "$petals_venv/bin/python" - <<'PY' >/dev/null 2>&1
import pkg_resources  # noqa: F401
PY
  then
    echo "setuptools missing pkg_resources; reinstalling a compatible version"
    "$petals_venv/bin/python" -m pip install --upgrade "setuptools<70"
  fi
  site_pkgs="$("$petals_venv/bin/python" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"
  if ! "$petals_venv/bin/python" -m pip install --upgrade --force-reinstall --no-deps "typing_extensions>=4.12"; then
    echo "Cleaning stale typing_extensions files"
    rm -f "$site_pkgs/typing_extensions.py" "$site_pkgs/typing_extensions.pyi"
    rm -rf "$site_pkgs/typing_extensions-"*.dist-info
    "$petals_venv/bin/python" -m pip install --upgrade --force-reinstall --no-deps "typing_extensions>=4.12"
  fi
  "$petals_venv/bin/python" -m pip install --upgrade --no-deps grpcio "protobuf>=3.12.2,<7.0.0" grpcio-tools
  torch_spec="${BEAM_PETALS_TORCH_SPEC:-torch>=2.4,<2.6}"
  torchvision_spec="${BEAM_PETALS_TORCHVISION_SPEC:-torchvision>=0.19,<0.21}"
  torchaudio_spec="${BEAM_PETALS_TORCHAUDIO_SPEC:-torchaudio>=2.4,<2.6}"
  # Auto-detect CUDA version and set index URL if not provided
  if [[ -z "${BEAM_PETALS_TORCH_INDEX_URL:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    cuda_ver="$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || true)"
    if [[ -n "$cuda_ver" ]]; then
      cuda_major="${cuda_ver%%.*}"
      if [[ "$cuda_major" -ge 12 ]]; then
        BEAM_PETALS_TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
        echo "Auto-detected CUDA $cuda_ver; using torch index: $BEAM_PETALS_TORCH_INDEX_URL"
      elif [[ "$cuda_major" -ge 11 ]]; then
        BEAM_PETALS_TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
        echo "Auto-detected CUDA $cuda_ver; using torch index: $BEAM_PETALS_TORCH_INDEX_URL"
      fi
    fi
  fi
  if [[ "${BEAM_PETALS_SKIP_TORCH_INSTALL:-}" != "true" ]]; then
    if [[ -n "${BEAM_PETALS_TORCH_INDEX_URL:-}" ]]; then
      "$petals_venv/bin/python" -m pip install --upgrade "$torch_spec" "$torchvision_spec" "$torchaudio_spec" --index-url "$BEAM_PETALS_TORCH_INDEX_URL"
    else
      "$petals_venv/bin/python" -m pip install --upgrade "$torch_spec" "$torchvision_spec" "$torchaudio_spec"
    fi
  fi
  hf_hub_spec="${BEAM_PETALS_HF_HUB_SPEC:-huggingface-hub>=0.28.0}"
  transformers_spec="${BEAM_PETALS_TRANSFORMERS_SPEC:-transformers>=5.2.0,<6.0}"
  numpy_spec="${BEAM_PETALS_NUMPY_SPEC:-numpy<2}"
  pydantic_spec="${BEAM_PETALS_PYDANTIC_SPEC:-pydantic>=2.0.0}"
  accelerate_specs="${BEAM_PETALS_ACCELERATE_SPECS:-}"
  if [[ -z "$accelerate_specs" && -n "${BEAM_PETALS_ACCELERATE_SPEC:-}" ]]; then
    accelerate_specs="${BEAM_PETALS_ACCELERATE_SPEC}"
  fi
  if [[ -z "$accelerate_specs" ]]; then
    accelerate_specs="1.3.0 1.2.0 1.1.0 1.0.0 0.34.2 0.34.1 0.31.0 0.30.1 0.29.3 0.28.0 0.27.2 0.26.1 0.25.0 0.24.1 0.23.0 0.22.0"
  fi
  
  petals_pip_args=()
  if [[ "${BEAM_PETALS_PIP_NO_BUILD_ISOLATION:-true}" == "true" ]]; then
    petals_pip_args+=(--no-build-isolation)
  fi

  # Install petals from the beam repo (already patched for transformers 5.x).
  # We avoid PyPI petals entirely — it pins transformers<4.35 and hf-hub<1.0.
  beam_repo_url="${BEAM_REPO_URL:-https://github.com/fuegocoding/beam.git}"
  beam_repo_ref="${BEAM_REPO_REF:-main}"
  echo "Installing petals from beam repo: $beam_repo_url @ $beam_repo_ref"
  "$petals_venv/bin/python" -m pip install --no-deps --no-cache-dir \
    "petals @ git+${beam_repo_url}@${beam_repo_ref}#subdirectory=engine"

  # Install all runtime deps at the correct versions (no stale pins from PyPI petals).
  "$petals_venv/bin/python" -m pip install "${petals_pip_args[@]}" \
    "$hf_hub_spec" \
    "$numpy_spec" \
    "$pydantic_spec" \
    "$transformers_spec" \
    "bitsandbytes>=0.43.0" \
    "accelerate>=0.33.0" \
    "tensor-parallel==1.0.23" \
    "tokenizers>=0.19.0" \
    "packaging>=20.9" \
    "sentencepiece>=0.1.99" \
    "peft>=0.12.0" \
    "safetensors>=0.4.0" \
    "aiohttp" "pyyaml" \
    "cpufeature" \
    "dijkstar>=2.6.0" \
    "async-timeout>=4.0" \
    "humanfriendly>=10.0" \
    "requests>=2.28.0" \
    "speedtest-cli==2.1.3" \
    || true
  hivemind_spec="${BEAM_PETALS_HIVEMIND_SPEC:-hivemind @ git+https://github.com/learning-at-home/hivemind.git@213bff98a62accb91f254e2afdccbf1d69ebdea9}"
  hivemind_pip_args=()
  if [[ "${BEAM_PETALS_HIVEMIND_PIP_NO_BUILD_ISOLATION:-true}" == "true" ]]; then
    hivemind_pip_args+=(--no-build-isolation)
  fi
  if [[ "${BEAM_PETALS_HIVEMIND_NO_DEPS:-true}" == "true" ]]; then
    hivemind_pip_args+=(--no-deps)
  fi
  "$petals_venv/bin/python" -m pip install --upgrade --force-reinstall \
    "${hivemind_pip_args[@]}" \
    "$hivemind_spec"
  # Install hivemind runtime dependencies that --no-deps skips
  "$petals_venv/bin/python" -m pip install --upgrade \
    "configargparse>=1.2.3" \
    "cryptography>=3.4.6" \
    "msgpack>=0.5.6" \
    "multiaddr>=0.0.9" \
    "prefetch-generator>=1.0.1" \
    "pymultihash>=0.8.2" \
    "scipy>=1.2.1" \
    "sortedcontainers" \
    "psutil" \
    "protobuf>=3.12.2,<7.0.0"
  if [[ "$(uname -s)" == "Linux" ]]; then
    "$petals_venv/bin/python" -m pip install --upgrade "uvloop>=0.14.0"
  fi
  "$petals_venv/bin/python" -m pip install --upgrade --force-reinstall "$pydantic_spec"
  accelerate_ok="false"
  for accelerate_ver in $accelerate_specs; do
    accelerate_pkg="$accelerate_ver"
    if [[ "$accelerate_pkg" != accelerate==* ]]; then
      accelerate_pkg="accelerate==$accelerate_pkg"
    fi
    echo "Trying accelerate runtime: $accelerate_pkg"
    if ! "$petals_venv/bin/python" -m pip install --upgrade --force-reinstall --no-deps "$accelerate_pkg"; then
      continue
    fi
    if "$petals_venv/bin/python" - <<'PY' >/dev/null 2>&1
import accelerate  # noqa: F401
from accelerate import init_empty_weights  # noqa: F401
PY
    then
      accelerate_ok="true"
      echo "Using accelerate runtime: $accelerate_pkg"
      break
    fi
  done
  if [[ "$accelerate_ok" != "true" ]]; then
    echo "ERROR: accelerate runtime is incompatible with the pinned Petals stack."
    echo "Tried versions: $accelerate_specs"
    echo "Set BEAM_PETALS_ACCELERATE_SPECS to override the fallback list."
    return 1
  fi
  if ! "$petals_venv/bin/python" - <<'PY' >/dev/null 2>&1
import numpy
if int(numpy.__version__.split(".")[0]) >= 2:
    raise RuntimeError(f"incompatible numpy version: {numpy.__version__}")
PY
  then
    echo "NumPy 2.x detected; forcing NumPy < 2 for torch/hivemind compatibility"
    "$petals_venv/bin/python" -m pip install --upgrade --force-reinstall "numpy<2"
  fi
  if ! "$petals_venv/bin/python" - <<'PY' >/dev/null 2>&1
import hivemind  # noqa: F401
from hivemind.optim.grad_scaler import GradScaler  # noqa: F401
PY
  then
    echo "Detected incompatible torch/hivemind combination. Reinstalling a compatible torch stack."
    torch_compat_spec="${BEAM_PETALS_TORCH_COMPAT_SPEC:-torch>=2.4,<2.6}"
    torchvision_compat_spec="${BEAM_PETALS_TORCHVISION_COMPAT_SPEC:-torchvision>=0.19,<0.21}"
    torchaudio_compat_spec="${BEAM_PETALS_TORCHAUDIO_COMPAT_SPEC:-torchaudio>=2.4,<2.6}"
    if [[ -n "${BEAM_PETALS_TORCH_INDEX_URL:-}" ]]; then
      "$petals_venv/bin/python" -m pip install --upgrade --force-reinstall \
        "$torch_compat_spec" "$torchvision_compat_spec" "$torchaudio_compat_spec" \
        --index-url "$BEAM_PETALS_TORCH_INDEX_URL"
    else
      "$petals_venv/bin/python" -m pip install --upgrade --force-reinstall \
        "$torch_compat_spec" "$torchvision_compat_spec" "$torchaudio_compat_spec"
    fi
    "$petals_venv/bin/python" -m pip install --upgrade --force-reinstall "numpy<2" "setuptools<70"
    "$petals_venv/bin/python" -m pip install --upgrade --force-reinstall \
      "${hivemind_pip_args[@]}" \
      "$hivemind_spec"
    # Reinstall hivemind runtime dependencies after --no-deps reinstall
    "$petals_venv/bin/python" -m pip install --upgrade \
      "configargparse>=1.2.3" \
      "cryptography>=3.4.6" \
      "msgpack>=0.5.6" \
      "multiaddr>=0.0.9" \
      "prefetch-generator>=1.0.1" \
      "pymultihash>=0.8.2" \
      "scipy>=1.2.1" \
      "sortedcontainers" \
      "psutil" \
      "protobuf>=3.12.2,<7.0.0"
    if [[ "$(uname -s)" == "Linux" ]]; then
      "$petals_venv/bin/python" -m pip install --upgrade "uvloop>=0.14.0"
    fi
  fi
  if ! "$petals_venv/bin/python" - <<'PY' >/dev/null 2>&1
import hivemind  # noqa: F401
from hivemind.optim.grad_scaler import GradScaler  # noqa: F401
PY
  then
    echo "ERROR: Petals runtime is still incompatible (torch/hivemind)."
    echo "Try setting BEAM_PETALS_TORCH_COMPAT_SPEC and rerun install.sh."
    return 1
  fi
  if ! "$petals_venv/bin/python" - <<'PY' >/dev/null 2>&1
import numpy
if int(numpy.__version__.split(".")[0]) >= 2:
    raise RuntimeError(f"incompatible numpy version: {numpy.__version__}")
PY
  then
    echo "ERROR: Petals runtime is still incompatible (numpy)."
    return 1
  fi
  # -- Frontier model overlay -----------------------------------------------------
  # Models (qwen3, qwen3_5_moe, glm_moe_dsa, etc.) are already built into the
  # beam repo's petals source. No inline patching needed.
  echo "Frontier models already included in beam repo petals — skipping inline patcher."
  # -- End frontier model overlay -------------------------------------------------

  # Verify petals imports correctly after all deps are installed
  if ! "$petals_venv/bin/python" -c "import petals; print(f'petals {petals.__version__} imported OK')" 2>&1; then
    echo "WARNING: petals import check failed. Check dependency compatibility."
  fi

  petals_python_real="$petals_venv/bin/python"
  petals_model_shim="${BEAM_PETALS_ENABLE_MODEL_ARG_SHIM:-true}"
  if [[ "$petals_model_shim" == "true" ]]; then
    petals_python_shim="$petals_venv/bin/beam-petals-python"
    cat > "$petals_python_shim" <<EOF
#!/usr/bin/env bash
set -euo pipefail
real_python="$petals_python_real"
if [[ "\$#" -ge 4 && "\$1" == "-m" && "\$2" == "petals.cli.run_server" && "\$3" == "--model" ]]; then
  model_id="\$4"
  shift 4
  exec "\$real_python" -m petals.cli.run_server "\$model_id" "\$@"
fi
exec "\$real_python" "\$@"
EOF
    chmod +x "$petals_python_shim"
    export BEAM_PETALS_PYTHON="$petals_python_shim"
  else
    export BEAM_PETALS_PYTHON="$petals_python_real"
  fi
}

echo

if [[ "${BEAM_ACCEPT_DEFAULTS:-}" == "true" ]]; then
  start_now="Y"
else
  read -r -p "Start the agent now? [Y/n]: " start_now < /dev/tty || start_now="Y"
fi
start_now=${start_now:-Y}
if [[ ! "$start_now" =~ ^[Yy]$ ]]; then
echo "You can start it later with:"
  echo "./start_agent.sh"
  exit 0
fi

ensure_petals_runtime

if [[ -n "${BEAM_PETALS_PYTHON:-}" ]]; then
  petals_site_pkgs="$("$BEAM_PETALS_PYTHON" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"
  if [[ -d "$petals_site_pkgs" ]]; then
    export BEAM_PETALS_SITE_PACKAGES="$petals_site_pkgs"
    if [[ "${BEAM_EXPORT_PYTHONPATH:-false}" == "true" ]]; then
      if [[ -n "${PYTHONPATH:-}" ]]; then
        export PYTHONPATH="$petals_site_pkgs:$PYTHONPATH"
      else
        export PYTHONPATH="$petals_site_pkgs"
      fi
      echo "Exported PYTHONPATH with Petals runtime: $petals_site_pkgs"
    else
      echo "Using Petals runtime via BEAM_PETALS_SITE_PACKAGES=$petals_site_pkgs"
    fi
  fi
fi

if [[ "${BEAM_SINGLE_NODE:-}" == "true" ]]; then
  export BEAM_HOP_COUNTS="A=1,B=1,C=1"
  export BEAM_MAX_BLOCKS=40
  echo "Single-node mode enabled. BEAM_HOP_COUNTS set to $BEAM_HOP_COUNTS and BEAM_MAX_BLOCKS to $BEAM_MAX_BLOCKS"
fi

{
  echo '#!/usr/bin/env bash'
  [[ -n "${BEAM_PETALS_PYTHON:-}" ]] && echo "export BEAM_PETALS_PYTHON=\"${BEAM_PETALS_PYTHON}\""
  echo "export BEAM_CONTROL_PLANE_URL=\"$control_plane_url\""
  echo "export CONTROL_PLANE_URL=\"$control_plane_url\""
  [[ -n "${BEAM_HOP_COUNTS:-}" ]] && echo "export BEAM_HOP_COUNTS=\"${BEAM_HOP_COUNTS}\""
  [[ -n "${BEAM_SINGLE_NODE:-}" ]] && echo "export BEAM_SINGLE_NODE=\"${BEAM_SINGLE_NODE}\""
  [[ -n "${BEAM_MAX_BLOCKS:-}" ]] && echo "export BEAM_MAX_BLOCKS=\"${BEAM_MAX_BLOCKS}\""
  echo "exec \"$binary_path\" --config \"$config_path\""
} > start_agent.sh
chmod +x start_agent.sh

echo "Running: ./start_agent.sh"
exec ./start_agent.sh


