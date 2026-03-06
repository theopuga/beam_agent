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
  transformers_spec="${BEAM_PETALS_TRANSFORMERS_SPEC:-transformers>=4.51.0,<5.0}"
  numpy_spec="${BEAM_PETALS_NUMPY_SPEC:-numpy<2}"
  pydantic_spec="${BEAM_PETALS_PYDANTIC_SPEC:-pydantic<2.0.0}"
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

  # Install petals with a transformers version it supports, then upgrade
  # transformers separately (petals caps transformers<5 but Qwen3.5 MoE needs >=5.2).
  "$petals_venv/bin/python" -m pip install --upgrade --force-reinstall \
    "$hf_hub_spec" \
    "$numpy_spec" \
    "${petals_pip_args[@]}" \
    petals
  "$petals_venv/bin/python" -m pip install --upgrade "$transformers_spec"
  if [[ "${BEAM_PETALS_SKIP_TORCH_INSTALL:-}" != "true" ]]; then
    if [[ -n "${BEAM_PETALS_TORCH_INDEX_URL:-}" ]]; then
      "$petals_venv/bin/python" -m pip install --upgrade --force-reinstall \
        "$torch_spec" "$torchvision_spec" "$torchaudio_spec" \
        --index-url "$BEAM_PETALS_TORCH_INDEX_URL"
    else
      "$petals_venv/bin/python" -m pip install --upgrade --force-reinstall \
        "$torch_spec" "$torchvision_spec" "$torchaudio_spec"
    fi
  fi
  "$petals_venv/bin/python" -m pip install --upgrade --force-reinstall "numpy<2" "setuptools<70"
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
  patch_falcon_mqa="${BEAM_PETALS_PATCH_FALCON_MQA:-true}"
  if [[ "$patch_falcon_mqa" == "true" ]]; then
    falcon_block_path="$("$petals_venv/bin/python" - <<'PY'
import pathlib
import sysconfig

site_pkgs = pathlib.Path(sysconfig.get_paths()["purelib"])
candidate = site_pkgs / "petals" / "models" / "falcon" / "block.py"
print(candidate if candidate.exists() else "")
PY
)"
    if [[ -n "$falcon_block_path" && -f "$falcon_block_path" ]]; then
      patch_result="$("$petals_venv/bin/python" - "$falcon_block_path" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()
fixed = "num_kv_heads = self.num_heads if self.new_decoder_architecture else self.num_kv_heads"
legacy = "num_kv_heads = self.num_heads"
if fixed in text:
    print("already-patched")
    raise SystemExit(0)
if legacy not in text:
    print("pattern-not-found")
    raise SystemExit(0)
text = text.replace(legacy, fixed, 1)
path.write_text(text)
print("patched")
PY
)"
      case "$patch_result" in
        patched)
          echo "Patched Petals Falcon multi-query attention bug in $falcon_block_path"
          ;;
        already-patched)
          echo "Petals Falcon multi-query patch already present"
          ;;
        pattern-not-found)
          echo "Petals Falcon file pattern not found; skipping patch (likely already fixed upstream)"
          ;;
        *)
          echo "Petals Falcon patch returned unexpected status: $patch_result"
          ;;
      esac
    fi
  fi
  # Patch petals server/from_pretrained.py: get_file_from_repo was removed from
  # transformers in newer versions; shim it via huggingface_hub directly.
  "$petals_venv/bin/python" - <<'PY'
import pathlib, sysconfig
site_pkgs = pathlib.Path(sysconfig.get_paths()["purelib"])
path = site_pkgs / "petals" / "server" / "from_pretrained.py"
if not path.exists():
    print("petals/server/from_pretrained.py not found; skipping")
    raise SystemExit(0)
old = "from transformers.utils import get_file_from_repo"
new = """try:
    from transformers.utils import get_file_from_repo
except ImportError:
    import os as _os
    from huggingface_hub import hf_hub_download as _hf_dl
    def get_file_from_repo(path_or_repo, filename, **kw):
        kw.pop("use_auth_token", None)
        if _os.path.isdir(path_or_repo):
            p = _os.path.join(path_or_repo, filename)
            return p if _os.path.exists(p) else None
        return _hf_dl(path_or_repo, filename, **kw)"""
text = path.read_text()
if old not in text:
    print("petals get_file_from_repo shim already applied or pattern not found")
else:
    path.write_text(text.replace(old, new, 1))
    print("Patched petals/server/from_pretrained.py: shimmed get_file_from_repo")
PY

  # Patch petals __init__.py to remove the hard transformers version assertion
  # (petals 2.x asserts transformers<4.35 but Beam needs newer transformers).
  "$petals_venv/bin/python" - <<'PY'
import pathlib, sysconfig
site_pkgs = pathlib.Path(sysconfig.get_paths()["purelib"])
init_path = site_pkgs / "petals" / "__init__.py"
if not init_path.exists():
    print("petals __init__.py not found; skipping version assertion patch")
    raise SystemExit(0)
marker = "version.parse(transformers.__version__)"
text = init_path.read_text()
if marker not in text:
    print("petals transformers version assertion not found or already patched")
    raise SystemExit(0)
lines = text.splitlines(keepends=True)
# Find the line index containing the marker
marker_idx = next(i for i, l in enumerate(lines) if marker in l)
# Walk backward to find the assert keyword
start = marker_idx
while start > 0 and "assert" not in lines[start]:
    start -= 1
# Walk forward to find the end of the assert (closing paren + optional message)
end = marker_idx
paren_depth = sum(l.count("(") - l.count(")") for l in lines[start:end + 1])
while end + 1 < len(lines) and (paren_depth > 0 or lines[end].rstrip().endswith("\\")):
    end += 1
    paren_depth += lines[end].count("(") - lines[end].count(")")
indent = len(lines[start]) - len(lines[start].lstrip())
out = (
    lines[:start]
    + [" " * indent + "pass  # assertion removed by Beam installer\n"]
    + lines[end + 1:]
)
init_path.write_text("".join(out))
print("Patched petals __init__.py: removed transformers version assertion")
PY

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
import pydantic
if int(pydantic.__version__.split(".")[0]) >= 2:
    raise RuntimeError(f"incompatible pydantic version: {pydantic.__version__}")
PY
  then
    echo "Pydantic 2.x detected; forcing pydantic < 2 for hivemind compatibility"
    "$petals_venv/bin/python" -m pip install --upgrade --force-reinstall "$pydantic_spec"
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
    "$petals_venv/bin/python" -m pip install --upgrade --force-reinstall "$pydantic_spec"
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
  if ! "$petals_venv/bin/python" - <<'PY' >/dev/null 2>&1
import pydantic
if int(pydantic.__version__.split(".")[0]) >= 2:
    raise RuntimeError(f"incompatible pydantic version: {pydantic.__version__}")
PY
  then
    echo "ERROR: Petals runtime is still incompatible (pydantic/hivemind)."
    return 1
  fi
  # -- Frontier model overlay: inject model support into installed petals --------
  patch_frontier_models="${BEAM_PETALS_PATCH_FRONTIER_MODELS:-true}"
  if [[ "$patch_frontier_models" == "true" ]]; then
    # Try standalone patch script first (available when running from source tree)
    _patch_script=""
    for _candidate in \
        "$(dirname "${BASH_SOURCE[0]}")/scripts/patch_petals_models.py" \
        "./scripts/patch_petals_models.py"; do
      if [[ -f "$_candidate" ]]; then
        _patch_script="$_candidate"
        break
      fi
    done
    if [[ -n "$_patch_script" ]]; then
      echo "Running frontier model patcher: $_patch_script"
      "$petals_venv/bin/python" "$_patch_script"
    else
      echo "Patch script not found; using inline patcher"
      "$petals_venv/bin/python" - <<'QWEN_PATCH'
import pathlib, sys, sysconfig, textwrap

site_pkgs = pathlib.Path(sysconfig.get_paths()["purelib"])
models_dir = site_pkgs / "petals" / "models"
if not models_dir.exists():
    print("petals models dir not found; skipping Qwen3.5 MoE patch")
    sys.exit(0)

qwen_dir = models_dir / "qwen3_5_moe"
qwen_dir.mkdir(exist_ok=True)

# ---------- __init__.py ----------
(qwen_dir / "__init__.py").write_text(textwrap.dedent("""\
    from petals.models.qwen3_5_moe.block import WrappedQwen3_5MoeBlock
    from petals.models.qwen3_5_moe.config import DistributedQwen3_5MoeConfig
    from petals.models.qwen3_5_moe.model import (
        DistributedQwen3_5MoeForCausalLM,
        DistributedQwen3_5MoeModel,
    )
    from petals.utils.auto_config import register_model_classes

    register_model_classes(
        config=DistributedQwen3_5MoeConfig,
        model=DistributedQwen3_5MoeModel,
        model_for_causal_lm=DistributedQwen3_5MoeForCausalLM,
    )
"""))

# ---------- config.py ----------
(qwen_dir / "config.py").write_text(textwrap.dedent("""\
    import os
    from typing import Optional, Union
    from hivemind import get_logger
    from transformers import AutoConfig
    from petals.client.config import ClientConfig
    from petals.client.lm_head import LMHeadConfig
    from petals.client.ptune import PTuneConfig
    from petals.models.qwen3_5_moe.block import WrappedQwen3_5MoeBlock
    logger = get_logger(__name__)
    try:
        from transformers.models.qwen3_5_moe import Qwen3_5MoeTextConfig as _BaseConfig
    except ImportError:
        from transformers.models.qwen3_5_moe import Qwen3_5MoeConfig as _BaseConfig
    try:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeTextAttention as _AttnClass,
        )
    except ImportError:
        _AttnClass = None

    class DistributedQwen3_5MoeConfig(_BaseConfig, ClientConfig, PTuneConfig, LMHeadConfig):
        block_class = WrappedQwen3_5MoeBlock
        attn_class = _AttnClass
        block_prefix = "model.layers"
        @property
        def num_key_value_groups(self):
            return self.num_attention_heads // self.num_key_value_heads
        @classmethod
        def from_pretrained(cls, model_name_or_path, *args, dht_prefix=None, **kwargs):
            loading_from_repo = model_name_or_path is not None and not os.path.isdir(str(model_name_or_path))
            if loading_from_repo and dht_prefix is None:
                dht_prefix = str(model_name_or_path).split("/")[-1].replace(".", "-")
                if not dht_prefix.endswith("-hf"):
                    dht_prefix += "-hf"
                logger.info(f"Using DHT prefix: {dht_prefix}")
            result = super().from_pretrained(model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)
            config = result[0] if isinstance(result, tuple) else result
            config.use_cache = True
            if config.pad_token_id is None:
                config.pad_token_id = 0
            return result
"""))

# ---------- block.py ----------
(qwen_dir / "block.py").write_text(textwrap.dedent("""\
    from typing import Optional, Tuple
    import torch
    from transformers.cache_utils import DynamicCache
    try:
        from transformers.modeling_attn_mask_utils import (
            _prepare_4d_causal_attention_mask,
            _prepare_4d_causal_attention_mask_for_sdpa,
        )
    except ImportError:
        _prepare_4d_causal_attention_mask = None
        _prepare_4d_causal_attention_mask_for_sdpa = None
    try:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeTextDecoderLayer as _DecoderLayer,
        )
    except ImportError:
        try:
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
                Qwen3_5MoeDecoderLayer as _DecoderLayer,
            )
        except ImportError:
            _DecoderLayer = None
    try:
        from transformers.models.qwen3_5_moe import Qwen3_5MoeTextConfig as _Config
    except ImportError:
        from transformers.models.qwen3_5_moe import Qwen3_5MoeConfig as _Config

    if _DecoderLayer is not None:
        class WrappedQwen3_5MoeBlock(_DecoderLayer):
            def __init__(self, config, layer_idx):
                super().__init__(config, layer_idx)
                self._attn_implementation = getattr(config, "_attn_implementation", "eager")
                self.layer_idx = layer_idx
            def forward(self, hidden_states, *args, attention_mask=None, layer_past=None, use_cache=False, **kwargs):
                batch_size, seq_length, _ = hidden_states.shape
                seq_length_with_past = seq_length
                past_key_values_length = 0
                past_key_value = layer_past
                if past_key_value is not None:
                    past_key_values_length = past_key_value[0].shape[2]
                    seq_length_with_past += past_key_values_length
                    _pkv = self._reorder_cache_from_bloom(past_key_value, batch_size, past_key_values_length)
                    past_key_value = DynamicCache()
                    past_key_value.key_cache = [torch.empty(0) for _ in range(self.layer_idx)] + [_pkv[0]]
                    past_key_value.value_cache = [torch.empty(0) for _ in range(self.layer_idx)] + [_pkv[1]]
                    past_key_value._seen_tokens = past_key_values_length
                if self._attn_implementation == "flash_attention_2":
                    attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
                elif self._attn_implementation == "sdpa" and _prepare_4d_causal_attention_mask_for_sdpa:
                    attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length)
                elif _prepare_4d_causal_attention_mask:
                    attention_mask = _prepare_4d_causal_attention_mask(attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length)
                position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=hidden_states.device).unsqueeze(0).view(-1, seq_length)
                outputs = super().forward(hidden_states, *args, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_value, use_cache=use_cache, **kwargs)
                if use_cache:
                    present_key_value = outputs[-1]
                    present_key_value = present_key_value[self.layer_idx]
                    present_key_value = self._reorder_cache_to_bloom(present_key_value, batch_size, seq_length_with_past)
                    outputs = outputs[:-1] + (present_key_value,)
                return outputs
            def _reorder_cache_from_bloom(self, key_value, batch_size, seq_length):
                key_states, value_states = key_value
                num_kv = getattr(self.self_attn, "num_key_value_heads", getattr(self.self_attn, "num_heads", 1))
                hd = getattr(self.self_attn, "head_dim", key_states.shape[-1])
                key_states = key_states.permute(0, 2, 1).view(batch_size, num_kv, seq_length, hd)
                value_states = value_states.view(*key_states.shape)
                return (key_states, value_states)
            def _reorder_cache_to_bloom(self, key_value, batch_size, seq_length):
                key_states, value_states = key_value
                num_kv = getattr(self.self_attn, "num_key_value_heads", getattr(self.self_attn, "num_heads", 1))
                hd = getattr(self.self_attn, "head_dim", key_states.shape[-1])
                value_states = value_states.view(batch_size * num_kv, seq_length, hd)
                key_states = key_states.view(*value_states.shape).permute(0, 2, 1)
                return (key_states, value_states)
    else:
        class WrappedQwen3_5MoeBlock:
            def __init__(self, *a, **kw):
                raise ImportError("Qwen3.5 MoE requires transformers >= 5.2.0")
"""))

# ---------- model.py ----------
(qwen_dir / "model.py").write_text(textwrap.dedent("""\
    from typing import Optional
    import torch, torch.nn as nn
    from hivemind import DHT
    from hivemind.utils.logging import get_logger
    from transformers.modeling_outputs import MoeModelOutputWithPast
    try:
        from transformers.models.qwen3_5_moe import (
            Qwen3_5MoeTextModel,
            Qwen3_5MoeForCausalLM,
            Qwen3_5MoePreTrainedModel,
        )
    except ImportError:
        Qwen3_5MoeTextModel = None
        Qwen3_5MoeForCausalLM = None
        Qwen3_5MoePreTrainedModel = None
    from petals.client.from_pretrained import FromPretrainedMixin
    from petals.client.lm_head import LMHead
    from petals.client.ptune import PTuneMixin
    from petals.client.remote_generation import RemoteGenerationMixin, RemotePastKeyValues
    from petals.client.remote_sequential import RemoteSequential
    from petals.models.qwen3_5_moe.config import DistributedQwen3_5MoeConfig
    from petals.utils.auto_config import DefaultRevisionMixin
    logger = get_logger(__name__)

    if Qwen3_5MoeTextModel is not None:
        class DistributedQwen3_5MoeModel(DefaultRevisionMixin, FromPretrainedMixin, PTuneMixin, Qwen3_5MoeTextModel):
            _keys_to_ignore_on_load_missing = PTuneMixin._keys_to_ignore_on_load_missing
            _keys_to_ignore_on_load_unexpected = [r"^model\\\\.layers\\\\."]
            config_class = DistributedQwen3_5MoeConfig
            def __init__(self, config, *, dht=None):
                n_layer, config.num_hidden_layers = config.num_hidden_layers, 0
                super().__init__(config)
                assert len(self.layers) == 0
                config.num_hidden_layers = n_layer
                self.layers = RemoteSequential(config, dht=dht)
                self.requires_grad_(False)
                self.init_prompts(config)
            def forward(self, input_ids=None, past_key_values=None, attention_mask=None,
                        position_ids=None, head_mask=None, inputs_embeds=None, use_cache=None,
                        output_attentions=None, output_hidden_states=None,
                        output_router_logits=None, return_dict=None, cache_position=None):
                if input_ids is not None and inputs_embeds is not None:
                    raise ValueError("Cannot specify both input_ids and inputs_embeds")
                elif input_ids is not None:
                    input_shape = input_ids.size()
                    input_ids = input_ids.view(-1, input_shape[-1])
                elif inputs_embeds is not None:
                    input_shape = inputs_embeds.size()[:-1]
                else:
                    raise ValueError("Must specify either input_ids or inputs_embeds")
                assert attention_mask is None or (attention_mask == 1).all()
                assert use_cache is None or use_cache
                assert not output_attentions
                assert not output_hidden_states
                assert return_dict is None or return_dict
                if inputs_embeds is None:
                    inputs_embeds = self.embed_tokens(input_ids)
                use_prompts = self.config.tuning_mode and "ptune" in self.config.tuning_mode and self.h.position == 0
                if use_prompts:
                    prompts, intermediate_prompts = self.get_prompt(inputs_embeds.shape[0])
                    inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)
                else:
                    prompts = intermediate_prompts = None
                hidden_states = inputs_embeds
                output_shape = input_shape + (hidden_states.size(-1),)
                if past_key_values is None:
                    past_key_values = RemotePastKeyValues()
                past_key_values.update_seen(hidden_states.size(1))
                hidden_states = self.layers(hidden_states, prompts=intermediate_prompts,
                    hypo_ids=past_key_values.hypo_ids if past_key_values is not None else None)
                if use_prompts:
                    hidden_states = hidden_states[:, self.pre_seq_len:]
                hidden_states = self.norm(hidden_states)
                hidden_states = hidden_states.view(output_shape)
                return MoeModelOutputWithPast(last_hidden_state=hidden_states,
                    past_key_values=past_key_values, hidden_states=None, attentions=None)
            @property
            def word_embeddings(self): return self.embed_tokens
            @property
            def word_embeddings_layernorm(self): return nn.Identity()
            @property
            def h(self): return self.layers
            @property
            def ln_f(self): return self.norm

        class DistributedQwen3_5MoeForCausalLM(FromPretrainedMixin, RemoteGenerationMixin, Qwen3_5MoeForCausalLM):
            _keys_to_ignore_on_load_missing = DistributedQwen3_5MoeModel._keys_to_ignore_on_load_missing
            _keys_to_ignore_on_load_unexpected = DistributedQwen3_5MoeModel._keys_to_ignore_on_load_unexpected
            config_class = DistributedQwen3_5MoeConfig
            def __init__(self, config):
                Qwen3_5MoePreTrainedModel.__init__(self, config)
                self.model = DistributedQwen3_5MoeModel(config)
                self.lm_head = LMHead(config)
                self.post_init()
            def get_output_embeddings(self): return self.lm_head
            @property
            def transformer(self): return self.model
    else:
        class DistributedQwen3_5MoeModel:
            def __init__(self, *a, **kw): raise ImportError("Qwen3.5 MoE requires transformers >= 5.2.0")
        class DistributedQwen3_5MoeForCausalLM:
            def __init__(self, *a, **kw): raise ImportError("Qwen3.5 MoE requires transformers >= 5.2.0")
"""))

# ---------- Update models/__init__.py ----------
models_init = models_dir / "__init__.py"
text = models_init.read_text()
marker = "from petals.models.qwen3_5_moe import *"
if marker not in text:
    text += "\ntry:\n    from petals.models.qwen3_5_moe import *\nexcept ImportError:\n    pass  # Qwen3.5 MoE requires transformers >= 5.2.0\n"
    models_init.write_text(text)
    print("Patched petals models/__init__.py with Qwen3.5 MoE support")
else:
    print("Qwen3.5 MoE already registered in petals models/__init__.py")

print("Qwen3.5 MoE overlay applied to petals")
QWEN_PATCH
    fi  # end inline patcher fallback
  fi
  # -- End frontier model overlay -------------------------------------------------

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


