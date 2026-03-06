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
  # Re-pin torch/numpy BEFORE the import check: the petals install above uses
  # --force-reinstall which upgrades torch to its latest (2.10+), breaking
  # hivemind's grad_scaler import (OptState removed in torch>=2.6).
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
  if ! "$petals_venv/bin/python" - <<'PY' >/dev/null 2>&1
import petals  # noqa: F401
import petals.cli  # noqa: F401
PY
  then
    echo "ERROR: petals is not importable after installation."
    echo "The pip install may have failed partway. Try reinstalling manually:"
    echo "  $petals_venv/bin/pip install --force-reinstall petals"
    return 1
  fi
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
#!/usr/bin/env python3
"""Inline patcher — mirrors beam_agent/scripts/patch_petals_models.py exactly."""
import pathlib
import sys
import sysconfig
import textwrap

site_pkgs = pathlib.Path(sysconfig.get_paths()["purelib"])
models_dir = site_pkgs / "petals" / "models"
utils_dir = site_pkgs / "petals" / "utils"

if not models_dir.exists():
    print("petals models dir not found; skipping model patches")
    sys.exit(0)

BLOCK_TEMPLATE = textwrap.dedent("""\
from typing import Optional, Tuple
import torch
from transformers.cache_utils import DynamicCache

{decoder_import}

if _DecoderLayer is not None:
    class {block_class}(_DecoderLayer):
        def __init__(self, config, layer_idx=0):
            super().__init__(config, layer_idx)
            # Ensure attn_implementation is set (Qwen3Config may leave it None)
            self._attn_implementation = getattr(config, "_attn_implementation", "eager") or "eager"
            if not getattr(config, "_attn_implementation", None):
                config._attn_implementation = "eager"
            self.layer_idx = layer_idx
            # Pre-compute RoPE for transformers>=5.x (position_embeddings required by attention)
            self._beam_rope = None
            try:
                import importlib, inspect
                mod = importlib.import_module(type(self).__mro__[1].__module__)
                for name in dir(mod):
                    if "rotary" in name.lower() and "embedding" in name.lower():
                        cls = getattr(mod, name)
                        if isinstance(cls, type):
                            sig = inspect.signature(cls.__init__)
                            if "config" in sig.parameters:
                                self._beam_rope = cls(config=config)
                            else:
                                self._beam_rope = cls(
                                    config.hidden_size // config.num_attention_heads,
                                    config.max_position_embeddings,
                                )
                            break
            except Exception:
                pass

        def forward(self, hidden_states, *args, attention_mask=None, layer_past=None, use_cache=False, **kwargs):
            bs, sl, _ = hidden_states.shape
            pkvl = 0
            sl_wp = sl

            # Always create a fresh DynamicCache; populate with past KV when available
            pkv = DynamicCache()
            if layer_past is not None:
                # beam format: k=(bs*nkv, hd, pkvl), v=(bs*nkv, pkvl, hd)
                pkvl = layer_past[0].shape[2]
                sl_wp = sl + pkvl
                k_m, v_m = self._rcfb(layer_past, bs, pkvl)
                pkv.update(k_m, v_m, layer_idx=self.layer_idx)

            pos_ids = torch.arange(pkvl, sl + pkvl, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
            if "position_embeddings" not in kwargs and self._beam_rope is not None:
                kwargs["position_embeddings"] = self._beam_rope(hidden_states, pos_ids)

            # transformers>=5.x: past_key_values (plural), DynamicCache updated in-place,
            # returns tuple (hidden_states, [opt: attn_weights]) -- no past_key_values in output
            outputs = super().forward(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=pos_ids,
                past_key_values=pkv,
                use_cache=use_cache,
                **kwargs
            )
            h = outputs[0]  # hidden_states is always first element

            if use_cache:
                # DynamicCache updated in-place; extract via key_cache/value_cache lists
                k_out = pkv.key_cache[self.layer_idx]    # (bs, nkv, sl_wp, hd)
                v_out = pkv.value_cache[self.layer_idx]  # (bs, nkv, sl_wp, hd)
                beam_kv = self._rctb((k_out, v_out), bs, sl_wp)
                return (h, beam_kv)
            return (h,)

        def _rcfb(self, kv, bs, sl):
            # beam format -> model format: k (bs*nkv, hd, sl) -> (bs, nkv, sl, hd)
            k, v = kv
            nkv = k.shape[0] // bs   # derived from tensor shape (avoids missing attr)
            hd = k.shape[1]
            k = k.permute(0, 2, 1).view(bs, nkv, sl, hd)
            v = v.view(bs, nkv, sl, hd)
            return k, v

        def _rctb(self, kv, bs, sl):
            # model format (bs, nkv, sl, hd) -> beam format
            k, v = kv
            nkv = k.shape[1]   # derived from tensor shape
            hd = k.shape[3]
            k = k.view(bs * nkv, sl, hd).permute(0, 2, 1)
            v = v.view(bs * nkv, sl, hd)
            return k, v
else:
    class {block_class}:
        def __init__(self, *a, **kw):
            raise ImportError("{model_name} requires a newer transformers version")
""")

CONFIG_TEMPLATE = textwrap.dedent("""\
import os
from hivemind import get_logger
from petals.client.config import ClientConfig
from petals.client.lm_head import LMHeadConfig
from petals.client.ptune import PTuneConfig
from petals.models.{pkg}.block import {block_class}
logger = get_logger(__name__)

{base_config_import}
{attn_import}

class {config_class}(_BaseConfig, ClientConfig, PTuneConfig, LMHeadConfig):
    block_class = {block_class}
    attn_class = _AttnClass
    block_prefix = "model.layers"

    @property
    def num_key_value_groups(self):
        nkv = getattr(self, "num_key_value_heads", self.num_attention_heads)
        return self.num_attention_heads // nkv

    @classmethod
    def from_pretrained(cls, model_name_or_path, *args, dht_prefix=None, **kwargs):
        if model_name_or_path and not os.path.isdir(str(model_name_or_path)) and not dht_prefix:
            dht_prefix = str(model_name_or_path).split("/")[-1].replace(".", "-")
            if not dht_prefix.endswith("-hf"):
                dht_prefix += "-hf"
            logger.info(f"Using DHT prefix: {{dht_prefix}}")
        result = super().from_pretrained(model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)
        config = result[0] if isinstance(result, tuple) else result
        config.use_cache = True
        if config.pad_token_id is None:
            config.pad_token_id = 0
        return result
""")

MODEL_TEMPLATE = textwrap.dedent("""\
from typing import Optional
import torch, torch.nn as nn
from hivemind import DHT
from hivemind.utils.logging import get_logger
from transformers.modeling_outputs import {output_class}

{base_model_import}

from petals.client.from_pretrained import FromPretrainedMixin
from petals.client.lm_head import LMHead
from petals.client.ptune import PTuneMixin
from petals.client.remote_generation import RemoteGenerationMixin, RemotePastKeyValues
from petals.client.remote_sequential import RemoteSequential
from petals.models.{pkg}.config import {config_class}
from petals.utils.auto_config import DefaultRevisionMixin
logger = get_logger(__name__)

if _BaseModel is not None:
    class {dist_model}(DefaultRevisionMixin, FromPretrainedMixin, PTuneMixin, _BaseModel):
        _keys_to_ignore_on_load_missing = PTuneMixin._keys_to_ignore_on_load_missing
        _keys_to_ignore_on_load_unexpected = [r"^model\\\\.layers\\\\."]
        config_class = {config_class}

        def __init__(self, config, *, dht=None):
            n, config.num_hidden_layers = config.num_hidden_layers, 0
            super().__init__(config)
            assert len(self.layers) == 0
            config.num_hidden_layers = n
            self.layers = RemoteSequential(config, dht=dht)
            self.requires_grad_(False)
            self.init_prompts(config)

        def forward(self, input_ids=None, past_key_values=None, attention_mask=None,
                    position_ids=None, head_mask=None, inputs_embeds=None, use_cache=None,
                    output_attentions=None, output_hidden_states=None,
                    return_dict=None, cache_position=None, **kwargs):
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("Cannot specify both input_ids and inputs_embeds")
            elif input_ids is not None:
                input_shape = input_ids.size(); input_ids = input_ids.view(-1, input_shape[-1])
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
            else:
                raise ValueError("Must specify input_ids or inputs_embeds")
            assert attention_mask is None or (attention_mask == 1).all()
            assert use_cache is None or use_cache
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            use_prompts = self.config.tuning_mode and "ptune" in self.config.tuning_mode and self.h.position == 0
            if use_prompts:
                prompts, ip = self.get_prompt(inputs_embeds.shape[0])
                inputs_embeds = torch.cat([prompts, inputs_embeds], dim=1)
            else:
                prompts = ip = None
            hs = inputs_embeds; out_shape = input_shape + (hs.size(-1),)
            if past_key_values is None:
                past_key_values = RemotePastKeyValues()
            hs = self.layers(hs, prompts=ip, hypo_ids=past_key_values.hypo_ids if past_key_values else None)
            if use_prompts:
                hs = hs[:, self.pre_seq_len:]
            hs = self.norm(hs).view(out_shape)
            return {output_class}(last_hidden_state=hs, past_key_values=past_key_values, hidden_states=None, attentions=None)

        @property
        def word_embeddings(self): return self.embed_tokens
        @property
        def word_embeddings_layernorm(self): return nn.Identity()
        @property
        def h(self): return self.layers
        @property
        def ln_f(self): return self.norm

    class {dist_causal_lm}(FromPretrainedMixin, RemoteGenerationMixin, _BaseCausalLM):
        _keys_to_ignore_on_load_missing = {dist_model}._keys_to_ignore_on_load_missing
        _keys_to_ignore_on_load_unexpected = {dist_model}._keys_to_ignore_on_load_unexpected
        config_class = {config_class}

        def __init__(self, config):
            _BasePreTrained.__init__(self, config)
            self.model = {dist_model}(config)
            self.lm_head = LMHead(config)
            self.post_init()

        def get_output_embeddings(self): return self.lm_head

        @property
        def transformer(self): return self.model
else:
    class {dist_model}:
        def __init__(self, *a, **kw): raise ImportError("{model_name} requires a newer transformers")
    class {dist_causal_lm}:
        def __init__(self, *a, **kw): raise ImportError("{model_name} requires a newer transformers")
""")

INIT_TEMPLATE = textwrap.dedent("""\
from petals.models.{pkg}.block import {block_class}
from petals.models.{pkg}.config import {config_class}
from petals.models.{pkg}.model import {dist_causal_lm}, {dist_model}
from petals.utils.auto_config import register_model_classes
register_model_classes(
    config={config_class},
    model={dist_model},
    model_for_causal_lm={dist_causal_lm},
)
""")

MODELS = [
    {
        "pkg": "qwen3",
        "model_name": "Qwen3",
        "block_class": "WrappedQwen3Block",
        "config_class": "DistributedQwen3Config",
        "dist_model": "DistributedQwen3Model",
        "dist_causal_lm": "DistributedQwen3ForCausalLM",
        "output_class": "BaseModelOutputWithPast",
        "decoder_import": textwrap.dedent("""\
            try:
                from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer as _DecoderLayer
            except ImportError:
                _DecoderLayer = None"""),
        "base_config_import": textwrap.dedent("""\
            try:
                from transformers.models.qwen3 import Qwen3Config as _BaseConfig
            except ImportError:
                _BaseConfig = None
            if _BaseConfig is None:
                raise ImportError("Qwen3 requires transformers >= 5.0.0")"""),
        "attn_import": textwrap.dedent("""\
            try:
                from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention as _AttnClass
            except ImportError:
                _AttnClass = None"""),
        "base_model_import": textwrap.dedent("""\
            try:
                from transformers.models.qwen3 import Qwen3Model as _BaseModel, Qwen3ForCausalLM as _BaseCausalLM, Qwen3PreTrainedModel as _BasePreTrained
            except ImportError:
                _BaseModel = _BaseCausalLM = _BasePreTrained = None"""),
    },
    {
        "pkg": "qwen3_5_moe",
        "model_name": "Qwen3.5 MoE",
        "block_class": "WrappedQwen3_5MoeBlock",
        "config_class": "DistributedQwen3_5MoeConfig",
        "dist_model": "DistributedQwen3_5MoeModel",
        "dist_causal_lm": "DistributedQwen3_5MoeForCausalLM",
        "output_class": "MoeModelOutputWithPast",
        "decoder_import": textwrap.dedent("""\
            try:
                from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextDecoderLayer as _DecoderLayer
            except ImportError:
                try:
                    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer as _DecoderLayer
                except ImportError:
                    _DecoderLayer = None"""),
        "base_config_import": textwrap.dedent("""\
            try:
                from transformers.models.qwen3_5_moe import Qwen3_5MoeTextConfig as _BaseConfig
            except ImportError:
                from transformers.models.qwen3_5_moe import Qwen3_5MoeConfig as _BaseConfig"""),
        "attn_import": textwrap.dedent("""\
            try:
                from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextAttention as _AttnClass
            except ImportError:
                _AttnClass = None"""),
        "base_model_import": textwrap.dedent("""\
            try:
                from transformers.models.qwen3_5_moe import Qwen3_5MoeTextModel as _BaseModel, Qwen3_5MoeForCausalLM as _BaseCausalLM, Qwen3_5MoePreTrainedModel as _BasePreTrained
            except ImportError:
                _BaseModel = _BaseCausalLM = _BasePreTrained = None"""),
    },
    {
        "pkg": "glm_moe_dsa",
        "model_name": "GLM-5",
        "block_class": "WrappedGlmMoeDsaBlock",
        "config_class": "DistributedGlmMoeDsaConfig",
        "dist_model": "DistributedGlmMoeDsaModel",
        "dist_causal_lm": "DistributedGlmMoeDsaForCausalLM",
        "output_class": "BaseModelOutputWithPast",
        "decoder_import": textwrap.dedent("""\
            try:
                from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaDecoderLayer as _DecoderLayer
            except ImportError:
                _DecoderLayer = None"""),
        "base_config_import": textwrap.dedent("""\
            try:
                from transformers.models.glm_moe_dsa import GlmMoeDsaConfig as _BaseConfig
            except ImportError:
                _BaseConfig = None
            if _BaseConfig is None:
                raise ImportError("GLM-5 requires transformers >= 5.2.0")"""),
        "attn_import": textwrap.dedent("""\
            try:
                from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaAttention as _AttnClass
            except ImportError:
                _AttnClass = None"""),
        "base_model_import": textwrap.dedent("""\
            try:
                from transformers.models.glm_moe_dsa import GlmMoeDsaModel as _BaseModel, GlmMoeDsaForCausalLM as _BaseCausalLM, GlmMoeDsaPreTrainedModel as _BasePreTrained
            except ImportError:
                _BaseModel = _BaseCausalLM = _BasePreTrained = None"""),
    },
]


def patch_model(models_dir, spec):
    pkg = spec["pkg"]
    pkg_dir = models_dir / pkg
    pkg_dir.mkdir(exist_ok=True)
    (pkg_dir / "__init__.py").write_text(INIT_TEMPLATE.format(**spec))
    (pkg_dir / "block.py").write_text(BLOCK_TEMPLATE.format(**spec))
    (pkg_dir / "config.py").write_text(CONFIG_TEMPLATE.format(**spec))
    (pkg_dir / "model.py").write_text(MODEL_TEMPLATE.format(**spec))
    print(f"  {spec['model_name']} ({pkg}) overlay written")


def patch_auto_config(utils_dir):
    auto_config_path = utils_dir / "auto_config.py"
    if not auto_config_path.exists():
        print("  auto_config.py not found; skipping")
        return
    text = auto_config_path.read_text()
    ALIASES_DEF = '_MODEL_TYPE_ALIASES = {"kimi_k25": "kimi_k2"}'
    if ALIASES_DEF not in text:
        text = text.replace("_CLASS_MAPPING = {}", f'{ALIASES_DEF}\n\n_CLASS_MAPPING = {{}}', 1)
    if "trust_remote_code=True" not in text:
        text = text.replace(
            "config = AutoConfig.from_pretrained(model_name_or_path, *args, **kwargs)",
            "config = AutoConfig.from_pretrained(model_name_or_path, *args, trust_remote_code=True, **kwargs)",
        )
    if "_MODEL_TYPE_ALIASES.get" not in text:
        text = text.replace(
            'if config.model_type not in _CLASS_MAPPING:',
            'model_type = _MODEL_TYPE_ALIASES.get(config.model_type, config.model_type)\n        if model_type not in _CLASS_MAPPING:',
        )
        text = text.replace(
            "proper_cls = getattr(_CLASS_MAPPING[config.model_type], cls._mapping_field)",
            "proper_cls = getattr(_CLASS_MAPPING[model_type], cls._mapping_field)",
        )
    if "is already registered" in text and "logger.warning" not in text:
        text = text.replace(
            '    assert (\n        config.model_type not in _CLASS_MAPPING\n    ), f"Model type {config.model_type} is already registered"\n\n    _CLASS_MAPPING[config.model_type]',
            '    if config.model_type in _CLASS_MAPPING:\n        return  # already registered\n    _CLASS_MAPPING[config.model_type]',
        )
    auto_config_path.write_text(text)
    print("  auto_config.py patched (trust_remote_code, aliases, lenient registration)")


def patch_models_init(models_dir, model_pkgs):
    init_path = models_dir / "__init__.py"
    text = init_path.read_text()
    changed = False
    for pkg in model_pkgs:
        marker = f"from petals.models.{pkg} import *"
        if marker not in text:
            text += f"\ntry:\n    from petals.models.{pkg} import *\nexcept ImportError:\n    pass\n"
            changed = True
    if changed:
        init_path.write_text(text)
        print("  models/__init__.py updated")
    else:
        print("  models/__init__.py already up to date")


def patch_convert_block(utils_dir):
    cb_path = utils_dir / "convert_block.py"
    if not cb_path.exists():
        print("  convert_block.py not found; skipping")
        return
    text = cb_path.read_text()
    old = "total_heads += submodule.num_heads"
    new = (
        "total_heads += getattr(submodule, 'num_heads', None) "
        "or getattr(submodule, 'num_attention_heads', model_config.num_attention_heads)"
    )
    if old in text:
        cb_path.write_text(text.replace(old, new, 1))
        print("  convert_block.py patched (num_heads fallback)")
    else:
        print("  convert_block.py already patched or has different format")


def patch_hivemind_grad_scaler(site_pkgs):
    gs_path = site_pkgs / "hivemind" / "optim" / "grad_scaler.py"
    if not gs_path.exists():
        print("  hivemind grad_scaler.py not found; skipping")
        return
    text = gs_path.read_text()
    old_import = "from torch.cuda.amp.grad_scaler import OptState, _refresh_per_optimizer_state"
    new_import = (
        "try:\n"
        "    from torch.cuda.amp.grad_scaler import OptState, _refresh_per_optimizer_state\n"
        "except ImportError:\n"
        "    import enum\n"
        "    class OptState(enum.Enum):\n"
        "        READY = 0; UNSCALED = 1; STEPPED = 2\n"
        "    def _refresh_per_optimizer_state():\n"
        "        return {'stage': OptState.READY, 'found_inf_per_device': {}}"
    )
    if old_import in text:
        gs_path.write_text(text.replace(old_import, new_import, 1))
        print("  hivemind grad_scaler.py patched (torch>=2.5 compat)")
    elif "except ImportError" in text and "OptState" in text:
        print("  hivemind grad_scaler.py already patched")
    else:
        print("  hivemind grad_scaler.py: unexpected format, skipping")


print("Patching petals with frontier model support...")
patch_auto_config(utils_dir)
patch_convert_block(utils_dir)
patch_hivemind_grad_scaler(site_pkgs)
for spec in MODELS:
    patch_model(models_dir, spec)
patch_models_init(models_dir, [m["pkg"] for m in MODELS])
print("Done.")
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


