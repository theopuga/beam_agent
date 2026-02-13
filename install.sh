#!/usr/bin/env bash
set -euo pipefail

echo "Beam Node Agent Quickstart"
echo "This will download the agent, write a config, and start pairing."
echo "Keep this terminal open to see the pairing code."
echo

read -r -p "Continue? [Y/n]: " confirm < /dev/tty
confirm=${confirm:-Y}
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
  echo "Aborted."
  exit 1
fi

control_plane_url="${BEAM_CONTROL_PLANE_URL:-https://www.openbeam.me/}"
control_plane_url="${control_plane_url%/}"
echo "Control plane URL: $control_plane_url"

release_base_default="https://github.com/theopuga/beam_agent/releases/latest/download"
read -r -p "Release base URL [$release_base_default]: " release_base < /dev/tty
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

archive_name="${asset_name}.tar.gz"
archive_path="./$archive_name"
download_url="$release_base/$archive_name"

if [[ -f "$archive_path" ]]; then
  read -r -p "$archive_path exists. Redownload? [y/N]: " redownload < /dev/tty
  redownload=${redownload:-N}
  if [[ "$redownload" =~ ^[Yy]$ ]]; then
    curl -fL "$download_url" -o "$archive_path"
    echo "Downloaded $archive_path"
  else
    echo "Using existing $archive_path"
  fi
else
  curl -fL "$download_url" -o "$archive_path"
  echo "Downloaded $archive_path"
fi

extract_dir="./${asset_name}_extracted"
rm -rf "$extract_dir"
mkdir -p "$extract_dir"
tar -xzf "$archive_path" -C "$extract_dir"

if [[ -f "$extract_dir/$asset_name" ]]; then
  binary_path="$extract_dir/$asset_name"
elif [[ -f "$extract_dir/$asset_name/$asset_name" ]]; then
  binary_path="$extract_dir/$asset_name/$asset_name"
else
  found_binary="$(find "$extract_dir" -type f -name "$asset_name" | head -n 1)"
  if [[ -n "$found_binary" ]]; then
    binary_path="$found_binary"
  else
    echo "Failed to locate $asset_name inside extracted archive."
    exit 1
  fi
fi

chmod +x "$binary_path"

read -r -p "Config path [config.yaml] (press Enter for default): " config_path < /dev/tty
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
  read -r -p "$config_path exists. Overwrite? [y/N]: " overwrite < /dev/tty
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
  "$petals_venv/bin/python" -m pip install --upgrade --no-deps grpcio protobuf grpcio-tools
  torch_spec="${BEAM_PETALS_TORCH_SPEC:-torch==2.2.2}"
  torchvision_spec="${BEAM_PETALS_TORCHVISION_SPEC:-torchvision==0.17.2}"
  torchaudio_spec="${BEAM_PETALS_TORCHAUDIO_SPEC:-torchaudio==2.2.2}"
  if [[ "${BEAM_PETALS_SKIP_TORCH_INSTALL:-}" != "true" ]]; then
    if [[ -n "${BEAM_PETALS_TORCH_INDEX_URL:-}" ]]; then
      "$petals_venv/bin/python" -m pip install --upgrade "$torch_spec" "$torchvision_spec" "$torchaudio_spec" --index-url "$BEAM_PETALS_TORCH_INDEX_URL"
    else
      "$petals_venv/bin/python" -m pip install --upgrade "$torch_spec" "$torchvision_spec" "$torchaudio_spec"
    fi
  fi
  hf_hub_spec="${BEAM_PETALS_HF_HUB_SPEC:-huggingface-hub==0.17.3}"
  transformers_spec="${BEAM_PETALS_TRANSFORMERS_SPEC:-transformers==4.34.1}"
  numpy_spec="${BEAM_PETALS_NUMPY_SPEC:-numpy<2}"
  accelerate_specs="${BEAM_PETALS_ACCELERATE_SPECS:-}"
  if [[ -z "$accelerate_specs" && -n "${BEAM_PETALS_ACCELERATE_SPEC:-}" ]]; then
    accelerate_specs="${BEAM_PETALS_ACCELERATE_SPEC}"
  fi
  if [[ -z "$accelerate_specs" ]]; then
    accelerate_specs="0.31.0 0.30.1 0.29.3 0.28.0 0.27.2 0.26.1 0.25.0 0.24.1 0.23.0 0.22.0"
  fi
  
  petals_pip_args=()
  if [[ "${BEAM_PETALS_PIP_NO_BUILD_ISOLATION:-true}" == "true" ]]; then
    petals_pip_args+=(--no-build-isolation)
  fi

  "$petals_venv/bin/python" -m pip install --upgrade --force-reinstall \
    "$hf_hub_spec" \
    "$transformers_spec" \
    "$numpy_spec" \
    "${petals_pip_args[@]}" \
    petals
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
  hivemind_spec="${BEAM_PETALS_HIVEMIND_SPEC:-hivemind==1.1.10.post2}"
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
    torch_compat_spec="${BEAM_PETALS_TORCH_COMPAT_SPEC:-torch==2.2.2}"
    torchvision_compat_spec="${BEAM_PETALS_TORCHVISION_COMPAT_SPEC:-torchvision==0.17.2}"
    torchaudio_compat_spec="${BEAM_PETALS_TORCHAUDIO_COMPAT_SPEC:-torchaudio==2.2.2}"
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
  export BEAM_PETALS_PYTHON="$petals_venv/bin/python"
}

echo

read -r -p "Start the agent now? [Y/n]: " start_now < /dev/tty
start_now=${start_now:-Y}
if [[ ! "$start_now" =~ ^[Yy]$ ]]; then
  echo "You can start it later with:"
  echo "$binary_path --config $config_path"
  exit 0
fi

ensure_petals_runtime

echo "Running: $binary_path --config $config_path"
export BEAM_CONTROL_PLANE_URL="$control_plane_url"
export CONTROL_PLANE_URL="$control_plane_url"
exec "$binary_path" --config "$config_path"
