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
  torch_spec="${BEAM_PETALS_TORCH_SPEC:-torch<2.2}"
  torchvision_spec="${BEAM_PETALS_TORCHVISION_SPEC:-torchvision<0.17}"
  torchaudio_spec="${BEAM_PETALS_TORCHAUDIO_SPEC:-torchaudio<2.2}"
  if [[ "${BEAM_PETALS_SKIP_TORCH_INSTALL:-}" != "true" ]]; then
    if [[ -n "${BEAM_PETALS_TORCH_INDEX_URL:-}" ]]; then
      "$petals_venv/bin/python" -m pip install --upgrade "$torch_spec" "$torchvision_spec" "$torchaudio_spec" --index-url "$BEAM_PETALS_TORCH_INDEX_URL"
    else
      "$petals_venv/bin/python" -m pip install --upgrade "$torch_spec"
    fi
  fi
  hf_hub_spec="${BEAM_PETALS_HF_HUB_SPEC:-huggingface-hub>=0.24,<1.0}"
  transformers_spec="${BEAM_PETALS_TRANSFORMERS_SPEC:-transformers==4.43.1}"
  "$petals_venv/bin/python" -m pip install --upgrade "$hf_hub_spec" "$transformers_spec"
  petals_pip_args=()
  if [[ "${BEAM_PETALS_PIP_NO_BUILD_ISOLATION:-true}" == "true" ]]; then
    petals_pip_args+=(--no-build-isolation)
  fi
  "$petals_venv/bin/python" -m pip install "${petals_pip_args[@]}" petals
  if ! "$petals_venv/bin/python" - <<'PY' >/dev/null 2>&1
from huggingface_hub import split_torch_state_dict_into_shards  # noqa: F401
PY
  then
    echo "huggingface-hub too old for Petals; upgrading to a compatible version"
    "$petals_venv/bin/python" -m pip install --upgrade --force-reinstall "$hf_hub_spec" "$transformers_spec"
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
