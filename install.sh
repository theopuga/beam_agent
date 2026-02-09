#!/usr/bin/env bash
set -euo pipefail

echo "Beam Node Agent Quickstart"
echo "This will download the agent, write a config, and start pairing."
echo "Keep this terminal open to see the pairing code."
echo

read -r -p "Continue? [Y/n]: " confirm
confirm=${confirm:-Y}
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
  echo "Aborted."
  exit 1
fi

control_plane_url="${BEAM_CONTROL_PLANE_URL:-https://www.openbeam.me/}"
control_plane_url="${control_plane_url%/}"
echo "Control plane URL: $control_plane_url"

release_base_default="https://download.openbeam.me"
read -r -p "Release base URL [$release_base_default]: " release_base
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

if [[ -f "$binary_path" ]]; then
  read -r -p "$binary_path exists. Redownload? [y/N]: " redownload
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

read -r -p "Config path [config.yaml] (press Enter for default): " config_path
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
  read -r -p "$config_path exists. Overwrite? [y/N]: " overwrite
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
    # max_blocks: 12
    # max_model_class: "B"
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
    # max_blocks: 12
    # max_model_class: "B"
EOF
  echo "Wrote $config_path"
fi

echo
echo "Next: the agent will start and print a 6-digit pair code."
echo "Open the Rent Panel and enter the code to link this machine."
echo

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
  if [[ "$petals_venv" != /* ]]; then
    petals_venv="$(pwd -P)/$petals_venv"
  fi
  echo "Installing Petals runtime in $petals_venv"
  python3 -m venv "$petals_venv"
  "$petals_venv/bin/python" -m pip install --upgrade pip
  if ! "$petals_venv/bin/python" - <<'PY' >/dev/null 2>&1
import pkg_resources  # noqa: F401
PY
  then
    "$petals_venv/bin/python" -m pip install setuptools
  fi
  if ! "$petals_venv/bin/python" - <<'PY' >/dev/null 2>&1
import wheel  # noqa: F401
PY
  then
    "$petals_venv/bin/python" -m pip install wheel
  fi
  if [[ -n "${BEAM_PETALS_TORCH_INDEX_URL:-}" ]]; then
    "$petals_venv/bin/python" -m pip install torch torchvision torchaudio --index-url "$BEAM_PETALS_TORCH_INDEX_URL"
  fi
  PIP_NO_BUILD_ISOLATION=1 "$petals_venv/bin/python" -m pip install --no-build-isolation petals
  export BEAM_PETALS_PYTHON="$petals_venv/bin/python"
}

read -r -p "Start the agent now? [Y/n]: " start_now
start_now=${start_now:-Y}
if [[ ! "$start_now" =~ ^[Yy]$ ]]; then
  echo "You can start it later with:"
  echo "$binary_path --config $config_path"
  exit 0
fi

ensure_petals_runtime

echo "Running: $binary_path --config $config_path"
export BEAM_CONTROL_PLANE_URL="$control_plane_url"
exec "$binary_path" --config "$config_path"
