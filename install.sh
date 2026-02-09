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
echo

read -r -p "Start the agent now? [Y/n]: " start_now
start_now=${start_now:-Y}
if [[ ! "$start_now" =~ ^[Yy]$ ]]; then
  echo "You can start it later with:"
  echo "$binary_path --config $config_path"
  exit 0
fi

echo "Running: $binary_path --config $config_path"
export BEAM_CONTROL_PLANE_URL="$control_plane_url"
exec "$binary_path" --config "$config_path"
