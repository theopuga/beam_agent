#!/bin/bash
set -e

# -----------------------------------------------------------------------
# Beam Node Agent - RunPod Setup Script for Qwen 3.5 35B-A3B (Ollama single-node mode)
# Usage: bash setup_runpod.sh
# -----------------------------------------------------------------------

CONTROL_PLANE_URL="${BEAM_CONTROL_PLANE_URL:-https://www.openbeam.me}"
OLLAMA_MODEL="${BEAM_OLLAMA_MODEL:-qwen3.5:35b-a3b}"

# Auto-detect GPU specs via nvidia-smi if not set by env vars
if [ -n "$BEAM_GPU_NAME" ]; then
    GPU_NAME="$BEAM_GPU_NAME"
elif command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
else
    GPU_NAME="Unknown GPU"
fi

if [ -n "$BEAM_GPU_VRAM_GB" ]; then
    GPU_VRAM="$BEAM_GPU_VRAM_GB"
elif command -v nvidia-smi &> /dev/null; then
    GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | awk '{sum += $1} END {printf "%d", sum/1024}')
else
    GPU_VRAM="0"
fi

if [ -n "$BEAM_GPU_COUNT" ]; then
    GPU_COUNT="$BEAM_GPU_COUNT"
elif command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    GPU_COUNT="1"
fi
BRANCH="main"
REPO="https://github.com/beam-open-node/beam_agent"

echo "=== Beam Node Agent Setup ==="
echo "Control plane: $CONTROL_PLANE_URL"
echo "Model: $OLLAMA_MODEL"
echo "GPU: $GPU_NAME (${GPU_VRAM}GB x${GPU_COUNT})"
echo ""

# 1. Install system dependencies (including Tor for onion routing support)
echo ">>> Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq curl git zstd pciutils tor

# 2. Install Ollama
# Always run the Ollama installer — it's idempotent and ensures GPU support
# is properly configured (especially after pciutils is installed).
echo ">>> Installing/updating Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# 3. Start Ollama daemon (with persistent model storage if /workspace is available)
echo ">>> Starting Ollama daemon..."
if [ -d "/workspace" ]; then
    export OLLAMA_MODELS=/workspace/ollama_models
    mkdir -p /workspace/ollama_models
    echo "    Using persistent model storage: /workspace/ollama_models"
fi
if pgrep -x ollama > /dev/null; then
    echo "    Already running."
else
    ollama serve &> /tmp/ollama.log &
    sleep 3
fi

# 4. Pull model
echo ">>> Pulling model $OLLAMA_MODEL (this may take a while)..."
ollama pull "$OLLAMA_MODEL"

# 5. Clone beam_agent
if [ -d "beam_agent" ]; then
    echo ">>> Updating existing beam_agent repo..."
    cd beam_agent
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
    cd ..
else
    echo ">>> Cloning beam_agent ($BRANCH)..."
    git clone -b "$BRANCH" "$REPO"
fi

cd beam_agent

# 6. Python venv
echo ">>> Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install -q aiohttp aiohttp-socks pyyaml "pydantic>=2.0" "cryptography>=42.0.0" faster-whisper

# 7. Write config.yaml
echo ">>> Writing config.yaml..."
cat > config.yaml << EOF
control_plane:
  url: $CONTROL_PLANE_URL

ollama:
  base_url: http://localhost:11434
  model_tag: $OLLAMA_MODEL

agent:
  heartbeat_interval_sec: 15
  pairing_ports:
    - 51337
    - 51338
    - 51339
    - 51340
  tor:
    enabled: true
  capabilities:
    supports_heavy_middle_layers: false
    max_concurrent_jobs: 1
    max_model_class: "B"
    preferred_model_id: "Qwen/Qwen3.5-35B-A3B-Ollama"
EOF

# 8. Write start script
cat > start_agent.sh << EOF
#!/bin/bash
cd "$(pwd)"
source venv/bin/activate

# Use persistent model storage on RunPod volume
if [ -d "/workspace" ]; then
    export OLLAMA_MODELS=/workspace/ollama_models
    mkdir -p /workspace/ollama_models
fi

# Ensure Ollama is running
if ! pgrep -x ollama > /dev/null; then
    echo "Starting Ollama..."
    ollama serve &> /tmp/ollama.log &
    sleep 3
fi

BEAM_GPU_NAME="$GPU_NAME" BEAM_GPU_VRAM_GB=$GPU_VRAM BEAM_GPU_COUNT=$GPU_COUNT python -m beam_node_agent.main --config config.yaml
EOF
chmod +x start_agent.sh

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To start the agent:"
echo "  cd beam_agent && bash start_agent.sh"
echo ""
echo "The agent will print a 6-digit pairing code in the logs."
echo "Enter it in the Rent Panel at $CONTROL_PLANE_URL to link your node."
echo ""

# 9. Auto-start the agent
echo ">>> Starting agent..."
bash start_agent.sh
