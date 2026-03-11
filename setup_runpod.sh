#!/bin/bash
set -e

# -----------------------------------------------------------------------
# Beam Node Agent - RunPod Setup Script (Ollama single-node mode)
# Usage: bash setup_runpod.sh
# -----------------------------------------------------------------------

CONTROL_PLANE_URL="${BEAM_CONTROL_PLANE_URL:-https://www.openbeam.me}"
OLLAMA_MODEL="${BEAM_OLLAMA_MODEL:-qwen3.5:35b-a3b}"
GPU_NAME="${BEAM_GPU_NAME:-NVIDIA GeForce RTX 4090}"
GPU_VRAM="${BEAM_GPU_VRAM_GB:-24}"
GPU_COUNT="${BEAM_GPU_COUNT:-1}"
BRANCH="feat/ollama-single-node"
REPO="https://github.com/beam-open-node/beam_agent"

echo "=== Beam Node Agent Setup ==="
echo "Control plane: $CONTROL_PLANE_URL"
echo "Model: $OLLAMA_MODEL"
echo "GPU: $GPU_NAME (${GPU_VRAM}GB x${GPU_COUNT})"
echo ""

# 1. Install Ollama
if ! command -v ollama &> /dev/null; then
    echo ">>> Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo ">>> Ollama already installed."
fi

# 2. Start Ollama daemon
echo ">>> Starting Ollama daemon..."
if pgrep -x ollama > /dev/null; then
    echo "    Already running."
else
    ollama serve &> /tmp/ollama.log &
    sleep 3
fi

# 3. Pull model
echo ">>> Pulling model $OLLAMA_MODEL (this may take a while)..."
ollama pull "$OLLAMA_MODEL"

# 4. Clone beam_agent
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

# 5. Python venv
echo ">>> Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install -q aiohttp pyyaml "pydantic>=2.0"

# 6. Write config.yaml
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
  capabilities:
    supports_heavy_middle_layers: false
    max_concurrent_jobs: 1
EOF

# 7. Write start script
cat > start_agent.sh << EOF
#!/bin/bash
cd "$(pwd)"
source venv/bin/activate

# Ensure Ollama is running
if ! pgrep -x ollama > /dev/null; then
    echo "Starting Ollama..."
    ollama serve &> /tmp/ollama.log &
    sleep 3
fi

BEAM_GPU_NAME="$GPU_NAME" \\
BEAM_GPU_VRAM_GB=$GPU_VRAM \\
BEAM_GPU_COUNT=$GPU_COUNT \\
python -m beam_node_agent.main --config config.yaml
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

# 8. Optionally start immediately
read -p "Start the agent now? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    bash start_agent.sh
fi
