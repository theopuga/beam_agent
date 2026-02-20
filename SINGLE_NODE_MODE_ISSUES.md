# Single Node Mode Issues

**Context:** We attempted to run a single `beam-node-agent` on a Runpod instance (RTX 3090, 24GB VRAM) to serve the full `tiiuae/falcon-7b-instruct` model (which consists of 32 blocks) using Petals.

**The Issue:** The chat interface on `openbeam.me` would fail to generate complete responses, often hanging at a single black dot. We traced this back to the node only loading 12 out of 32 blocks (`blocks [0, 12]`), rendering it ineligible for a full single-node "fast pool" assignment.

**What We Fixed:**
1. **Control Plane Defaults:** The Open WebUI backend originally hardcoded `max_blocks` to 8 for class A models. We updated this in the `beam_scheduler.py` to `32`, `60`, and `80` for classes A, B, and C respectively (deployed to Railway).
2. **Agent Capabilities:** The `beam-node-agent` codebase defaulted its capability `max_blocks` to `12` in `config.py`. We updated this default to `40`, allowed parsing of a `BEAM_MAX_BLOCKS` environment variable, and released binary version `v0.1.11` via GitHub. 
3. **Installer Hardcoding:** We modified `install.sh` to explicitly add `export BEAM_MAX_BLOCKS=40` when booting in `BEAM_SINGLE_NODE=true` mode.
4. **Environment Variables:** We correctly exported `BEAM_HOP_COUNTS="A=1,B=1,C=1"` to force the scheduler to assign the entire model to a single hop.

**Remaining Failure / Next Steps:**
Despite the above patches to both the control plane (Open WebUI) and the data plane (beam-node-agent), the setup still failed to fully load all 32 blocks or successfully stream chat inferencing. 

The next person troubleshooting this should investigate:
1. **Petals Internal Limits:** Petals might be internally capping blocks based on its own calculation of the node's VRAM constraints (`torch.cuda.get_device_properties`).
2. **Scheduler Assignment Logic:** The `beam_scheduler.py` logic combining `BEAM_HOP_COUNTS` and `.max_blocks` could still be truncating the assignment before it reaches the Runpod node.
3. **Agent State Overwrites:** The agent's `state.json` might be keeping an older cached `max_blocks` value that supersedes the new `config.yaml` defaults.
4. **Control Plane Caching:** Consider if Railway deployment caching or node staleness in Open WebUI is returning a stale node footprint.
