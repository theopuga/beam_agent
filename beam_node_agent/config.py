import os
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, ConfigDict


class ControlPlaneConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    url: str = Field(default="http://localhost:8080", alias="BEAM_CONTROL_PLANE_URL")


class PetalsConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    port: int = Field(default=31337, alias="PETALS_PORT")
    public_ip: Optional[str] = Field(default=None, alias="PETALS_PUBLIC_IP")
    gpu_vram_limit: float = Field(default=0.9, alias="GPU_VRAM_LIMIT")


class AgentConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    heartbeat_interval_sec: int = 15
    max_retries: int = 5
    state_file: str = "node_state.json"
    transports: List[str] = Field(default_factory=lambda: ["fast"])
    onion_address: Optional[str] = None
    pairing_token: Optional[str] = None
    pairing_host: str = "0.0.0.0"
    pairing_ports: List[int] = Field(default_factory=lambda: [51337, 51338, 51339, 51340])
    mock_inference: bool = False
    capabilities: Dict[str, object] = Field(
        default_factory=lambda: {
            "supports_heavy_middle_layers": False,
            "max_concurrent_jobs": 1,
            "max_blocks": None,
            "max_model_class": None,
        }
    )


class BeamConfig(BaseModel):
    control_plane: ControlPlaneConfig = Field(default_factory=ControlPlaneConfig)
    petals: PetalsConfig = Field(default_factory=PetalsConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)


def load_config(config_path: str = "config.yaml") -> BeamConfig:
    # 1. Defaults
    config_data = {}
    config_dir = os.path.dirname(os.path.abspath(config_path))

    # 2. File Override
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f) or {}

    # 3. Env Vars (Manual override logic or rely on Pydantic's alias if constructed field by field)
    # For simplicity, we assume the YAML structure matches, and env vars are picked up
    # if we instantiate sub-models with os.environ defaults.
    # However, Pydantic V2 doesn't auto-read env vars without pydantic-settings.
    # We will do explicit env var checking for critical keys if not in config_data.

    cp_data = config_data.get("control_plane", {})
    if "BEAM_CONTROL_PLANE_URL" in os.environ:
        cp_data["url"] = os.environ["BEAM_CONTROL_PLANE_URL"]

    petals_data = config_data.get("petals", {})
    if "PETALS_PORT" in os.environ:
        petals_data["port"] = int(os.environ["PETALS_PORT"])

    agent_data = config_data.get("agent", {})
    if "BEAM_PAIRING_TOKEN" in os.environ:
        agent_data["pairing_token"] = os.environ["BEAM_PAIRING_TOKEN"]
    if "BEAM_PAIRING_PORT" in os.environ:
        agent_data["pairing_ports"] = [int(os.environ["BEAM_PAIRING_PORT"])]
    if "BEAM_PAIRING_PORTS" in os.environ:
        ports = []
        for value in os.environ["BEAM_PAIRING_PORTS"].split(","):
            value = value.strip()
            if value:
                ports.append(int(value))
        if ports:
            agent_data["pairing_ports"] = ports
    if "BEAM_MOCK_INFERENCE" in os.environ:
        agent_data["mock_inference"] = (
            os.environ["BEAM_MOCK_INFERENCE"].lower() == "true"
        )
        
    capabilities = agent_data.get("capabilities", {})
    if "BEAM_MAX_BLOCKS" in os.environ:
        capabilities["max_blocks"] = int(os.environ["BEAM_MAX_BLOCKS"])
    if capabilities:
        agent_data["capabilities"] = capabilities

    config = BeamConfig(
        control_plane=ControlPlaneConfig(**cp_data),
        petals=PetalsConfig(**petals_data),
        agent=AgentConfig(**agent_data),
    )

    state_file = config.agent.state_file
    if state_file and not os.path.isabs(state_file):
        config.agent.state_file = os.path.join(config_dir, state_file)

    return config
