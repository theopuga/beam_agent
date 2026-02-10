import argparse
import asyncio
import logging

from .config import load_config
from .service import NodeAgent


def main():
    parser = argparse.ArgumentParser(description="Beam Node Agent")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to configuration file"
    )
    args = parser.parse_args()

    # Setup Logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Load Config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return

    # Start Agent
    agent = NodeAgent(config)
    try:
        asyncio.run(agent.start())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
