from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.config.loader import load_config
from app.logging_config import setup_logging
from app.state import App, AppState


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rhythm Game Keyboard Automation")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML (default: app/config/config.yaml)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--log-file", type=str, default=None,
        help="Optional log file path",
    )
    args = parser.parse_args(argv)

    config = load_config(config_path=args.config)
    logger = setup_logging(level=args.log_level, log_file=args.log_file)

    app = App()
    logger.info("Application started: %s v%s", config.name, config.version)
    logger.info("State: %s", app.state.value)

    return 0


if __name__ == "__main__":
    sys.exit(main())
