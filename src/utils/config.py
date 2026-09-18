"""YAML configuration loader.

Config is accessed as a nested dict: ``config['data']['gene_size']``.
Missing keys raise KeyError at the point of use, which names the key.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML config file into a nested dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def parse_args_with_config() -> dict:
    """Parse ``--config`` / ``--single_gpu`` and load the config.

    Expected CLI usage:
        python src/training/trainer.py --config configs/default.yaml
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--single_gpu", action="store_true", help="Run on single GPU (no DDP)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.single_gpu:
        config["distributed"]["num_gpus"] = 1
    return config
