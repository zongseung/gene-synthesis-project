"""YAML configuration loader with CLI override support.

Loads nested YAML config and supports dot-notation CLI overrides
(e.g., training.lr=1e-3). Config is accessed as nested dict:
    config['data']['gene_size']
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml


# Keys that must exist in every valid config.
REQUIRED_KEYS: list[tuple[str, ...]] = [
    ("data", "gene_size"),
    ("data", "num_channels"),
    ("data", "zero_mask_path"),
    ("data", "label_hierarchy_path"),
    ("model", "name"),
    ("diffusion", "max_timesteps"),
    ("training", "batch_size"),
    ("training", "epochs"),
    ("training", "lr"),
    ("save_dir",),
]


def _set_nested(d: dict, keys: list[str], value: Any) -> None:
    """Set a value in a nested dict using a list of keys."""
    for key in keys[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


def _cast_value(value: str) -> int | float | bool | str:
    """Auto-cast a CLI string value to the appropriate Python type."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "none":
        return None
    # Try int
    try:
        return int(value)
    except ValueError:
        pass
    # Try float
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _get_nested(d: dict, keys: tuple[str, ...]) -> Any:
    """Retrieve a value from a nested dict using a tuple of keys."""
    for key in keys:
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


def load_config(
    path: str | Path,
    overrides: list[str] | None = None,
) -> dict:
    """Load a YAML config file and apply optional CLI overrides.

    Args:
        path: Path to the YAML config file.
        overrides: List of dot-notation overrides, e.g.
            ["training.lr=1e-3", "training.batch_size=64"].

    Returns:
        Nested dict with config values.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If a required key is missing after loading.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    # Apply CLI overrides
    if overrides:
        for override in overrides:
            if "=" not in override:
                raise ValueError(
                    f"Invalid override format '{override}'. "
                    "Expected dot.notation=value"
                )
            key_str, value_str = override.split("=", 1)
            keys = key_str.strip().split(".")
            value = _cast_value(value_str.strip())
            _set_nested(config, keys, value)

    # Validate required keys
    validate_config(config)

    return config


def validate_config(config: dict) -> None:
    """Check that all required keys exist in the config.

    Raises:
        ValueError: If any required key path is missing.
    """
    missing = []
    for key_path in REQUIRED_KEYS:
        if _get_nested(config, key_path) is None:
            missing.append(".".join(key_path))
    if missing:
        raise ValueError(
            f"Missing required config keys: {', '.join(missing)}"
        )


def parse_args_with_config() -> dict:
    """Parse command-line arguments, load config, and apply overrides.

    Expected CLI usage:
        python script.py --config configs/default.yaml training.lr=1e-3

    Returns:
        Loaded and validated config dict.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--single_gpu",
        action="store_true",
        help="Run on single GPU (no DDP)",
    )
    args, unknown = parser.parse_known_args()

    # unknown args are treated as dot-notation overrides
    overrides = [arg for arg in unknown if "=" in arg]

    config = load_config(args.config, overrides=overrides)

    if args.single_gpu:
        config["distributed"]["num_gpus"] = 1

    return config
