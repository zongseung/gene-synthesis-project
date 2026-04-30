"""Experiment logger wrapping wandb with DDP-aware logging.

Only logs on rank 0 in DDP. Supports WANDB_MODE=offline for no-internet
runs. Degrades gracefully if wandb is not installed.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.utils.ddp import is_main_process

logger = logging.getLogger(__name__)

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    logger.info("wandb not installed; ExperimentLogger will log to console only.")


class ExperimentLogger:
    """Wrapper around wandb for experiment tracking.

    Handles DDP rank filtering (only rank 0 logs) and graceful
    degradation when wandb is unavailable.

    Args:
        project: wandb project name.
        run_name: Name for this run.
        tags: Optional list of tags.
        enabled: If False, disables all logging. Default True.
    """

    def __init__(
        self,
        project: str = "HiPoDiT",
        run_name: str | None = None,
        tags: list[str] | None = None,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled and is_main_process() and _WANDB_AVAILABLE
        self._run = None

        if self.enabled:
            try:
                self._run = wandb.init(
                    project=project,
                    name=run_name,
                    tags=tags,
                    reinit=True,
                )
            except Exception as e:
                logger.warning(f"wandb.init() failed: {e}. Falling back to console.")
                self.enabled = False

    def log_config(self, config: dict[str, Any]) -> None:
        """Log the full config dict.

        Args:
            config: Nested config dictionary.
        """
        if self.enabled and self._run is not None:
            wandb.config.update(config, allow_val_change=True)
        elif is_main_process():
            logger.info(f"Config: {config}")

    def log_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        """Log metrics at a given step.

        Args:
            step: Global step number.
            metrics: Dictionary of metric name -> value.
        """
        if self.enabled and self._run is not None:
            wandb.log(metrics, step=step)
        elif is_main_process():
            metrics_str = ", ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
            logger.info(f"[step {step}] {metrics_str}")

    def finish(self) -> None:
        """Finish the wandb run."""
        if self.enabled and self._run is not None:
            wandb.finish()
            self._run = None
