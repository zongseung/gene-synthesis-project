"""Exponential Moving Average (EMA) for model parameters.

Shadow parameters are stored in fp32 regardless of model precision.
Default decay of 0.9999 follows standard practice for diffusion models.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn


class EMAModel:
    """Exponential Moving Average of model parameters.

    Maintains shadow copies of all model parameters in fp32 and updates
    them with exponential moving average at each training step.

    Args:
        model: The model whose parameters to track.
        decay: EMA decay rate. Higher = slower update. Default 0.9999.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        # Store shadow params in fp32 for numerical stability
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().float()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow parameters with EMA of current model parameters.

        Args:
            model: The model with updated parameters (after optimizer step).
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_average = (
                    self.decay * self.shadow[name]
                    + (1.0 - self.decay) * param.data.float()
                )
                self.shadow[name] = new_average

    def apply_shadow(self, model: nn.Module) -> None:
        """Copy shadow parameters into the model for inference.

        Call ``restore()`` afterwards to revert to training parameters.

        Args:
            model: The model to apply shadow parameters to.
        """
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name].to(param.data.dtype))

    def restore(self, model: nn.Module) -> None:
        """Restore original model parameters after inference.

        Args:
            model: The model to restore parameters to.
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict[str, Any]:
        """Return the EMA state for checkpoint saving."""
        return {
            "decay": self.decay,
            "shadow": {k: v.clone() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load EMA state from a checkpoint.

        Args:
            state_dict: Dict containing 'decay' and 'shadow' keys.
        """
        self.decay = state_dict["decay"]
        self.shadow = {k: v.clone() for k, v in state_dict["shadow"].items()}
