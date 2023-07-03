# Author: Chang Liu


"""Utility functions."""


import copy
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from typing import Dict, Any, Optional

from pytorch_dl.core.logging import get_logger


_logger = get_logger(__name__)


def build_module_from_cfg(
        prefix: str,
        cfg: Dict[str, Any],
        registry: Dict[str, Module],
    ) -> Module:
    cfg = copy.deepcopy(cfg)
    module_type = cfg.pop("type")
    assert module_type in registry.keys(), \
        (f"{prefix.capitalize()} '{module_type}' is not one of the "
         f"supported types '{list(registry.keys())}'")
    module = registry[module_type](**cfg)
    return module


def save_checkpoint(
        model: Module,
        checkpoint_path: str,
        optimizer: Optimizer,
        scheduler: Scheduler,
    ) -> None:
    _logger.info(f"Saving checkpoint at '{checkpoint_path}'...")
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict()
    }
    torch.save(checkpoint, checkpoint_path)
    _logger.info(f"Saving checkpoint completes.")


def load_checkpoint(
        model: Module,
        checkpoint_path: str,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[Scheduler] = None
    ) -> None:
    _logger.info(f"Loading checkpoint from '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path)
    model_state = checkpoint.get("model_state", None)
    optimizer_state = checkpoint.get("optimizer_state", None)
    scheduler_state = checkpoint.get("scheduler_state", None)
    if model_state:
        model.load_state_dict(model_state)
    if optimizer and optimizer_state:
        optimizer.load_state_dict(optimizer_state)
    if scheduler and scheduler_state:
        scheduler.load_state_dict(scheduler_state)