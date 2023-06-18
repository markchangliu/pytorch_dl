# Author: Chang Liu
# Some codes are refered from https://github.com/facebookresearch/pycls


"""Optimizers."""


import torch
from torch.nn import Module, DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Tuple, Union


__all__ = ["build_optimizer_scheduler"]


def build_optimizer_scheduler(
        model: Module,
        lr: int,
        checkpoint_path: str
    ) -> Tuple[SGD, CosineAnnealingLR]:
    optimizer = SGD(
        model.parameters(),
        lr,
        momentum=0.9,
        weight_decay=5e-4
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=50,
        eta_min=1e-6
    )
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return optimizer, scheduler
