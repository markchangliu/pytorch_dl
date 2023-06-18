# Author: Chang Liu


"""Metrics."""


import copy
from torch import Tensor
from typing import Callable, Dict, Any

from pytorch_dl.metrics.classifications import (
    CrossEntropyLoss,
    Accuracy,
    Precision,
    Recall
)


def _build_classification_loss_func(
        loss_func_cfg: Dict[str, Any]
    ) -> Callable[[Tensor, Tensor], Tensor]:
    supported_loss_funcs = {
        "CrossEntropyLoss": CrossEntropyLoss
    }
    loss_func_cfg = copy.deepcopy(loss_func_cfg)
    loss_func_type = loss_func_cfg.pop("type")
    assert loss_func_type in supported_loss_funcs.keys(), \
        (f"Loss func type '{loss_func_type}' is not one "
         f"of the supported types '{list(supported_loss_funcs.keys())}'.")
    loss_func = supported_loss_funcs[loss_func_type](**loss_func_cfg)
    return loss_func





def build_loss_func(
        task_type: str,
        loss_func_cfg: Dict[str, Any]
    ) -> Callable[[Any], Tensor]:
    build_funcs = {
        "classification": _build_classification_loss_func
    }
    loss_func = build_funcs[task_type](loss_func_cfg)
    return loss_func