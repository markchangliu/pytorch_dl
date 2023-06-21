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
    loss_func_param = loss_func_cfg.get(loss_func_type, None)
    if loss_func_cfg:
        loss_func = supported_loss_funcs[loss_func_type](**loss_func_param)
    else:
        loss_func = supported_loss_funcs[loss_func_type]()
    return loss_func


def _build_classification_metrics(
        metric_func_cfg: Dict[str, Any]
    ) -> Dict[str, Callable[[Tensor, Tensor], float]]:
    supported_metric_funcs = {
        "Accuracy": Accuracy,
        "Precision": Precision,
        "Recall": Recall
    }
    metric_func_cfg = copy.deepcopy(metric_func_cfg)
    metric_types = metric_func_cfg.pop("types")
    metric_funcs = {}
    for metric_type in metric_types:
        assert metric_type in supported_metric_funcs.keys(), \
        (f"Metric func type '{metric_type}' is not one "
         f"of the supported types '{list(supported_metric_funcs.keys())}'.")
        metric_func_cfg = metric_func_cfg[metric_type]
        metric_func_param = metric_func_cfg.get(metric_type, None)
        if metric_func_param:
            metric_func = supported_metric_funcs[metric_type](**metric_func_param)
        else:
            metric_func = supported_metric_funcs[metric_type]()
        metric_funcs.update({metric_type: metric_func})


def build_loss_func(
        task_type: str,
        loss_func_cfg: Dict[str, Any]
    ) -> Callable[[Any], Tensor]:
    build_funcs = {
        "classification": _build_classification_loss_func
    }
    loss_func = build_funcs[task_type](loss_func_cfg)
    return loss_func


def build_metric_funcs(
        task_type: str,
        metric_func_cfgs: Dict[str, Any]
    ) -> Callable[[Any], float]:
    build_funcs = {
        "classification": _build_classification_metrics
    }
    metric_funcs = build_funcs[task_type](metric_func_cfgs)
    return metric_funcs