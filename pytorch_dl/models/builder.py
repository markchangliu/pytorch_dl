# Author: Chang Liu
# Some codes are refered from https://github.com/open-mmlab/mmcv


"""Model building module."""


import copy
import torch
import torch.nn as nn
from pytorch_dl.core.registry import Registry
from typing import Dict, Any, Callable


stem_registry = Registry("stem")
stage_registry = Registry("stage")
head_registry = Registry("head")
net_registry = Registry("net")


def _init_weight(m: nn.Module) -> None:
    """Performs Xavier initialization to Conv and linear layers,
    constant initialization to batchnormalization layers.
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def build_from_cfg(
        registry: Registry,
        cfg_dict: Dict[str, Callable[..., Any]]
    ) -> Callable[..., Any]:
    cfg_dict = copy.deepcopy(cfg_dict)
    module_dict = registry.get_module_dict()
    cls_name = cfg_dict.pop("name")
    cls_obj = module_dict[cls_name]
    return cls_obj(**cfg_dict)


def build_stem(
        cfg: Dict[str, Callable[..., Any]]
    ) -> nn.Module:
    stem = build_from_cfg(stem_registry, cfg)
    stem.apply(_init_weight)
    return stem


def build_stage(
        cfg: Dict[str, Callable[..., Any]]
    ) -> nn.Module:
    stage = build_from_cfg(stage_registry, cfg)
    stage.apply(_init_weight)
    return stage


def build_head(
        cfg: Dict[str, Callable[..., Any]]
    ) -> nn.Module:
    head = build_from_cfg(head_registry, cfg)
    head.apply(_init_weight)
    return head


def build_net(
        cfg: Dict[str, Callable[..., Any]]
    ) -> nn.Module:
    net = build_from_cfg(net_registry, cfg)
    net.apply(_init_weight)
    return net