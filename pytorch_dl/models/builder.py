# Author: Chang Liu
# Some codes are refered from https://github.com/open-mmlab/mmcv


"""Model building module."""


import copy
import torch
import torch.nn as nn
from pytorch_dl.core.registry import Registry
from typing import Dict, Any, Callable, Union


stem_registry = Registry("stem")
stage_registry = Registry("stage")
head_registry = Registry("head")
net_registry = Registry("net")


def _init_weight(m: nn.Module) -> None:
    """Performs Xavier initialization to Conv and linear layers,
    constant initialization to batchnormalization layers.

    Args:
        m (nn.Module):
            The pytorch nn.Module you want to initialize weights on.

    Returns:
        None.
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def build_from_cfg(
        registry: Registry,
        cfg_dict: Dict[str, Callable[..., Any]]
    ) -> Union[Callable[..., Any], Any]:
    """Build models by calling the class and functions
    with args provided by config file.

    Args:
        registry (Registry):
            The registry that hold the module_dict.
        cfg_dict (Dict[str, Callable[..., Any]]):
            The configuration dict. `cfg_dict["name"]`
            specifies the name of the called class or function, 
            and the remaining (key, value) are arguments 
            for that class or function.
    
    Returns:
        A pytorch nn.Module made by calling a class, or any
        returned types output from a function.
    """
    cfg_dict = copy.deepcopy(cfg_dict)
    module_dict = registry.get_module_dict()
    cls_name = cfg_dict.pop("name")
    cls_obj = module_dict[cls_name]
    return cls_obj(**cfg_dict)


def build_stem(
        cfg: Dict[str, Callable[..., Any]]
    ) -> nn.Module:
    """Build a stem from configuration.

    Args:
        cfg (Dict[str, Callable[..., Any]]):
            The configuration dict.
    
    Returns:
        stem (nn.Module):
            A stem object.
    """
    stem = build_from_cfg(stem_registry, cfg)
    stem.apply(_init_weight)
    return stem


def build_stage(
        cfg: Dict[str, Callable[..., Any]]
    ) -> nn.Module:
    """Build a stage from configuration.

    Args:
        cfg (Dict[str, Callable[..., Any]]):
            The configuration dict.
    
    Returns:
        stage (nn.Module):
            A stage object.
    """
    stage = build_from_cfg(stage_registry, cfg)
    stage.apply(_init_weight)
    return stage


def build_head(
        cfg: Dict[str, Callable[..., Any]]
    ) -> nn.Module:
    """Build a head from configuration.

    Args:
        cfg (Dict[str, Callable[..., Any]]):
            The configuration dict.
    
    Returns:
        Head (nn.Module):
            A head object.
    """
    head = build_from_cfg(head_registry, cfg)
    head.apply(_init_weight)
    return head


def build_net(
        cfg: Dict[str, Callable[..., Any]]
    ) -> nn.Module:
    """Build a net from configuration.

    Args:
        cfg (Dict[str, Callable[..., Any]]):
            The configuration dict.
    
    Returns:
        Net (nn.Module):
            A net object.
    """
    net = build_from_cfg(net_registry, cfg)
    net.apply(_init_weight)
    return net