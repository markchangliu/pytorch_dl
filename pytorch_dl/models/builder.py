# Author: Chang Liu
# Some codes are refered from https://github.com/open-mmlab/mmcv


"""Model building module."""


import copy
import torch
import torch.nn as nn
from typing import Dict, Any, Callable, Union, Optional

from pytorch_dl.core.logging import get_logger
from pytorch_dl.core.registry import Registry


_logger = get_logger(__name__)

# stem_registry = Registry("stem")
# stage_registry = Registry("stage")
# body_registry = Registry("body")
# head_registry = Registry("head")
classifier_registry = Registry("classifier")


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


def _build_from_cfg(
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
    cls_name = cfg_dict.pop("name")
    _logger.info(f"Get the module class name as {cls_name}.")
    cls_obj = registry.get_module(cls_name)
    pretrained_weights = cfg_dict.get("pretrained_weights")
    if pretrained_weights:
        _logger(f"Loading pretrained weights from {pretrained_weights}...")
        cfg_dict.pop("pretrained_weights")
        checkpoints = torch.load(pretrained_weights)
        model_state = checkpoints["model_state"]
        cls_instance = cls_obj(**cfg_dict)
        cls_instance.load_state_dict(model_state)
        _logger(f"Loading pretrained weights completes.")
    else:
        cls_instance = cls_obj(**cfg_dict)
        cls_instance.apply(_init_weight)
    return cls_instance


# def build_stem(
#         cfg: Dict[str, Callable[..., Any]]
#     ) -> nn.Module:
#     """Build a stem from configuration.

#     Args:
#         cfg (Dict[str, Callable[..., Any]]):
#             The configuration dict.
    
#     Returns:
#         stem (nn.Module):
#             A stem object.
#     """
#     stem = _build_from_cfg(stem_registry, cfg)
#     return stem


# def build_stage(
#         cfg: Dict[str, Callable[..., Any]]
#     ) -> nn.Module:
#     """Build a stage from configuration.

#     Args:
#         cfg (Dict[str, Callable[..., Any]]):
#             The configuration dict.
    
#     Returns:
#         stage (nn.Module):
#             A stage object.
#     """
#     stage = _build_from_cfg(stage_registry, cfg)
#     return stage


# def build_body(
#         cfg: Dict[str, Callable[..., Any]]
#     ) -> nn.Module:
#     body = _build_from_cfg(body_registry, cfg)
#     return body


# def build_head(
#         cfg: Dict[str, Callable[..., Any]]
#     ) -> nn.Module:
#     """Build a head from configuration.

#     Args:
#         cfg (Dict[str, Callable[..., Any]]):
#             The configuration dict.
    
#     Returns:
#         Head (nn.Module):
#             A head object.
#     """
#     head = _build_from_cfg(head_registry, cfg)
#     return head


def build_classifier(
        cfg: Dict[str, Callable[..., Any]],
        data_parallel_cfg: Dict[str, Any]
    ) -> Union[nn.Module, nn.DataParallel]:
    """Build a net from configuration.

    Args:
        cfg (Dict[str, Callable[..., Any]]):
            The configuration dict.
    
    Returns:
        Net (nn.Module):
            A net object.
    """
    _logger.info("Building classifier...")
    classifier = _build_from_cfg(classifier_registry, cfg)
    if data_parallel_cfg:
        device_ids = data_parallel_cfg["device_ids"]
        output_device = data_parallel_cfg["output_device"]
        classifier = nn.DataParallel(
            classifier,
            device_ids,
            output_device
        )
    _logger.info("Building classifier completes.")
    return classifier