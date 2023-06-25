# Author: Chang Liu
# Some codes are refered from https://github.com/open-mmlab/mmdetection/tree/2.x


"""Model init code.

All registered class and functions should be imported here.
"""


import copy
import torch
import torch.nn as nn
from typing import Dict, Any, Callable, Union, Optional

from pytorch_dl.models.classifiers import *


__all__ = ["build_model"]

_MODELS = {
    "classification": build_classifier
}

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


def _load_pretrained_weights(
        model: nn.Module,
        checkpoint_path: str
    ) -> nn.Module:
    checkpoints = torch.load(checkpoint_path)
    model_state = checkpoints["model_state"]
    model.load_state_dict(model_state)


def build_model(
        task_type: str,
        model_cfg: Dict[str, Any], 
        checkpoint_path: Optional[str] = None
    ) -> nn.Module:
    assert task_type in _MODELS.keys(), \
        (f"`task_type`={task_type} is not one of the supported "
         f"model types {list(_MODELS.keys())}.")

    model = _MODELS[task_type](model_cfg)

    if checkpoint_path:
        _load_pretrained_weights(model, checkpoint_path)
    else:
        model.apply(_init_weight)
    
    return model