# Author: Chang Liu
# Some codes are refered from https://github.com/open-mmlab/mmdetection/tree/2.x


"""Models."""


import copy
import torch
import torch.nn as nn
from typing import Dict, Any, Callable, Union, Optional

# from pytorch_dl.models.classifiers import (
#     BaseClassifier,
#     ResNetClassifier,
#     ResNet18Classifier,
#     ResNet50Classifier,
# )


# __all__ = ["build_model"]


# def _init_weight(m: nn.Module) -> None:
#     """Performs Xavier initialization to Conv and linear layers,
#     constant initialization to batchnormalization layers.

#     Args:
#         m (nn.Module):
#             The pytorch nn.Module you want to initialize weights on.

#     Returns:
#         None.
#     """
#     if isinstance(m, (nn.Conv2d, nn.Linear)):
#         nn.init.xavier_normal_(m.weight)
#     elif isinstance(m, nn.BatchNorm2d):
#         nn.init.constant_(m.weight, 1)
#         nn.init.constant_(m.bias, 0)


# def _load_pretrained_weights(
#         model: nn.Module,
#         checkpoint_path: str
#     ) -> nn.Module:
#     checkpoints = torch.load(checkpoint_path)
#     model_state = checkpoints["model_state"]
#     model.load_state_dict(model_state)