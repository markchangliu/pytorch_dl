# Author: Chang Liu
# Some codes are refered from https://github.com/facebookresearch/pycls


"""Stem module."""


import copy
import torch
import torch.nn as nn
from typing import Dict, Any


__all__ = ["ResStem"]


class ResStem(nn.Module):
    """The stem used in all ResNet series.
    Composition: 7x7, BN, Relu, MaxPool.
    """
    def __init__(self, c_out: int) -> None:
        """
        Args:
            c_out (int):
                The number of output channels.
        Returns:
            None
        """
        super(ResStem, self).__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(3, c_out, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )

        self._cfg = {
            "type": type(self).__name__,
            "c_out": c_out
        }

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward propogation.

        Args:
            X (Tensor):
                Input tensor.
        
        Returns:
            output (Tensor):
                Output feature maps.
        """
        X = self.l1(X)
        return X

    @property
    def cfg(self) -> Dict[str, Any]:
        return copy.deepcopy(self._cfg)