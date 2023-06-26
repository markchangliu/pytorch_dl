# Author: Chang Liu
# Some codes are refered from https://github.com/facebookresearch/pycls


"""Head module."""


import copy
import torch
import torch.nn as nn
from typing import Optional, Dict, Any


__all__ = ["ResLinearHead", "ResConvHead", "ResLightHead"]


############### Classification head ###############

class ResLinearHead(nn.Module):
    """The linear head used in all ResNet series.
    Composition: AdaptiveAvgPool, Linear.
    """
    def __init__(self, c_in: int, num_classes: int) -> None:
        """Args:
            c_in (int):
                The number of input channels.
            num_classes (int):
                The number of classes.
        
        Returns:
            None.
        """
        super(ResLinearHead, self).__init__()
        self.l1_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.l1_2 = nn.Linear(c_in, num_classes)

        self._cfg = {
            "type": type(self).__name__,
            "c_in": c_in,
            "num_classes": num_classes
        }


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward propogation.

        Args:
            X (Tensor):
                Input tensor.
        
        Returns:
            logit (Tensor):
                The predicted classification logit.
        """
        X = self.l1_1(X)
        X = self.l1_2(X.contiguous().view(X.shape[0], -1))
        # X = self.l1_2(X)
        return X
    
    @property
    def cfg(self) -> Dict[str, Any]:
        return copy.deepcopy(self._cfg)
    

class ResConvHead(nn.Module):
    """Replace the fc-layer in official ResNet head with a conv1x1 layer.
    Composition: AdaptiveAvgPool, Conv1x1.
    """
    def __init__(self, c_in: int, num_classes: int) -> None:
        """Args:
            c_in (int):
                The number of input channels.
            num_classes (int):
                The number of classes.
        
        Returns:
            None.
        """
        super(ResConvHead, self).__init__()
        self.l1_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.l1_2 = nn.Conv2d(c_in, num_classes, 1)

        self._cfg = {
            "type": type(self).__name__,
            "c_in": c_in,
            "num_classes": num_classes
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propogation.

        Args:
            X (Tensor):
                Input tensor.
        
        Returns:
            logit (Tensor):
                The predicted classification logit.
        """
        X = self.l1_1(X)
        X = self.l1_2(X.contiguous().view(X.shape[0], -1))
        X = self.l1_2(X)
        return X
    
    @property
    def cfg(self) -> Dict[str, Any]:
        return copy.deepcopy(self._cfg)
    

class ResLightHead(nn.Module):
    """A light-weight classification head for all ResNet series.
    Composition: AdaptiveAvgPool2d.

    This head is supposed to receive a (H, W, num_classes) shaped 
    tensor from ResStage. The adaptive pool operation of this head will
    then directly output the score tensor of shape (1, 1, num_classes).
    """
    def __init__(self) -> None:
        """Args:
            None
        
        Returns:
            None.
        """
        super(ResLightHead, self).__init__()
        self.l1 = nn.AdaptiveAvgPool2d((1, 1))

        self._cfg = {
            "type": type(self).__name__
        }

    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward propogation.

        Args:
            X (Tensor):
                Input tensor. The shape of this tensor
                should be (H, W, num_classes).
        
        Returns:
            logit (Tensor):
                The predicted classification logit.
        """
        X = self.l1(X)
        return X

    @property
    def cfg(self) -> Dict[str, Any]:
        return copy.deepcopy(self._cfg)