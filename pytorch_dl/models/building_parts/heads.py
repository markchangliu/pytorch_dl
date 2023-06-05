# Author: Chang Liu
# Some codes are refered from https://github.com/facebookresearch/pycls


"""Head module."""


import torch
import torch.nn as nn
from typing import Optional
from pytorch_dl.models.builder import head_registry


############### Classification head ###############

@head_registry.register_module("ResLinearHead")
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
    

@head_registry.register_module("ResConvHead")
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
    

@head_registry.register_module("ResLightHead")
class ResLightHead(nn.Module):
    """A light-weight classification head for all ResNet series.
    Composition: AdaptiveAvgPool2d.

    This head is supposed to receive a (H, W, num_classes) shaped 
    tensor from ResStage. The adaptive pool operation of this head will
    then directly output the score tensor of shape (1, 1, num_classes).
    """
    def __init__(
            self, 
            c_in: Optional[int], 
            num_classes: Optional[int]
        ) -> None:
        """Args:
            c_in (Optional[int]):
                The number of input channel, not used here.
                It's here for the purpose of api consistency.
            num_classes (Optional[int]):
                The number of classes, not used here.
                It's here for the purpose of api consistency.
        
        Returns:
            None.
        """
        super(ResLightHead, self).__init__()
        self.l1 = nn.AdaptiveAvgPool2d((1, 1))
    
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
