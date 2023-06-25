# Author: Chang Liu
# Some codes are refered from https://github.com/facebookresearch/pycls


"""Block module."""


import copy
import torch
import torch.nn as nn
from typing import Optional, Union, Dict, Any


__all__ = ["ResResidualBlock"]


def _get_trans_block(
        trans_block_type: str
    ) -> Union["ResBasicBlock", "ResBottleneckBlock"]:
    """Return ResNet block class based on the string name provided by
    `trans_blcok_name`.

    Args:
        trans_block_type (str):
            The name of transformation block, should be one of 
            `ResBasicBlock` or `ResBottleneckBlock`.
    Returns:
        ResBlockClass (Union["ResBasicBlock", "ResBottleneckBlock"]):
            The transformation block class that will be used in 
            ResNet construction.
    """
    _d = {
        "ResBasicBlock": ResBasicBlock,
        "ResBottleneckBlock": ResBottleneckBlock
    }
    return _d[trans_block_type]


class ResBasicBlock(nn.Module):
    """The basic transformation block in ResNet-18/34.
    Composition: 3x3, BN, AF, 3x3, BN.
    """
    
    def __init__(
            self, 
            stride: int,
            c_in: int,
            c_out: int,
            c_b: Optional[int] = None
        ) -> None:
        """
        Args:
            stride (int):
                Number of strides, stride=1 will maintain the feature
                size unchanged, stride=2 will halve the feature height 
                and width and double the channel number.
            c_in (int):
                Number of input channel.
            c_out (int):
                Number of output channel.
            c_b (Optional[int]):
                Number of bottleneck channel, not used in this class. 
                It's here for the purpose of api consistency.
        
        Returns:
            None
        """
        super(ResBasicBlock, self).__init__()
        self.l1_1 = nn.Conv2d(c_in, c_out, 3, stride, 1)
        self.l1_2 = nn.BatchNorm2d(c_out)
        self.l1_3 = nn.ReLU()
        self.l2_1 = nn.Conv2d(c_out, c_out, 3, 1, 1)
        self.l2_2 = nn.BatchNorm2d(c_out)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward propogation.

        Args:
            X (Tensor):
                Input tensor.
        
        Returns:
            output (Tensor):
                Output feature maps.
        """
        for layer in self.children():
            X = layer(X)
        return X


class ResBottleneckBlock(nn.Module):
    """The bottleneck transformation block in ResNet-50/101/152.
    Composition: 1x1, BN, AF, 3x3, BN, AF, 1x1, BN.
    """

    def __init__(
            self,
            stride: int,
            c_in: int,
            c_out: int,
            c_b: int
        ) -> None:
        """
        Args:
            stride (int):
                Number of strides, stride=1 will maintain the feature
                size unchanged, stride=2 will halve the feature height 
                and width and double the channel number.
            c_in (int):
                Number of input channel.
            c_out (int):
                Number of output channel.
            c_b (int):
                Number of bottleneck channel.
        
        Returns:
            None.
        """
        super(ResBottleneckBlock, self).__init__()
        self.l1_1 = nn.Conv2d(c_in, c_b, 1, 1, 0)
        self.l1_2 = nn.BatchNorm2d(c_b)
        self.l1_3 = nn.ReLU()
        self.L2_1 = nn.Conv2d(c_b, c_b, 3, stride, 1)
        self.L2_2 = nn.BatchNorm2d(c_b)
        self.L2_3 = nn.ReLU()
        self.l3_1 = nn.Conv2d(c_b, c_out, 1, 1, 0)
        self.l3_2 = nn.BatchNorm2d(c_out)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward propogation.

        Args:
            X (Tensor):
                Input tensor.
        
        Returns:
            output (Tensor):
                Output feature maps.
        """
        for layer in self.children():
            X = layer(X)
        return X


class ResResidualBlock(nn.Module):
    """The residual block used in all ResNet series.
    Composition: F(x) + x, where F(x) is one of `ResBasicBlock`
    and `ResBottleneckBlock`, and x is the shortcut connection.
    """

    def __init__(
            self,
            stride: int,
            c_in: int,
            c_out: int,
            c_b: Optional[int],
            trans_block_type: str
        ) -> None:
        """
        Args:
            stride (int):
                Number of strides, stride=1 will maintain the feature
                size unchanged, stride=2 will halve the feature height 
                and width and double the channel number.
            c_in (int):
                Number of input channel.
            c_out (int):
                Number of output channel.
            c_b (Optional[int]):
                Number of bottleneck channel. If `trans_block_type==
                res_basic_block`, this arg will not be used.
            trans_block_type (str):
                Type of the transformation block, should be one of
                `res_basic_block` or `res_bottleneck_block`.
        
        Returns:
            None
        """
        super(ResResidualBlock, self).__init__()
        trans_block_cls = _get_trans_block(trans_block_type)
        trans_block = trans_block_cls(stride, c_in, c_out, c_b)
        if c_in != c_out or stride > 1:
            shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride, 0),
                nn.BatchNorm2d(c_out)
            )
        else:
            shortcut = None
        self.l1_1 = nn.ModuleDict({
            "shortcut": shortcut,
            "trans_block": trans_block
        })
        self.l1_2 = nn.ReLU()

        self._cfg = {
            "type": type(self).__name__,
            "stride": stride,
            "c_in": c_in,
            "c_out": c_out,
            "c_b": c_b,
            "trans_block_type": trans_block_type
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
        X_ = self.l1_1["trans_block"](X)
        if self.l1_1["shortcut"]:
            X_ += self.l1_1["shortcut"](X)
        X_ = self.l1_2(X_)
        return X_
    
    @property
    def cfg(self) -> Dict[str, Any]:
        return copy.deepcopy(self._cfg)