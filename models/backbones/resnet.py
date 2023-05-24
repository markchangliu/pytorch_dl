# Author: Chang Liu
# Some codes are refered from https://github.com/facebookresearch/pycls


import torch
import torch.nn as nn
from core.config import cfg
from typing import Union, Optional


# _resnet_type = cfg.RESNET.TYPE


def _get_trans_block(trans_block_name: str) -> Union["ResBasicBlock", "ResBottleneckBlock"]:
    _d = {
        "res_basic_block": ResBasicBlock,
        "res_bottleneck_block": ResBottleneckBlock
    }
    return _d[trans_block_name]


class ResBasicBlock(nn.Module):
    """The basic transformation block in ResNet-18/34.
    """
    
    def __init__(
            self, 
            stride: int,
            c_in: int,
            c_out: int,
            c_b: int = None
        ) -> None:
        super(ResBasicBlock, self).__init__()
        self.a_cnn = nn.Conv2d(c_in, c_out, 3, stride, 1)
        self.a_bn = nn.BatchNorm2d(c_out)
        self.a_af = nn.ReLU()
        self.b_cnn = nn.Conv2d(c_out, c_out, 3, 1, 1)
        self.b_bn = nn.BatchNorm2d(c_out)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for layer in self.children():
            X = layer(X)
        return X


class ResBottleneckBlock(nn.Module):
    """The bottleneck transformation block in ResNet-50/101/152: 
    """

    def __init__(
            self,
            stride: int,
            c_in: int,
            c_out: int,
            c_b: int
        ) -> None:
        super(ResBottleneckBlock, self).__init__()
        self.a_cnn = nn.Conv2d(c_in, c_b, 1, 1, 0)
        self.a_bn = nn.BatchNorm2d(c_b)
        self.a_af = nn.ReLU()
        self.b_cnn = nn.Conv2d(c_b, c_b, 3, stride, 1)
        self.b_bn = nn.BatchNorm2d(c_b)
        self.b_af = nn.ReLU()
        self.c_cnn = nn.Conv2d(c_b, c_out, 1, 1, 0)
        self.c_bn = nn.BatchNorm2d(c_out)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for layer in self.children():
            X = layer(X)
        return X


class ResResidualBlock(nn.Module):
    """The residual block used in all ResNet series.
    """

    def __init__(
            self,
            stride: int,
            c_in: int,
            c_out: int,
            c_b: int,
            trans_block_name: str
        ) -> None:
        super(ResResidualBlock, self).__init__()
        self.res_cnn = None
        self.res_bn = None
        self.af = nn.ReLU()
        if c_in != c_out or stride > 1:
            self.res_cnn = nn.Conv2d(c_in, c_out, 1, stride, 0)
            self.res_bn = nn.BatchNorm2d(c_out)
        trans_block = _get_trans_block(trans_block_name)
        self.trans_block = trans_block(stride, c_in, c_out, c_b)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X_ = self.trans_block(X)
        if self.res_cnn is not None:
            X_ += self.res_bn(self.res_cnn(X))
        X_ = self.af(X_)
        return X_


class ResStage(nn.Module):
    """Stage of ResNet
    """

    def __init__(
            self, 
            c_in: int, 
            c_out: int, 
            stride: int,
            d: int,
            c_b: Optional[int] = None,
        ):
        super(ResStage, self).__init__()
        for i in range(d):
            if i
        


if __name__ == "__main__":
    import sys
    print(sys.path)