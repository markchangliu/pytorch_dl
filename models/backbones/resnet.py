import torch
import torch.nn as nn
from core.config import cfg
from typing import Union


# _resnet_type = cfg.RESNET.TYPE


def _get_trans_block(trans_block_name: str) -> Union["ResBasicBlock", "ResBottleneckBlock"]:
    _d = {
        "res_basic_block": ResBasicBlock,
        "res_bottleneck_block": ResBottleneckBlock
    }
    return _d[trans_block_name]


class ResBasicBlock(nn.Module):
    """Basic block in ResNet-34.

    Layers:
    * conv (kernel_size = 3, stride = stride, 
        padding = 1, w_in = w_in, w_out = w_out)
    * batch_normalization
    * reLu
    * conv (kernel_size = 3, stride = 1, padding = 1,
        w_in = w_out, w_out = w_out)
    * batch_normalization
    * relu (if a residual connection exists,
        apply this relu after the residual)
    """
    
    def __init__(
            self, 
            stride: int,
            w_in: int,
            w_out: int,
            w_b: int = None
        ) -> None:
        super(ResBasicBlock, self).__init__()
        self.a_cnn = nn.Conv2d(w_in, w_out, 3, stride, 1)
        self.a_bn = nn.BatchNorm2d(w_out)
        self.a_af = nn.ReLU()
        self.b_cnn = nn.Conv2d(w_out, w_out, 3, 1, 1)
        self.b_bn = nn.BatchNorm2d(w_out)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for layer in self.children():
            X = layer(X)
        return X


class ResBottleneckBlock(nn.Module):
    """Bottleneck block in ResNet-50/101/152: 
    
    Layers:
    * conv (kernel_size = 1, stride = 1, padding = 0
        w_in = w_in, w_out = w_b)
    * barch_normalization
    * relu
    * conv (kernel_size = 3, stride = stride, padding = 1
        w_in = w_b, w_out = w_b)
    * batch_normalization
    * relu
    * conv (kernel_size = 1, stride = 1, padding = 0,
        w_in = w_b, w_out = w_out)
    * batch_normalization
    * relu (if a residual connection exists,
        apply this relu after the residual)
    """

    def __init__(
            self,
            stride: int,
            w_in: int,
            w_out: int,
            w_b: int
        ) -> None:
        super(ResBottleneckBlock, self).__init__()
        self.a_cnn = nn.Conv2d(w_in, w_b, 1, 1, 0)
        self.a_bn = nn.BatchNorm2d(w_b)
        self.a_af = nn.ReLU()
        self.b_cnn = nn.Conv2d(w_b, w_b, 3, stride, 1)
        self.b_bn = nn.BatchNorm2d(w_b)
        self.b_af = nn.ReLU()
        self.c_cnn = nn.Conv2d(w_b, w_out, 1, 1, 0)
        self.c_bn = nn.BatchNorm2d(w_out)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for layer in self.children():
            X = layer(X)
        return X


class ResResidualBlock(nn.Module):
    """
    Residual block used in all ResNet series.

    Layers:
    * ResBasicBlock or ResBottleneckBlock
    * residual connection
        * if feature map sized is halved:
            * conv (kernel_size = 1, stride = 1, padding = 0
                w_in = w_in, w_out = w_out)
            * batch_normalization
        * else:
            * identy_map
    """

    def __init__(
            self,
            stride: int,
            w_in: int,
            w_out: int,
            w_b: int,
            trans_block: str
        ) -> None:
        super(ResResidualBlock, self).__init__()
        self.res_cnn = None
        self.res_bn = None
        self.af = nn.ReLU()
        if w_in != w_out or stride > 1:
            self.res_cnn = nn.Conv2d(w_in, w_out, 1, stride, 0)
            self.res_bn = nn.BatchNorm2d(w_out)
        trans_block = _get_trans_block(trans_block)
        self.trans_block = trans_block(stride, w_in, w_out, w_b)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X_ = self.trans_block(X)
        if self.res_cnn is not None:
            X_ += self.res_bn(self.res_cnn(X))
        X_ = self.af(X_)
        return X_


if __name__ == "__main__":
    import sys
    print(sys.path)