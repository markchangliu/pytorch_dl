# Author: Chang Liu
# Some codes are refered from https://github.com/facebookresearch/pycls


"""ResNet backbone module."""


import torch
import torch.nn as nn
from typing import Union, Optional, List



def _get_trans_block(
        trans_block_name: str
    ) -> Union["ResBasicBlock", "ResBottleneckBlock"]:
    """Return ResNet Block class based on the string name provided by
    `trans_blcok_name`.

    Args:
        trans_block_name (str):
            The name of transformation block, should be one of 
            `res_basic_block` or `res_bottleneck_block`.
    Returns:
        ResBlockClass (Union["ResBasicBlock", "ResBottleneckBlock"]):
            The Transformation Block class that will be used in 
            ResNet construction.
    """
    _d = {
        "res_basic_block": ResBasicBlock,
        "res_bottleneck_block": ResBottleneckBlock
    }
    return _d[trans_block_name]


def _init_weight(m: nn.Module) -> None:
    """Performs Xavier initialization to Conv and linear layers,
    constant initialization to batchnormalization layers.
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)     


class ResBasicBlock(nn.Module):
    """The basic transformation block in ResNet-18/34.
    Composition: 3x3, BN, AF, 3x3, BN.
    """
    
    def __init__(
            self, 
            stride: int,
            c_in: int,
            c_out: int,
            c_b: int = None
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
            None
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
            None
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
            c_b: int,
            trans_block_name: str
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
                Number of bottleneck channel. If `trans_block_name==
                res_basic_block`, this arg will not be used.
            trans_block_name (str):
                Name of the transformation block, should be one of
                `res_basic_block` or `res_bottleneck_block`.
        
        Returns:
            None
        """
        super(ResResidualBlock, self).__init__()
        trans_block_cls = _get_trans_block(trans_block_name)
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
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward propogation.

        Args:
            X (Tensor):
                Input tensor.
        
        Returns:
            None
        """
        X_ = self.l1_1["trans_block"](X)
        if self.l1_1["shortcut"]:
            X_ += self.self.l1_1["shortcut"](X)
        X_ = self.l1_2(X_)
        return X_


class ResStage(nn.Module):
    """The stage of ResNet. 
    Composition: `d` numbers of `ResResidualBlock`.
    """

    def __init__(
            self, 
            stride: int,
            c_in: int, 
            c_out: int, 
            d: int,
            trans_block_name: str,
            c_b: Optional[int] = None,
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
            d (int):
                Depth, or the number of blocks, in this stage.
            trans_block_name (str):
                Name of the transformation block, should be one of
                `res_basic_block` or `res_bottleneck_block`.
            c_b (int):
                Number of bottleneck channel. If `trans_block_name==
                res_basic_block`, this arg will not be used. Default
                None.
        
        Returns:
            None
        """
        super(ResStage, self).__init__()
        res_blocks = []
        for i in range(d):
            if i == 0:
                res_block = ResResidualBlock(stride, c_in, c_out, 
                    c_b, trans_block_name)
            else:
                res_block = ResResidualBlock(1, c_out, c_out, c_b,
                    trans_block_name)
            res_blocks.append(res_block)
        self.res_blocks = nn.ModuleList(res_blocks)
                
                
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward propogation.

        Args:
            X (Tensor):
                Input tensor.
        
        Returns:
            None
        """
        for res_block in self.res_blocks:
            X = res_block(X)
        return X
    

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

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward propogation.

        Args:
            X (Tensor):
                Input tensor.
        
        Returns:
            None
        """
        X = self.l1(X)
        return X
    

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
            None
        """
        X = self.l1_1(X)
        X = self.l1_2(X.contiguous().view(X.shape[0], -1))
        X = self.l1_2(X)
        return X
    

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
            None
        """
        X = self.l1_1(X)
        X = self.l1_2(X.contiguous().view(X.shape[0], -1))
        X = self.l1_2(X)
        return X
    

class ResNet(nn.Module):
    """ResNet.
    Composition: ResStem + d * ResBlock + ResHead.
    """

    def __init__(
            self, 
            num_classes: int,
            stage_depths: List[int],
            stage_widths: List[int],
            stage_bottleneck_widths: List[Optional[int]],
            trans_block_name: str,
        ) -> None:
        """Args:
            stage_depths (List[int]):
                The depth of each stage.
            stage_widths (List[int]):
                The number of output channel of each stage.
            stage_bottleneck_widths (List[int]):
                The number of bottleneck channels.
            trans_block_name (str):
                The name of transformation block in ResStage, should be either
                `res_basic_block` or `res_bottleneck_block`.
            num_classes (int):
                The number of classes.
        """
        super(ResNet, self).__init__()
        assert len(stage_depths) == 0 and len(stage_widths), \
            "stage_depths and stage_widths should have 4 elements."
        stage_strides = [1, 2, 2, 2]
        c_init = stage_widths[0]
        c_last = stage_widths[0]
        res_stages = []
        for s, d, c, c_b in zip(stage_strides, 
                stage_depths, stage_widths, stage_bottleneck_widths):
            res_stage = ResStage(s, c_last, c, d, trans_block_name, c_b)
            res_stages.append(res_stage)
            c_last = c
        self.l0 = ResStem(c_init)
        self.l1 = nn.ModuleList(res_stages)
        self.l2 = ResLinearHead(c_last, num_classes)
        self.apply(_init_weight)
    

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward propogation.

        Args:
            X (Tensor):
                Input tensor.
        
        Returns:
            None
        """
        X = self.l0(X)
        for res_stage in self.l1:
            X = res_stage(X)
        X = self.l2(X)
        return X


if __name__ == "__main__":
    pass