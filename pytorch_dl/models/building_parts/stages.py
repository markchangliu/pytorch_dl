# Author: Chang Liu
# Some codes are refered from https://github.com/facebookresearch/pycls


"""Stage module."""


import torch
import torch.nn as nn
from typing import Optional
from pytorch_dl.models.building_parts.blocks import ResResidualBlock
from pytorch_dl.models.builder import stage_registry


@stage_registry.register_module("ResStage")
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
            c_b (Optional[int]):
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
            self.add_module(f"block_{i}", res_block)
                
                
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward propogation.

        Args:
            X (Tensor):
                Input tensor.
        
        Returns:
            output (Tensor):
                Output feature maps.
        """
        for res_block in self.children():
            X = res_block(X)
        return X
