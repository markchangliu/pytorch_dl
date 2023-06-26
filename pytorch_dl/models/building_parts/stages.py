# Author: Chang Liu
# Some codes are refered from https://github.com/facebookresearch/pycls


"""Stage module."""


import copy
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from pytorch_dl.core.utils import build_module_from_cfg
from pytorch_dl.models.building_parts.blocks import ResResidualBlock


_BLOCKS: Dict[str, nn.Module] = {
    "ResResidualBlock": ResResidualBlock
}

__all__ = ["Stages", "ResStage"]


class Stages(nn.Module):
    def __init__(
            self,
            d: int,
            block_cfg: Dict[str, Any]
        ) -> None:
        super(Stages, self).__init__()
        block = build_module_from_cfg("block", block_cfg, _BLOCKS)
        for i in range(d):
            self.add_module(f"block_{i + 1}", block)
        
        self._cfg = {
            "type": type(self).__name__,
            "d": d,
            "block": block_cfg
        }

    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for block in self.children():
            X = block(X)
        return X
    
    
    @property
    def cfg(self) -> Dict[str, Any]:
        return copy.deepcopy(self._cfg)
    

class ResStage(Stages):
    def __init__(
            self, 
            d: int, 
            stride: int,
            c_in: int,
            c_out: int,
            c_b: Optional[int] = None
        ) -> None:
        block_cfg = {
            "type": "ResResidualBlock",
            "stride": stride,
            "c_in": c_in,
            "c_out": c_out
        }
        if c_b:
            block_cfg.update({
                "c_b": c_b,
                "trans_block_type": "ResBottleneckBlock"
            })
        else:
            block_cfg.update({
                "c_b": None,
                "trans_block_type": "ResBasicBlock"
            })

        super(ResStage, self).__init__(d, block_cfg)