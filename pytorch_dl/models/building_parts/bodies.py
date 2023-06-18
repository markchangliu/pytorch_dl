# Author: Chang Liu
# Some codes are refered from https://github.com/facebookresearch/pycls


"""Body module."""


import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any

from pytorch_dl.models.building_parts.stages import ResStage


class ResBody(nn.Module):
    def __init__(
            self,
            stage_strides: List[int],
            stage_depths: List[int],
            stage_widths: List[int],
            stage_bottleneck_widths: Optional[List[int]],
            trans_block_type: str
        ) -> None:
        super(ResBody, self).__init__()
        param_dict = {
            "stage_depths": stage_depths,
            "stage_widths": stage_widths,
            "stage_strides": stage_strides,
            "stage_bottleneck_widths": stage_bottleneck_widths,
            "trans_block_type": trans_block_type,
        }
        self._param_check(param_dict)
        self._construct_body()
            
    
    def _param_check(self, param_dict: Dict[str, Any]) -> None:
        stage_depths = param_dict["stage_depths"]
        stage_strides = param_dict["stage_strides"]
        stage_widths = param_dict["stage_widths"]
        stage_bottleneck_widths = param_dict["stage_bottleneck_widths"]
        trans_block_type = param_dict["trans_block_type"]
        _valid_trans_block_types = ["ResBasicBlock", "ResBottleneckBlock"]

        assert trans_block_type in _valid_trans_block_types, \
            ("'trans_block_type={0}' is not a valid block. Valid blocks "
            "are {1}".format(trans_block_type, _valid_trans_block_types))
        
        if trans_block_type == "ResBasicBlock":
            assert len(stage_depths) == len(stage_strides) == len(stage_widths), \
                ("`stage_depths`, `stage_strides`, `stage_widths` should "
                "be equal in length.")
            stage_bottleneck_widths = [None for i in range(len(stage_depths))]
        elif trans_block_type == "ResBottleneckBlock":
            assert len(stage_depths) == len(stage_strides) \
                == len(stage_widths) == len(stage_bottleneck_widths), \
                ("`stage_depths`, `stage_strides`, `stage_widths`, "
                "`stage_bottleneck_widths` should be equal in length.")
            assert None not in stage_bottleneck_widths, \
                ("`stage_bottleneck_widths` should not have `None`.")
        else:
            pass

        self.stage_depths = stage_depths
        self.stage_strides = stage_strides
        self.stage_widths = stage_widths
        self.stage_bottleneck_widths = stage_bottleneck_widths
        self.trans_block_type = trans_block_type

    
    def _construct_body(self) -> None:
        c_last = self.stage_widths[0]
        stage_idx = 0
        for s, d, c, c_b in zip(self.stage_strides, self.stage_depths, 
            self.stage_widths, self.stage_bottleneck_widths):
            res_stage = ResStage(s, c_last, c, d, self.trans_block_type, c_b)
            self.add_module(f"stage_{stage_idx}", res_stage)
            c_last = c
            stage_idx += 1
    

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for stage in self.children():
            X = stage(X)
        return X