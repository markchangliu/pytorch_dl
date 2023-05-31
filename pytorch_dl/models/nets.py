# Author: Chang Liu
# Some codes are refered from https://github.com/facebookresearch/pycls


"""Network module."""


import torch
import torch.nn as nn
from pytorch_dl.core.logging import get_logger
from pytorch_dl.models.stems import ResStem
from pytorch_dl.models.stages import ResStage
from pytorch_dl.models.heads import (
    ResLinearHead,
    ResConvHead,
    ResLightHead
)
from pytorch_dl.models.builder import net_registry
from typing import Union, Optional, List, Dict, Any


_VALID_TRANS_BLOCK_NAMES = ["ResBasicBlock", "ResBottleneckBlock"]
_VALID_HEAD_NAMES = ["ResLinearHead", "ResConvHead", "ResLightHead"]


_logger = get_logger(__name__)


def _get_head(
        head_name: str
    ) -> Union["ResLinearHead", "ResConvHead", "ResLightHead"]:
    """Return ResNet head class based on the string name provided by
    head_name.

    Args:
        head_name (str):
            The name of head, should be one of `ResLinearHead`, 
            `ResConvHead`, `ResLightHead`.
    
    Returns:
        ResHead (Union["ResLinearHead", "ResConvHead", "ResLightHead"]):
            The ResNet head class.
    """
    _d = {
        "ResLinearHead": ResLinearHead,
        "ResConvHead": ResConvHead,
        "ResLightHead": ResLightHead
    }
    return _d[head_name]


############### Classification nets ###############

@net_registry.register_module("ResNet")
class ResNet(nn.Module):
    """A ResNet classification net.
    Composition: ResStem + d * ResBlock + ResHead.
    """

    def __init__(
            self, 
            num_classes: int,
            stage_depths: List[int],
            stage_strides: List[int],
            stage_widths: List[int],
            stage_bottleneck_widths: Optional[List[Optional[int]]],
            trans_block_name: str,
            head_name: str
        ) -> None:
        f"""Args:
            num_classes (int):
                The number of classes.
            stage_depths (List[int]):
                The depth of each stage.
            stage_strides (List[int]):
                The stride of each stage.
            stage_widths (List[int]):
                The number of output channel of each stage.
            stage_bottleneck_widths (Optional[List[Optional[int]]]):
                The number of bottleneck channels.
            trans_block_name (str):
                The name of transformation block in ResStage, should be one of
                {_VALID_TRANS_BLOCK_NAMES}.
            head_name (str):
                The name of head, should be one of {_VALID_HEAD_NAMES}.
        
        Returns:
            None.
        """
        super(ResNet, self).__init__()
        self.param_dict = {
            "num_classes": num_classes,
            "stage_depths": stage_depths,
            "stage_widths": stage_widths,
            "stage_strides": stage_strides,
            "stage_bottleneck_widths": stage_bottleneck_widths,
            "trans_block_name": trans_block_name,
            "head_name": head_name
        }
        self._param_check()
        self._construct_resnet()
    
    
    def _param_check(self) -> None:
        f"""Perform the following parameter check:
            * If 'trans_block_name' is one of {_VALID_TRANS_BLOCK_NAMES}.
            * If 'head_name' is one of {_VALID_HEAD_NAMES}.
            * If 'stage_depths', 'stage_strides', 'stage_widths' have equal
            lengths for 'ResBasicBlock'.
            * If 'stage_depths', 'stage_strides', 'stage_widths', 
            'stage_bottleneck_widths' have equal lengths for
            'ResBottleneckBlock'.
            * For 'ResLightHead', if `stage_widths[-1] != num_classes`,
            raise a warning and reassign `stage_widths[-1] = num_classes`.
        
        Args:
            self.

        Returns:
            None.
        """
        num_classes = self.param_dict["num_classes"]
        stage_depths = self.param_dict["stage_depths"]
        stage_strides = self.param_dict["stage_strides"]
        stage_widths = self.param_dict["stage_widths"]
        stage_bottleneck_widths = self.param_dict["stage_bottleneck_widths"]
        trans_block_name = self.param_dict["trans_block_name"]
        head_name = self.param_dict["head_name"]
        
        assert trans_block_name in _VALID_TRANS_BLOCK_NAMES, \
            ("'trans_block_name={0}' is not a valid block. Valid blocks "
            "are {1}".format(trans_block_name, _VALID_TRANS_BLOCK_NAMES))
        assert head_name in _VALID_HEAD_NAMES, \
            ("'head_name={0}' is not a valid head. Valid heads "
            "are {1}".format(head_name, _VALID_HEAD_NAMES))
        if trans_block_name == "ResBasicBlock":
            assert len(stage_depths) == len(stage_strides) == len(stage_widths), \
                ("'stage_depths', 'stage_strides', 'stage_widths' should "
                "have equal lengths.")
            self.param_dict["stage_bottleneck_widths"] = \
                [None for i in range(len(stage_depths))]
        elif trans_block_name == "ResBottleneckBlock":
            assert len(stage_depths) == len(stage_strides) \
                == len(stage_widths) == len(stage_bottleneck_widths), \
                ("'stage_depths', 'stage_strides', 'stage_widths', "
                "'stage_bottleneck_widths' should have equal lengths.")
        else:
            pass
        
        if head_name == "ResLightHead":
            if stage_widths[-1] != num_classes:
                _logger.warn(
                    "You are using 'ResLightHead', but your 'stage_widths[-1]' "
                    "is not equal to 'num_class'. The program will reassign it to "
                    "'num_classes', but you should be conscious of this reassignment "
                    "and ensure it follows your willness."
                )
                stage_widths[-1] = num_classes

    
    def _construct_resnet(self,) -> None:
        """Build ResNet.
        
        Args:
            self.
        Returns:
            None.
        """
        num_classes = self.param_dict["num_classes"]
        stage_depths = self.param_dict["stage_depths"]
        stage_strides = self.param_dict["stage_strides"]
        stage_widths = self.param_dict["stage_widths"]
        stage_bottleneck_widths = self.param_dict["stage_bottleneck_widths"]
        trans_block_name = self.param_dict["trans_block_name"]
        head_name = self.param_dict["head_name"]

        c_init = stage_widths[0]
        c_last = stage_widths[0]
        res_stages = []
        head_cls = _get_head(head_name)
        for s, d, c, c_b in zip(stage_strides, 
                stage_depths, stage_widths, stage_bottleneck_widths):
            res_stage = ResStage(s, c_last, c, d, trans_block_name, c_b)
            res_stages.append(res_stage)
            c_last = c
        self.l0 = ResStem(c_init)
        self.l1 = nn.ModuleList(res_stages)
        self.l2 = head_cls(c_last, num_classes)

    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward propogation.

        Args:
            X (Tensor):
                Input tensor.
        
        Returns:
            logit (Tensor):
                The predicted classification logit.
        """
        X = self.l0(X)
        for res_stage in self.l1:
            X = res_stage(X)
        X = self.l2(X)
        return X


@net_registry.register_module("ResNet18")
class ResNet18(ResNet):
    """A ResNet-18 classification net."""

    def __init__(
            self,
            num_classes: int
        ) -> None:
        """Args:
            num_classes (int):
                The number of classes.
        
        Returns:
            None.
        """
        param_dict = {
            "num_classes": num_classes,
            "stage_strides": [1, 2, 2, 2],
            "stage_depths": [2, 2, 2, 2],
            "stage_widths": [64, 128, 256, 512],
            "stage_bottleneck_widths": None,
            "trans_block_name": "ResBasicBlock",
            "head_name": "ResLinearHead",
        }
        super(ResNet18, self).__init__(**param_dict)
