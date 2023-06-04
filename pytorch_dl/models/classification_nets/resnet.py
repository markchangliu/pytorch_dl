# Author: Chang Liu
# Some codes are refered from https://github.com/facebookresearch/pycls


"""ResNet Classifier moodule."""


from torch import Tensor
from torch.nn import Module
from typing import Dict, Any

from pytorch_dl.models.builder import (
        classifier_registry,
        build_stem,
        build_body,
        build_head
    )


@classifier_registry.register_module("ResNetClassifier")
class ResNetClassifier(Module):
    def __init__(
            self,
            num_classes: int,
            stem_cfg: Dict[str, Any],
            body_cfg: Dict[str, Any],
            head_cfg: Dict[str, Any]
        ) -> None:
        super(ResNetClassifier, self).__init__()
        assert type(num_classes) is int, (
            "`num_classes` should be an int."
        )
        self.stem = build_stem(stem_cfg)
        self.body = build_body(body_cfg)
        self.head = build_head(head_cfg)
    

    def forward(self, X: Tensor) -> Tensor:
        for building_part in self.children():
            X = building_part(X)
        return X
    

@classifier_registry.register_module("ResNet18Classifier")
class ResNet18Classifier(ResNetClassifier):
    def __init__(self, num_classes: int) -> None:
        stem_cfg = {
            "name": "ResStem",
            "c_out": 64
        }
        body_cfg = {
            "name": "ResBody",
            "stage_strides": [1, 2, 2, 2],
            "stage_depths": [2, 2, 2, 2],
            "stage_widths": [64, 128, 256, 512],
            "stage_bottleneck_widths": None,
            "trans_block_name": "ResBasicBlock"
        }
        head_cfg = {
            "name": "ResLinearHead",
            "c_in": 512,
            "num_classes": num_classes
        }
        super(ResNet18Classifier, self).__init__(
            num_classes,
            stem_cfg,
            body_cfg,
            head_cfg
        )


@classifier_registry.register_module("ResNet50Classifier")
class ResNet50Classifier(ResNetClassifier):
    def __init__(self, num_classes: int) -> None:
        stem_cfg = {
            "name": "ResStem",
            "c_out": 64
        }
        body_cfg = {
            "name": "ResBody",
            "stage_strides": [1, 2, 2, 2],
            "stage_depths": [3, 4, 6, 3],
            "stage_widths": [256, 512, 1024, 2048],
            "stage_bottleneck_widths": [64, 128, 256, 512],
            "trans_block_name": "ResBottleneckBlock"
        }
        head_cfg = {
            "name": "ResLinearHead",
            "c_in": 2048,
            "num_classes": num_classes
        }
        super(ResNet50Classifier, self).__init__(
            num_classes,
            stem_cfg,
            body_cfg,
            head_cfg
        )