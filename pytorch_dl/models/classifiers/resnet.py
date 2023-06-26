# Author: Chang Liu
# Some codes are refered from https://github.com/facebookresearch/pycls


"""ResNet Classifier moodule."""


from torch import Tensor
from torch.nn import Module
from typing import Dict, Any, List, Optional

from pytorch_dl.models.classifiers.base_classifier import BaseClassifier
from pytorch_dl.models.building_parts.bodies import ResBody
from pytorch_dl.models.building_parts.stems import ResStem
from pytorch_dl.models.building_parts.heads import ResLinearHead, ResConvHead


__all__ = ["ResNetClassifier", "ResNet18Classifier", "ResNet50Classifier"]


class ResNetClassifier(BaseClassifier):
    def __init__(
            self, 
            num_classes: int,
            stage_strides: List[int],
            stage_depths: List[int],
            stage_widths: List[int],
            stage_bottleneck_widths: Optional[List[int]] = None,
        ):
        self.stem = ResStem(64)
        self.body = ResBody(
            64,
            stage_strides,
            stage_depths,
            stage_widths,
            stage_bottleneck_widths
        )
        self.head = ResLinearHead(stage_widths[-1], num_classes)

        self._cfg = {
            "type": type(self).__name__,
            "stem": {
                "type": "ResStem",
                "c_out": 64
            },
            "body": {
                "type": "ResBody",
                "c_in": 64,
                "stage_strides": stage_strides,
                "stage_depths": stage_depths,
                "stage_widths": stage_widths,
                "stage_bottleneck_widths": stage_bottleneck_widths
            },
            "head": {
                "type": "ResLinearHead",
                "c_in": stage_widths[-1],
                "num_classes": num_classes
            }
        }


class ResNet18Classifier(ResNetClassifier):
    def __init__(self, num_classes: int) -> None:
        super(ResNet18Classifier, self).__init__(
            num_classes,
            [1, 2, 2, 2],
            [2, 2, 2, 2],
            [64, 128, 256, 512],
            None
        )


class ResNet50Classifier(ResNetClassifier):
    def __init__(self, num_classes: int) -> None:
        super(ResNet50Classifier, self).__init__(
            num_classes,
            [1, 2, 2, 2],
            [3, 4, 6, 3],
            [256, 512, 1024, 2048],
            [64, 128, 256, 512]
        )