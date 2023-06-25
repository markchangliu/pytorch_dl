# Author: Chang Liu
# Some codes are refered from https://github.com/facebookresearch/pycls


"""ResNet Classifier moodule."""


from torch import Tensor
from torch.nn import Module
from typing import Dict, Any

from pytorch_dl.models.building_parts.stems import ResStem
from pytorch_dl.models.building_parts.bodies import ResBody
from pytorch_dl.models.building_parts.heads import (
    ResConvHead,
    ResLightHead,
    ResLinearHead
)


_STEMS = {"ResStem": ResStem}
_BODIES = {"ResBody": ResBody}
_HEADS = {
    "ResConvHead": ResConvHead, 
    "ResLightHead": ResLightHead,
    "ResLinearHead": ResLinearHead
}


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

        stem_type = stem_cfg.pop("type")
        assert stem_type in _STEMS.keys(), \
            f"Stem type '{stem_type}' is not one of the supported types '{_STEMS.keys()}'."
        self.stem = _STEMS[stem_type](**stem_cfg)
        
        body_type = body_cfg.pop("type")
        assert body_type in _BODIES.keys(), \
            f"Body type '{body_type}' is not one of the supported bodies '{_BODIES.keys()}'."
        self.body = _BODIES[body_type](**body_cfg)

        head_type = head_cfg.pop("type")
        assert head_type in _HEADS.keys(), \
            f"Head type '{head_type}' is not one of the supported heads '{_HEADS.keys()}'."
        self.head = _HEADS[head_type](**head_cfg)
    

    def forward(self, X: Tensor) -> Tensor:
        for building_part in self.children():
            X = building_part(X)
        return X
    

class ResNet18Classifier(ResNetClassifier):
    def __init__(self, num_classes: int) -> None:
        stem_cfg = {
            "type": "ResStem",
            "c_out": 64
        }
        body_cfg = {
            "type": "ResBody",
            "stage_strides": [1, 2, 2, 2],
            "stage_depths": [2, 2, 2, 2],
            "stage_widths": [64, 128, 256, 512],
            "stage_bottleneck_widths": None,
            "trans_block_type": "ResBasicBlock"
        }
        head_cfg = {
            "type": "ResLinearHead",
            "c_in": 512,
            "num_classes": num_classes
        }
        super(ResNet18Classifier, self).__init__(
            num_classes,
            stem_cfg,
            body_cfg,
            head_cfg
        )


class ResNet50Classifier(ResNetClassifier):
    def __init__(self, num_classes: int) -> None:
        stem_cfg = {
            "type": "ResStem",
            "c_out": 64
        }
        body_cfg = {
            "type": "ResBody",
            "stage_strides": [1, 2, 2, 2],
            "stage_depths": [3, 4, 6, 3],
            "stage_widths": [256, 512, 1024, 2048],
            "stage_bottleneck_widths": [64, 128, 256, 512],
            "trans_block_type": "ResBottleneckBlock"
        }
        head_cfg = {
            "type": "ResLinearHead",
            "c_in": 2048,
            "num_classes": num_classes
        }
        super(ResNet50Classifier, self).__init__(
            num_classes,
            stem_cfg,
            body_cfg,
            head_cfg
        )