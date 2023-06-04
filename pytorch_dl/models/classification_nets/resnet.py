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