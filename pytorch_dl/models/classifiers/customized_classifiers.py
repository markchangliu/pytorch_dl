# Author: Chang Liu


"""Customized classifiers."""


from torch import Tensor
from torch.nn import Module
from typing import Dict, Any

from pytorch_dl.models.building_parts import *


class CustomizedClassifier(Module):
    def __init__(
            self,
            stem_cfg: Dict[str, Any],
            body_cfg: Dict[str, Any],
            head_cfg: Dict[str, Any],
        ):
        super(CustomizedClassifier, self).__init__()
        self.stem = build_stem(stem_cfg)
        self.body = build_body(body_cfg)
        self.head = build_head(head_cfg)


    def forward(self, X: Tensor) -> Tensor:
        for part in self.children():
            X = part(X)
        return X