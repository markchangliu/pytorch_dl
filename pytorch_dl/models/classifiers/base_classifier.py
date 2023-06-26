# Author: Chang Liu


"""Base Classifier."""


import copy
from torch import Tensor
from torch.nn import Module
from typing import Dict, Any

from pytorch_dl.core.utils import build_module_from_cfg
from pytorch_dl.models.building_parts import STEMS, BODIES, HEADS

__all__ = ["BaseClassifier"]


class BaseClassifier(Module):
    def __init__(
            self,
            stem_cfg: Dict[str, Any],
            body_cfg: Dict[str, Any],
            head_cfg: Dict[str, Any]
        ):
        super(BaseClassifier, self).__init__()
        self.stem = build_module_from_cfg("stem", stem_cfg, STEMS)
        self.body = build_module_from_cfg("body", body_cfg, BODIES)
        self.head = build_module_from_cfg("head", head_cfg, HEADS)

        self._cfg = {
            "type": type(self).__name__,
            "stem": stem_cfg,
            "body": body_cfg,
            "head": head_cfg
        }
        
    def forward(self, X: Tensor) -> Tensor:
        for module in self.children():
            X = module(X)
        return X
    
    @property
    def cfg(self) -> Dict[str, Any]:
        return copy.deepcopy(self._cfg)