# Author: Chang Liu


"""Base Classifier."""


import copy
from torch import Tensor
from torch.nn import Module
from typing import Dict, Any

from pytorch_dl.models.building_parts.bodies import ResBody
from pytorch_dl.models.building_parts.stems import ResStem
from pytorch_dl.models.building_parts.heads import (
    ResConvHead,
    ResLightHead,
    ResLinearHead
)


BODIES: Dict[str, Module] = {
    "ResBody": ResBody,
}

STEMS: Dict[str, Module] = {
    "ResStem": ResStem,
}

HEADS: Dict[str, Module] = {
    "ResConvHead": ResConvHead,
    "ResLinearHead": ResLinearHead,
    "ResLightHead": ResLightHead,
}

__all__ = ["BaseClassifier"]


def _build_from_cfg(name: str, cfg: Dict[str, Any]) -> Module:
    cfg = copy.deepcopy(cfg)
    module_type = cfg.pop("type")
    if name == "stem":
        modules = STEMS
    elif name == "body":
        modules = BODIES
    elif name == "head":
        modules = HEADS
    assert module_type in modules.keys(), \
        (f"{name.capitalize()} type '{module_type}' is not one of "
        f"the supported types '{list(modules.keys())}'")
    module = modules[module_type]
    return module(**cfg)


class BaseClassifier(Module):
    def __init__(
            self,
            stem_cfg: Dict[str, Any],
            body_cfg: Dict[str, Any],
            head_cfg: Dict[str, Any]
        ):
        super(BaseClassifier, self).__init__()
        self.stem = _build_from_cfg("stem", stem_cfg)
        self.body = _build_from_cfg("body", body_cfg)
        self.head = _build_from_cfg("head", head_cfg)
        
    def forward(self, X: Tensor) -> Tensor:
        for module in self.children():
            X = module(X)
        return X