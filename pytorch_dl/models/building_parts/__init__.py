# Author: Chang Liu


"""Model building parts: blocks, stages, stems, bodies, heads."""


import copy
from torch.nn import Module
from typing import Dict, Any

from pytorch_dl.models.building_parts.blocks import ResResidualBlock
from pytorch_dl.models.building_parts.stages import ResStage
from pytorch_dl.models.building_parts.bodies import ResBody
from pytorch_dl.models.building_parts.stems import ResStem
from pytorch_dl.models.building_parts.heads import (
    ResConvHead,
    ResLightHead,
    ResLinearHead
)


__all__ = ["build_stem", "build_body", "build_head"]

_BLOCKS = {
    "ResResidualBlock": ResResidualBlock,
}

_STAGES = {
    "ResStage": ResStage,
}

_BODIES = {
    "ResBody": ResBody,
}

_STEMS = {
    "ResStem": ResStem,
}

_HEADS = {
    "ResConvHead": ResConvHead,
    "ResLinearHead": ResLinearHead,
    "ResLightHead": ResLightHead,
}


def _build_from_cfg(
        registry: Dict[str, Module],
        part_cfg: Dict[str, Any],
        part_name: str
    ) -> Module:
    part_cfg = copy.deepcopy(part_cfg)
    part_type = part_cfg.pop("type")
    if part_cfg == f"Customized{part_name.capitalize()}":
        pass
    else:
        assert part_type in registry.keys(), \
            (f"{part_name.capitalize()} type '{part_type}' "
            f" is not one of supported types '{list(registry.keys())}'.")
        part = registry[part_type](**part_cfg)
    return part


def build_stem(stem_cfg: Dict[str, Any]) -> Module:
    stem = _build_from_cfg(_STEMS, stem_cfg, "stem")
    return stem


def build_body(body_cfg: Dict[str, Any]) -> Module:
    body = _build_from_cfg(_BODIES, body_cfg, "body")
    return body


def build_head(head_cfg: Dict[str, Any]) -> Module:
    head = _build_from_cfg(_HEADS, head_cfg, "head")
    return head