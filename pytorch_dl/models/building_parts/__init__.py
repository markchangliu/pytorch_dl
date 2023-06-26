# Author: Chang Liu


"""Model building parts: blocks, stages, stems, bodies, heads."""


import copy
from torch.nn import Module
from typing import Dict, Any

from pytorch_dl.models.building_parts.stems import ResStem
from pytorch_dl.models.building_parts.bodies import Body, ResBody
from pytorch_dl.models.building_parts.heads import (
    ResConvHead,
    ResLightHead,
    ResLinearHead
)


STEMS = {
    "ResStem": ResStem,
}

BODIES = {
    "Body": Body,
    "ResBody": ResBody,
}

HEADS = {
    "ResConvHead": ResConvHead,
    "ResLightHead": ResLightHead,
    "ResLinearHead": ResLinearHead
}

__all__ = ["STEMS", "BODIES", "HEADS"]