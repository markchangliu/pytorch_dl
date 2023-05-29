# Author: Chang Liu
# Some codes are refered from https://github.com/open-mmlab/mmdetection/tree/2.x


"""Init models package."""


from pytorch_dl.models.stems import ResStem
from pytorch_dl.models.stages import ResStage
from pytorch_dl.models.heads import (
    ResLinearHead,
    ResConvHead,
    ResLightHead
)
from pytorch_dl.models.nets import (
    ResNet,
    ResNet18
)