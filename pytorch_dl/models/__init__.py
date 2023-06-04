# Author: Chang Liu
# Some codes are refered from https://github.com/open-mmlab/mmdetection/tree/2.x


"""Model init code.

All registered class and functions should be imported here.
"""


from pytorch_dl.models.building_parts.stems import ResStem
from pytorch_dl.models.building_parts.bodies import ResBody
from pytorch_dl.models.building_parts.heads import (
    ResLinearHead,
    ResConvHead,
    ResLightHead
)
from pytorch_dl.models.classification_nets.resnet import (
    ResNetClassifier,
    ResNet18Classifier,
    ResNet50Classifier
)