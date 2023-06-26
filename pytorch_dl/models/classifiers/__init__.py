# Author: Chang Liu


"""Classifiers."""


import copy
import torch
from torch.nn import Module
from typing import Dict, Any, Callable, Union, Optional

from pytorch_dl.models.classifiers.base_classifier import BaseClassifier
from pytorch_dl.models.classifiers.resnet import (
    ResNetClassifier, 
    ResNet18Classifier,
    ResNet50Classifier
)


CLASSIFIERS: Dict[str, Module] = {
    "BaseClassifier": BaseClassifier,
    "ResNetClassifier": ResNetClassifier,
    "ResNet18Classifier": ResNet18Classifier,
    "ResNet50Classifier": ResNet50Classifier
}

__all__ = ["CLASSIFIERS"]