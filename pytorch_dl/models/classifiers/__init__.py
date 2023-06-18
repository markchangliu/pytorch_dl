# Author: Chang Liu


"""Classifiers."""


import copy
import torch
from torch.nn import Module
from typing import Dict, Any, Callable, Union, Optional

from pytorch_dl.models.classifiers.customized_classifiers import (
    CustomizedClassifier
)
from pytorch_dl.models.classifiers.resnet import (
    ResNetClassifier, 
    ResNet18Classifier,
    ResNet50Classifier
)


__all__ = ["build_classifier"]

_CLASSIFIERS = {
    "CustomizedClassifier": CustomizedClassifier,
    "ResNetClassifier": ResNetClassifier,
    "ResNet18Classifier": ResNet18Classifier,
    "ResNet50Classifier": ResNet50Classifier
}


def build_classifier(
        classifier_cfg: Dict[str, Callable[..., Any]]
    ) -> Module:
    classifier_cfg = copy.deepcopy(classifier_cfg)
    classifier_type = classifier_cfg.pop("type")
    assert classifier_type in _CLASSIFIERS.keys(), \
        (f"Classifier type '{classifier_type}' is not one of the "
        f"supported types '{list(_CLASSIFIERS.keys())}'.")
    if classifier_type == "CustomizedClassifier":
        stem_cfg = classifier_cfg["stem"]
        body_cfg = classifier_cfg["body"]
        head_cfg = classifier_cfg["head"]
        classifier = CustomizedClassifier(stem_cfg, body_cfg, head_cfg)
    else:
        classifier = _CLASSIFIERS[classifier_type](**classifier_cfg)
    return classifier