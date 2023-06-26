# Author: Chang Liu
# Some codes are refered from https://github.com/open-mmlab/mmdetection/tree/2.x


"""Models."""


from torch.nn import Module
from typing import Dict, Any, Callable, Union, Optional

from pytorch_dl.models.classifiers import CLASSIFIERS


MODELS: Dict[str, Dict[str, Module]] = {
    "classification": CLASSIFIERS
}

__all__ = ["MODELS"]