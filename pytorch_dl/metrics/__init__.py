# Author: Chang Liu


"""Metrics."""


import copy
from torch import Tensor
from typing import Callable, Dict, Any

from pytorch_dl.metrics.classifications import ClassifierMetric


METRICS = {
    "ClassifierMetric": ClassifierMetric
}


__all__ = ["METRICS"]