# Author: Chang Liu
# Some codes are refered from https://github.com/open-mmlab/mmcv


"""Model building module."""


import torch
import torch.nn as nn
from pytorch_dl.core.registry import Registry


stem_registry = Registry("stem")