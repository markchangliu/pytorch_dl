# Author: Chang Liu


"""Data module."""


from pytorch_dl.data.datasets import DATASETS
from pytorch_dl.data.transforms import (
    ResizePad,
    RandomCrop,
    FixedCrop,
    ToTensor
)


TRANSFORMS = {
    "ResizePad": ResizePad,
    "RandomCrop": RandomCrop,
    "FixedCrop": FixedCrop,
    "ToTensor": ToTensor
}

__all__ = ["TRANSFORMS", "DATASETS"]