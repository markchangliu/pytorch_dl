# Author: Chang Liu


"""Datasets."""


import copy
from torch.utils.data import Dataset
from typing import Dict, Any

from pytorch_dl.data.datasets.classification import (
    TrainTestImgFolderDataset,
    TrainTestPickleDataset,
    InferenceImgFolderDataset,
    InferencePickleDataset
)


DATASETS = {
    "TrainTestImgFolderDataset": TrainTestImgFolderDataset,
    "TrainTestPickleDataset": TrainTestPickleDataset,
    "InferenceImgFolderDataset": InferenceImgFolderDataset,
    "InferencePickleDataset": InferencePickleDataset
}

__all__ = ["DATASETS"]