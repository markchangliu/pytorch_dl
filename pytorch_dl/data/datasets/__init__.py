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


__all__ = ["build_dataset"]



def _build_classification_dataset(
        dataset_cfg: Dict[str, Any]
    ) -> Dataset:
    CLASSIFICATION_DS = {
        "TrainTestImgFolderDataset": TrainTestImgFolderDataset,
        "TrainTestPickleDataset": TrainTestPickleDataset,
        "InferenceImgFolderDataset": InferenceImgFolderDataset,
        "InferencePickleDataset": InferencePickleDataset
    }

    dataset_cfg = copy.deepcopy(dataset_cfg)
    dataset_type = dataset_cfg.pop("type")
    assert dataset_type in CLASSIFICATION_DS.keys(), \
        (f"Dataset_type '{dataset_type}' is not one of the supported "
         f"dataset types {list(CLASSIFICATION_DS.keys())}.")
    
    transform_cfgs = dataset_cfg.pop("transforms", None)
    dataset_cfg.update({"transform_cfgs": transform_cfgs})
    dataset = CLASSIFICATION_DS[dataset_type](dataset_cfg)
    return dataset


def build_dataset(
        task_type: str,
        dataset_cfg: Dict[str, Any]
    ) -> Dataset:
    DATASET_TYPES = {
        "classification": _build_classification_dataset
    }
    dataset = DATASET_TYPES[task_type](dataset_cfg)
    return dataset