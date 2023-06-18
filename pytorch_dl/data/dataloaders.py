# Author: Chang Liu


"""Dataloaders."""


import copy
from torch.utils.data import DataLoader
from typing import Dict, Any

from pytorch_dl.data.datasets import build_dataset


def build_dataloader(
        dataloader_cfg: Dict[str, Any]
    ) -> DataLoader:
    dataloader_cfg = copy.deepcopy(dataloader_cfg)
    dataset_cfg = dataloader_cfg.pop("dataset")
    dataset = build_dataset(dataset_cfg)
    dataloader_cfg.update({"dataset": dataset})
    dataloader = DataLoader(**dataloader_cfg)