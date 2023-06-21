# Author: Chang Liu


"""Train a model."""


import copy
import os
import torch
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple

from pytorch_dl.runners import build_runner
from pytorch_dl.models import build_model
from pytorch_dl.data.dataloaders import build_dataloader
from pytorch_dl.core.optimizers import build_optimizer_scheduler
from pytorch_dl.core.logging import get_logger


_logger = get_logger(__name__)


def setup_task(
        cfg: Dict[str, Any]
    ) -> Tuple[object, Module, Optimizer, Scheduler, DataLoader, Dict[str, Any]]:
    cfg = copy.deepcopy(cfg)
    task_cfg = cfg["task"]

    supported_types = ["classification"]
    task_type = task_cfg.get("type", None)
    assert task_type, \
        (f"`task.type` should be configured to one of the supported types "
         f"'{supported_types}'. ")
    assert task_type in supported_types, \
        (f"Task type '{task_type}' is not one of the supported types "
         f"'{supported_types}'.")

    if task_type == "classification":
        num_classes = task_cfg.get("num_classes", None)
        assert num_classes, \
            f"`task.num_classes` should be configured for a classification task."
        
        class_names = task_cfg.get("class_names", None)
        if class_names is None:
            _logger.info(
                f"`task.class_names` is not configured, thus "
                f"`range(task.num_classes)` is used as class names."
            )
            task_cfg.update({"class_names": list(range(num_classes))})
        
    work_dir = task_cfg.get("work_dir", None)
    if work_dir is None:
        _logger.info(
            f"`task.work_dir` is not configured, thus `./tmp` "
            f"is used as the working directory."
        )
        task_cfg.update({"work_dir": os.getcwd()})
    
    runner_type = task_cfg.get("runner_type", None)
    supported_runner_types = ["SingleGpuRunner", "DataParallelRunner"]
    if runner_type is None:
        _logger.info(
            f"`task.runner_type` is not configured, thus "
            f"`SingleGpuRunner` is applied."
        )
        runner_type = "SingleGpuRunner"
        task_cfg.update({"runner_type": runner_type})
    assert runner_type in supported_runner_types, \
        (f"`task.runner_type` is not one of the supported runner "
         f"types '{supported_runner_types}'")
    
    assert torch.cuda.is_available(), \
        (f"No availble GPU device!")
    
    device_ids = task_cfg.get("device_ids", None)
    if device_ids is None:
        _logger.info(
            f"`task.device_ids` is not configured, thus "
            f"all available devices will be used for parallel tasks."
        )
        task_cfg.update({"device_ids": list(range(torch.cuda.device_count()))})
    assert len(device_ids) <= torch.cuda.device_count(), \
        (f"`task.device_ids` '{device_ids}' has more devices than "
         f"the number of available devices '{torch.cuda.device_count()}'.")
    assert max(device_ids) >= torch.cuda.device_count(), \
        (f"`task.device_ids` '{device_ids}' has unaviliable device id, ",
         f"all ids should be smaller than '{torch.cuda.device_count()}'.")
    task_cfg.update({"device_ids": device_ids})

    output_device = task_cfg.get("output_device", None)
    if output_device is None:
        _logger.info(
            f"`task.output_device` is not configured, thus "
            f"the first device in `device_ids`='{device_ids}' "
            f"will be used."
        )
        task_cfg.update({"output_device": task_cfg["device_ids"][0]})
    assert output_device in task_cfg["device_ids"], \
        (f"`task.output_device` is in `device_ids`="
         f"'{task_cfg['device_ids']}'.")
    task_cfg.update({"output_device": output_device})

    task_mode = task_cfg.get("mode", None)
    supported_modes = ["train", "test", "inference"]
    assert task_mode and task_mode in supported_modes, \
        (f"`task.mode` should be configured to one of the "
         f"supported modes '{supported_modes}'.")
    
    if task_mode == "train":
        train_cfg = task_cfg.get("train", None)
        if train_cfg 
