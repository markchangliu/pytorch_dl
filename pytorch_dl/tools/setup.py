# Author: Chang Liu


"""Setup a task."""


import copy
import os
import torch
import toml
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional

from pytorch_dl.runners import build_runner
from pytorch_dl.models import build_model
from pytorch_dl.data.dataloaders import build_dataloader
from pytorch_dl.core.optimizers import build_optimizer_scheduler
from pytorch_dl.core.logging import get_logger


_logger = get_logger(__name__)


_SetupReturnType = Tuple[object, Module, Optional[Optimizer], Optional[Scheduler],
                         Tuple[DataLoader, DataLoader], Dict[str, Any]]

_TASK_TYPES = ["classification"]
_MODES = ["train", "test", "inference"]


def setup_task(
        cfg_path: str
    ) -> _SetupReturnType:
    _logger.info(f"Loading config from '{cfg_path}'...")
    cfg = toml.load(cfg_path)

    task_cfg = cfg.get("task", None)
    assert task_cfg, f"`task` is not confugured."

    _logger.info(f"Seting up task...")
    task_type = task_cfg.get("type", None)
    assert task_type, \
        (f"`task.type` should be configured to one of the supported types "
         f"'{_TASK_TYPES}'. ")
    assert task_type in _TASK_TYPES, \
        (f"Task type '{task_type}' is not one of the supported types "
         f"'{_TASK_TYPES}'.")
    
    checkpoint_path = cfg.get("checkpoint_path", None)
    if checkpoint_path is not None:
        _logger.info(
            f"Model and/or optimizer will load from '{checkpoint_path}'."
        )
    else:
        cfg.update({"checkpoint_path": None})

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
        
        _logger.info(f"Building model...")
        model_cfg = cfg.pop("model", None)
        assert model_cfg, f"`model` is not configured."
        model_cfg.update({"num_classes": num_classes})
        model = build_model(task_type, model_cfg, checkpoint_path)
        _logger.info(f"Model building complete.")
        
    work_dir = task_cfg.get("work_dir", None)
    if work_dir is None:
        _logger.info(
            f"`task.work_dir` is not configured, thus `./tmp` "
            f"is used as the working directory."
        )
        task_cfg.update({"work_dir": os.getcwd()})
    
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
    assert max(device_ids) < torch.cuda.device_count(), \
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

    _logger.info(f"Building runner...")
    runner_type = task_cfg.get("runner_type", None)
    if runner_type is None:
        _logger.info(
            f"`task.runner_type` is not configured, "
            f"`SingleGpuRunner` is applied as default."
        )
        runner_type = "SingleGpuRunner"
        task_cfg.update({"runner_type": runner_type})
    runner_cfg = {
        "type": runner_type,
        "work_dir": task_cfg["work_dir"],
        "output_device": task_cfg["output_device"],
        "device_ids": task_cfg["device_ids"]
    }
    runner = build_runner(runner_cfg)
    _logger.info(f"Runner building complete.")

    task_mode = task_cfg.get("mode", None)
    assert task_mode and task_mode in _MODES, \
        (f"`task.mode` should be configured to one of the "
         f"supported modes '{_MODES}'.")
    
    if task_mode == "train":
        _logger.info(f"Loading training config...")
        train_cfg = task_cfg.get("train", None)
        assert train_cfg, f"`task.train` is not configured."
        
        num_epoches = train_cfg.get("num_epoches", None)
        assert num_epoches, f"`task.train.num_epoches` is not configured."

        meter_win_size = train_cfg.get("meter_win_size", None)
        if meter_win_size is None:
            _logger.info(
                f"`task.train.meter_win_size` is not configured, '20' will be used "
                f"as the default value."
            )
            train_cfg.update({"meter_win_size": 20})

        iter_log_interval = train_cfg.get("iter_log_interval", None)
        if iter_log_interval is None:
            _logger.info(
                f"`task.train.iter_log_interval` is not configured, '10' will be used "
                f"as the default value."
            )
            train_cfg.update({"iter_log_interval": 10})

        train_epoch_log_interval = train_cfg.get("train_epoch_log_interval", None)
        if train_epoch_log_interval is None:
            _logger.info(
                f"`task.train.train_epoch_log_interval` is not configured, '1' will be used "
                f"as the default value."
            )
            train_cfg.update({"train_epoch_log_interval": 1})

        val_interval = train_cfg.get("val_interval", None)
        if val_interval is None:
            _logger.info(
                f"`task.train.val_interval` is not configured, '1' will be used "
                f"as the default value."
            )
            train_cfg.update({"val_interval": 1})

        checkpoint_interval = train_cfg.get("checkpoint_interval", None)
        if checkpoint_interval is None:
            _logger.info(
                f"`task.train.checkpoint_interval` is not configured, '5' will be used "
                f"as the default value."
            )
            train_cfg.update({"checkpoint_interval": 5})

        _logger.info(f"Train config loading complete.")

        _logger.info(f"Building optimizer and scheduler.")
        optimizer_cfg = cfg.pop("optimizer", None)
        assert optimizer_cfg, f"`optimizer` is not configured."
        lr = optimizer_cfg.pop("lr", None)
        assert lr, f"`optimizer.lr` is not configured, '1e-5' is used as default."
        optimizer, scheduler = build_optimizer_scheduler(
            model,
            lr,
            checkpoint_path
        )
        _logger.info(f"Optimizer and scheduler building complete.")
        
        _logger.info(f"Building train dataloader...")
        train_dataloader_name = train_cfg.pop("train_dataloader", None)
        assert train_dataloader_name, \
            (f"`task.train.train_dataloader` is not configured.")
        train_dataloader_cfg = cfg.pop(train_dataloader_name, None)
        assert train_dataloader_cfg, \
            (f"The train dataloader cfg is not found under the "
             f"namespace `task.{train_dataloader_name}`. Please ensure that "
             f"it is configured and the name '{train_dataloader_name}' "
             f"is mentioned at `task.train.train_dataloader`.")
        if task_cfg["class_names"]:
            train_dataloader_cfg["dataset"].update({"class_names": class_names})
        train_dataloader = build_dataloader(task_type, train_dataloader_cfg)
        _logger.info(f"Train dataloader building complete...")

        _logger.info(f"Building val dataloader...")
        val_dataloader_name = train_cfg.pop("val_dataloader", None)
        assert val_dataloader_name, \
            (f"`task.train.val_dataloader` is not configured.")
        val_dataloader_cfg = cfg.pop(val_dataloader_name, None)
        assert val_dataloader_cfg, \
            (f"The val dataloader cfg is not found under the "
             f"namespace `task.{val_dataloader_name}`. Please ensure that "
             f"it is configured and the name '{val_dataloader_name}' "
             f"is mentioned at `task.train.val_dataloader`.")
        if task_cfg["class_names"]:
            val_dataloader_cfg["dataset"].update({"class_names": class_names})
        val_dataloader = build_dataloader(task_type, val_dataloader_cfg)
        _logger.info(f"Val dataloader building complete.")

        _logger.info(f"Train task setup complete.")

        return runner, model, optimizer, scheduler, \
            (train_dataloader, val_dataloader), train_cfg