# Author: Chang Liu


"""Task configuration module."""


import os
import torch
import toml
from pytorch_dl.core.logging import get_logger
from typing import Dict, Any


_VALID_TASKS = ["classification"]
_VALID_MODES = ["train", "test", "inference"]
_VALID_CLASSIFICATION_DATASET_TYPE = ["img_folder", "csv_annotation", "pickle"]


_logger = get_logger(__name__)


class Config():
    """Base config class.
    """
    def __init__(self, sys_cfg_path: str) -> None:
        """Args:
            sys_cfg_path (str): 
                System config path.
        
        Returns:
            None.
        """
        try:
            self._load_sys_cfgs(sys_cfg_path)
        except Exception as e:
            _logger.error(e)
            _logger.info("ConfigError encountered, runner \
                does not launch.")
            _logger.info("Main process exits with code 1.")


    def _load_sys_cfgs(self, sys_cfg_path: str) -> Dict[str, Any]:
        f"""Check and load system configs.

        It will check the following:
            * If `task`, `mode` are set in `sys_cfg_path`.
            * If `task` value is one of {_VALID_TASKS}.
            * If `mode` value is one of {_VALID_MODES}.
            * If `train_dataset_type`, `train_dataset_paths`, `val_dataset_type`,
            `val_dataset_paths` are set in `sys_cfg_path` for train mode.
            * If `test_dataset_type`, `test_dataset_paths` are set
            in `sys_cfg_path` for test mode.
            * If `inference_dataset_type` and `inference_dataset_paths` are set
            in `sys_cfg_path` for inference mode.
            * If GPU devices are available.
        
        At the end, it will load the following config as its attributes:
            * `self.task`.
            * `self.mode`.
            * `self.train_dataset_type`, `self.train_dataset_paths` if in train mode.
            * `self.test_dataset_type`, `self.test_dataset_paths` if in test mode.
            * `self.inference_dataset_type`, `self.inference_dataset_paths` 
            if in inference mode.
            * `self.random_seed`, default 42.
            * `self.num_gpus`, default 1.
            * `self.report_dir`, default `os.getcwd()/report`.
            * `self.num_epochs` if in train mode, default 10.

        Args:
            sys_cfg_path (str):
                System config path.
        
        Returns:
            cfg (Dict(strm Any)):
                Config dictionary.
        """
        cfg = toml.load(sys_cfg_path)
        assert "task" in cfg, "'task' is not set in config file."
        assert "mode" in cfg, "'mode' is not set in config file."
        assert cfg["task"] in _VALID_TASKS, \
            "'task={0}' is not a valid task, \
            valid tasks are {1}".format(cfg["task"], _VALID_TASKS)
        assert cfg["mode"] in _VALID_MODES, \
            "'mode={0}' is not a valid mode, \
            valid modes are {1}".format(cfg["mode"], _VALID_MODES)
        self.task = cfg["task"]
        self.mode = cfg["mode"]

        if self.mode == "train":
            assert "train_dataset_type" in cfg, \
                "'train_dataset_type' is not set in config file."
            assert "'train_dataset_paths'" in cfg, \
                "'train_dataset_paths' is not set in config file."
            assert "val_dataset_type" in cfg, \
                "'val_dataset_type' is not set in config file."
            assert "'val_dataset_paths'" in cfg, \
                "'val_dataset_paths' is not set in config file."
            self.train_dataset_type = cfg["train_dataset_type"]
            self.train_dataset_paths = cfg["train_dataset_paths"]
        elif self.mode == "test":
            assert "test_dataset_type" in cfg, \
                "'test_dataset_type' is not set in config file."
            self.train_dataset_type = cfg["test_dataset_type"]
            self.train_dataset_paths = cfg["test_dataset_paths"]
        elif self.mode == "inference":
            assert "inference_dataset_type" in cfg, \
                "'inference_dataset_type' is not set in config file."
            self.train_dataset_type = cfg["inference_dataset_type"]
            self.train_dataset_paths = cfg["inference_dataset_paths"]
        else:
            pass

        assert torch.cuda.is_available(), "No available GPU."
        
        self.random_seed = cfg.get("random_seed", 42)
        self.num_gpus = cfg.get("num_gpus", 1)
        default_report_dir = os.path.join(os.getcwd(), "report")
        self.report_dir = cfg.get("report_dir", default_report_dir)
        if self.mode == "train":
            self.num_epochs = cfg.get("num_epochs", 10)
        
        del cfg["mode"]
        del cfg["task"]
        if self.mode == "train":
            del cfg["train_dataset_paths"]
            del cfg["train_dataset_type"]
        elif self.mode == "test":
            del cfg["test_dataset_type"]
            del cfg["test_dataset_paths"]
        elif self.mode == "inference":
            del cfg["inference_dataset_type"]
            del cfg["inference_dataset_paths"]
        else:
            pass
        cfg.pop("random_seed", None)
        cfg.pop("num_gpus", None)
        cfg.pop("report_dir", None)
        cfg.pop("num_epochs", None)

        return cfg
    

class ClassificationTaskConfig(Config):
    """Classification task config class."""

    def __init__(self, sys_cfg_path: str) -> None:
        """Args:
            sys_cfg_path (str):
                system_config_path.
        Return:
            None.
        """
        try:
            cfg = super(
                ClassificationTaskConfig,
                self
            )._load_sys_cfgs(sys_cfg_path)
            cfg = self._load_classification_task_configs(cfg)
        except Exception as e:
            _logger.error(e)
            _logger.info("ConfigError encountered, runner \
                does not launch.")
            _logger.info("Main process exits with code 1.")

    
    def _load_classification_task_configs(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        f"""Check and load classification task configs.

        It will check the following:
            * If `backbone_cfg_path` and `num_classes` are set in config file.
            * If `train_dataset_type` value is one of
            {_VALID_CLASSIFICATION_DATASET_TYPE} for train mode.
            * If `test_dataset_type` value is one of 
            {_VALID_CLASSIFICATION_DATASET_TYPE} for test mode.
            * If `inference_dataset_type` value is one of
            {_VALID_CLASSIFICATION_DATASET_TYPE} for inference mode.
        
        At the end, it will load the following config as its attributes:
            * `self.backbone_cfg_path`
            * `self.neck_cfg_path`, default None
            * `self.head_cfg_path`, default None
            * `self.num_classes`

        """
        assert "backbone_cfg_path" in cfg, "'backbone_cfg_path' \
            is not set in config file."
        assert "num_classes" in cfg, "'num_classes' is not set in \
            config file."
        self.backbone_cfg_path = cfg["backbone_cfg_path"]
        self.num_classes = cfg["num_classes"]

        if self.mode == "train":
            assert self.train_dataset_type in _VALID_CLASSIFICATION_DATASET_TYPE, \
                f"'train_dataset_type'={0} is not a valid type, valid types \
                are {1}".format(self.train_dataset_type, _VALID_CLASSIFICATION_DATASET_TYPE)
        elif self.mode == "test":
            assert self.test_dataset_type in _VALID_CLASSIFICATION_DATASET_TYPE, \
                f"'test_dataset_type'={0} is not a valid type, valid types \
                are {1}".format(self.test_dataset_type, _VALID_CLASSIFICATION_DATASET_TYPE)
        elif self.mode == "inference":
            assert self.test_dataset_type in _VALID_CLASSIFICATION_DATASET_TYPE, \
                f"'test_dataset_type'={0} is not a valid type, valid types \
                are {1}".format(self.test_dataset_type, _VALID_CLASSIFICATION_DATASET_TYPE)
        
        self.neck_cfg_path = cfg.get("neck_cfg_path", None)
        self.head_cfg_path = cfg.get("head_cfg_path", None)

        del cfg["backbone_cfg_path"]
        del cfg["num_classes"]
        cfg.pop("neck_cfg_path", None)
        cfg.pop("head_cfg_path", None)

        return cfg