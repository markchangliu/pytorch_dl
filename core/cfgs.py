# Author: Chang Liu


"""Configuration module."""


import torch
import toml
from core.logging import get_logger


_VALID_TASKS = ["classification"]
_VALID_MODES = ["train", "test", "inference"]


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
            self._load_sys_cfg(sys_cfg_path)
        except Exception as e:
            _logger.error(e)
            _logger.info("ConfigError encountered, runner \
                does not launch.")
            _logger.info("Main process exits with code 1.")


    def _load_sys_cfg(self, sys_cfg_path: str) -> None:
        f"""Perform preliminary config check and load system configs.

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


        Args:
            sys_cfg_path (str):
                System config path.
        
        Returns:
            None.
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
            self.train_dataset_type = cfg["train_dataset_type"]
            self.train_dataset_paths = cfg["train_dataset_paths"]
        elif self.mode == "inference":
            assert "inference_dataset_type" in cfg, \
                "'inference_dataset_type' is not set in config file."
            self.train_dataset_type = cfg["train_dataset_type"]
            self.train_dataset_paths = cfg["train_dataset_paths"]
        else:
            pass

        assert torch.cuda.is_available(), "No available GPU."
    

class ClassificationTaskConfig(Config):
    """Classification task config class."""

    def __init__(self, sys_cfg_path: str) -> None:
        """Args:
            sys_cfg_path (str):
                system_config_path.
        Return:
            None.
        """
        super(ClassificationTaskConfig, self).__init__(sys_cfg_path)

    
    # def _load_
