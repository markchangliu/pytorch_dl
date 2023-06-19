# Author: Chang Liu


"""Runners"""


import copy
from typing import Dict, Any

from pytorch_dl.runners.single_node_runner import (
    SingleGpuRunner,
    DataParallelRunner
)


__all__ = ["build_runner"]


def build_runner(runner_cfg: Dict[str, Any]) -> object:
    supported_runners = {
        "SingleGpuRunner": SingleGpuRunner,
        "DataParallelRunner": DataParallelRunner
    }
    runner_cfg = copy.deepcopy(runner_cfg)
    runner_type = runner_cfg.pop("type")
    assert runner_type in supported_runners.keys(), \
        (f"Runner type '{runner_type}' is not one of the "
         f"supported types '{list(supported_runners.keys())}'.")
    
    runner = supported_runners[runner_type](**runner_cfg)
    return runner