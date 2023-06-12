# Author: Chang Liu
# Some codes are refered from https://github.com/facebookresearch/pycls

"""Meters."""


from collections import deque
from statistics import mean
from time import time
from typing import List, Dict

from pytorch_dl.core.logging import get_logger


_logger = get_logger(__name__)


class Timer():
    def __init__(
            self
        ) -> None:
        self.total_time = 0.0
        self.avg_time = 0.0
        self._num_calls = 0
        self._diff = 0.0
        self._start_time = 0.0
    

    def reset(self) -> None:
        self.total_time = 0.0
        self.avg_time = 0.0
        self._num_calls = 0
        self._diff = 0.0
        self._start_time = 0.0


    def tic(self) -> None:
        self._start_time = time()


    def toc(self) -> None:
        assert self._start_time is not None, \
            (f"You must call `self.tic` at least once before calling `self.toc`.")
        self._diff = time() - self._start_time
        self.total_time += self._diff
        self._num_calls += 1
        self.avg_time = self.total_time / self._num_calls

    
    def log_info(self) -> str:
        info = f"avg time {self.avg_time:.4f}s"
        return info


class MetricMeter():
    def __init__(
            self, 
            window_size: int,
            metric_names: List[str]
        ) -> None:
        self._num_calls = 0

        self._records = {}
        for metric_name in metric_names:
            self._records[metric_name] = deque(window_size)
        
        self._totals = {}
        for metric_name in metric_names:
            self._totals[metric_name] = 0.0


    def reset(self) -> None:
        self._num_calls = 0

        for metric_deque in self._records.values():
            metric_deque.clear()

        for metric_total in self._totals.values():
            metric_total = 0.0
    

    def update_info(
            self, 
            metric_dict: Dict[str, float]
        ) -> None:
        self._num_calls += 1

        for metric_name, metric_val in metric_dict.items():
            self._records[metric_name].append(metric_val)
            self._totals[metric_name] += metric_val

    
    def log_global_info(self) -> None:
        info = ""
        for metric_name, metric_val in self._totals.items():
            info += f"{metric_name} global avg {metric_val:.4f}, "
        info = info.strip()[:-1]
        return info


    def log_win_info(self) -> None:
        info = ""
        for metric_name, record in self._records.items():
            win_avg = mean(record)
            info += f"{metric_name} win avg {record:.4f}, "
        info = info.strip()[:-1]
        return info