# Author: Chang Liu
# Some codes are refered from https://github.com/facebookresearch/pycls

"""Meters."""


from collections import deque
from statistics import mean
from time import time
from typing import List, Dict

from pytorch_dl.core.logging import get_logger


_logger = get_logger(__name__)


def _cvt_dict_to_str(d: Dict[str, float]) -> str:
    info = ""
    for k, v in d.items():
        info += f"{k} {v:.4f}, "
    info = info.strip()[:-1]
    return info


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


class ScalarMeter():
    def __init__(
            self, 
            window_size: int,
            metric_names: List[str]
        ) -> None:
        self._num_samples = 0

        self._records = {}
        for metric_name in metric_names:
            self._records[metric_name] = deque(maxlen=window_size)
        
        self._totals = {}
        for metric_name in metric_names:
            self._totals[metric_name] = 0.0


    def reset(self) -> None:
        self._num_samples = 0

        for metric_deque in self._records.values():
            metric_deque.clear()

        self._totals = {k: 0.0 for k in self._totals.keys()}
    

    def update_info(
            self, 
            num_samples: int,
            metric_dict: Dict[str, float]
        ) -> None:
        self._num_samples += num_samples

        for metric_name, metric_val in metric_dict.items():
            self._records[metric_name].append(metric_val)
            self._totals[metric_name] += metric_val * num_samples


    def get_global_avg(self) -> Dict[str, float]:
        info = {}
        for metric_name, metric_total in self._totals.items():
            info[metric_name] = metric_total / self._num_samples
        return info
    

    def get_win_avg(self) -> Dict[str, float]:
        info = {}
        for metric_name, metric_record in self._records.items():
            info[metric_name] = mean(metric_record)
        return info

    
    def log_global_avg(self) -> str:
        info = self.get_global_avg()
        info_str = _cvt_dict_to_str(info)
        return info_str


    def log_win_avg(self) -> str:
        info = self.get_win_avg()
        info_str = _cvt_dict_to_str(info)
        return info_str