# Author: Chang Liu
# Some codes are refered from https://github.com/open-mmlab/mmcv


"""Registry module."""


import torch.nn as nn
from typing import Callable, Any


class Registry:
    def __init__(
            self,
            name: str,
        ) -> None:
        self.name = name
        self.module_dict = {}

    
    def __len__(self) -> int:
        return len(self.module_dict)
    
    
    def register_module(
            register_name: str,
            target_cls: object
        ) -> Callable[[Any], nn.Module]:
