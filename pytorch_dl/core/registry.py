# Author: Chang Liu
# Some codes are refered from https://github.com/open-mmlab/mmcv


"""Registry module."""


import torch.nn as nn
from typing import Callable, Any, Optional


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
            self,
            register_name: Optional[str] = None,
        ) -> Callable[..., Any]:
        
        def inner(target_callable):
            self.module_dict[register_name] = target_callable
            return target_callable
        
        return inner