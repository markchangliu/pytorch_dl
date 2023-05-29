# Author: Chang Liu
# Some codes are refered from https://github.com/open-mmlab/mmcv


"""Registry module."""


import copy
import torch.nn as nn
from typing import Callable, Any, Optional, Dict


class Registry:
    def __init__(
            self,
            name: str,
        ) -> None:
        self.name = name
        self._module_dict = {}

    
    def __len__(self) -> int:
        return len(self.module_dict)
    

    def get_module_dict(
            self
        ) -> Dict[str, Callable[..., Any]]:
        return copy.deepcopy(self._module_dict)

    
    def register_module(
            self,
            register_name: Optional[str] = None,
        ) -> Callable[..., Any]:
        
        def inner(
                target_callable: Callable[..., Any]
            ) -> Callable[..., Any]:
            assert register_name not in self._module_dict, \
                "'register_name={0}' had been already registered." \
                .format(register_name)
            self._module_dict[register_name] = target_callable
            return target_callable
        
        return inner