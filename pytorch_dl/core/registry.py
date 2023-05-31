# Author: Chang Liu
# Some codes are refered from https://github.com/open-mmlab/mmcv


"""Registry module."""


import copy
import torch.nn as nn
from typing import Callable, Any, Optional, Dict


class Registry:
    """Registry class maintains a dictionary called `module_dict` to 
    map str to classes or functions. 

    For example:
    >>> head_registry = Registry("head")
    >>> module_dict = head_registry.module_dict 
    >>> module_dict
    {
        "ResLinearHead": ResLinearHead,
        "ResConvHead": ResConvHead,
        "ResLightHead": ResLightHead
    }
    
    This allows the program to build model based on names
    provided by config files.
    """
    def __init__(
            self,
            name: str,
        ) -> None:
        """
        Args:
            name (str):
                The name of the registry.
        Returns:
            None.
        """
        self.name = name
        self._module_dict = {}

    
    def __len__(self) -> int:
        """Get the length of the registry's module_dict.

        Args:
            None.

        Returns:
            length (int):
                The length of the registry's module_dict.
        """
        return len(self.module_dict)
    

    def get_module_dict(
            self
        ) -> Dict[str, Callable[..., Any]]:
        """Returns a copy of the registry's module_dict.

        Args:
            None.

        Returns:
            module_dict (Dict[str, Callable[..., Any]]):
                The copy of `self.module_dict`.
        """
        return copy.deepcopy(self._module_dict)

    
    def register_module(
            self,
            register_name: Optional[str] = None,
        ) -> Callable[..., Any]:
        """
        Register a class or function with `register_name`, that is,
        add {register_name: class_or_function} to its `self.module_dict`.

        This method should be used as a decorator.

        For example:
        >>> head_registry = Registry("head")
        >>> @head_registry.register_module("ResLinearHead")
        >>> class ResLinearHead(nn.Module):
        >>>     pass
        >>> head_registry.get_module_dict()
        {"ResLinearHead": ResLinearHead}
        """
        def inner(
                target_callable: Callable[..., Any]
            ) -> Callable[..., Any]:
            assert register_name not in self._module_dict, \
                "'register_name={0}' had been already registered." \
                .format(register_name)
            self._module_dict[register_name] = target_callable
            return target_callable
        
        return inner