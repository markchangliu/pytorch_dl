# Author: Chang Liu


"""Utility functions."""


import copy
from torch.nn import Module
from typing import Dict, Any


def build_module_from_cfg(
        prefix: str,
        cfg: Dict[str, Any],
        registry: Dict[str, Module],
    ) -> Module:
    cfg = copy.deepcopy(cfg)
    module_type = cfg.pop("type")
    assert module_type in registry.keys(), \
        (f"{prefix.capitalize()} '{module_type}' is not one of the "
         f"supported types '{list(registry.keys())}'")
    module = registry[module_type](**cfg)
    return module