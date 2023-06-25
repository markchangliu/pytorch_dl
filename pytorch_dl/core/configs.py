# Author: Chang Liu


"""Config."""


import copy
import toml
from torch.nn import Module
from typing import Dict, Any, Iterable


def _get_registry_element(
        prefix: str, 
        ele_type: str, 
        registry: Dict[str, Any]
    ) -> None:
    assert ele_type in registry.keys(), \
        (f"{prefix.capitalize()} is not one of the supported "
         f"types '{list(registry.keys())}'.")
    return registry[ele_type]


class ModelConfig(object):
    def __init__(self, model_cfg: Dict[str, Any]) -> None:
        self._cfg = copy.deepcopy(model_cfg)
    
    def build(self, model_registry: Dict[str, Module]) -> Module:
        model_type = self._cfg.pop("type")
        stem_cfg = self._cfg.pop("stem", None)
        body_cfg = self._cfg.pop("body", None)
        head_cfg = self._cfg.pop("head", None)
        model_cls = _get_registry_element(model_registry, model_type)
        param_dict = {}
        param_dict.update(self._cfg)
        if stem_cfg:
            param_dict.update({"stem_cfg": stem_cfg})
        if body_cfg:
            param_dict.update({"body_cfg": body_cfg})
        if head_cfg:
            param_dict.update({"head_cfg": head_cfg})
        model = model_cls(**param_dict)
        return model




class Config(object):
    def __init__(self, cfg_path: str) -> None:
        cfg = toml.load(cfg_path)
        pass