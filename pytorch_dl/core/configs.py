# Author: Chang Liu


"""Config."""


import copy
import toml
from collections import OrderedDict
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from typing import Dict, Any, Iterable

from pytorch_dl.core.utils import build_module_from_cfg


class Config(object):
    def __init__(self) -> None:
        self._cfg: Dict[str, Any] = {}
        self._children: Dict[str, "Config"] = OrderedDict()
        self._parent: "Config" = None
    
    def _add_child(self, prefix: str, config: "Config") -> None:
        self._children.update({prefix: Config})
        config._parent = self

    def get_cfg(self) -> Dict[str, Any]:
        cfg = copy.deepcopy(self._cfg)
        if self._children:
            for prefix, child in self._children.items():
                cfg.update({prefix: child.get_cfg()})
        return cfg


class ModelConfig(Config):
    def __init__(self, model_cfg: Dict[str, Any]) -> None:
        super(ModelConfig, self).__init__()
        self._cfg = {
            "type": None,
            "stem": {},
            "body": {},
            "head": {},
        }
        self._cfg.update(model_cfg)
    
    def build(self, model_registry: Dict[str, Module]) -> Module:
        cfg = self.get_cfg()
        stem_cfg = cfg.pop("stem", None)
        body_cfg = cfg.pop("body", None)
        head_cfg = cfg.pop("head", None)
        param_dict = {}
        param_dict.update(cfg)
        if stem_cfg:
            param_dict.update({"stem_cfg": stem_cfg})
        if body_cfg:
            param_dict.update({"body_cfg": body_cfg})
        if head_cfg:
            param_dict.update({"head_cfg": head_cfg})
        model = build_module_from_cfg("model", cfg, model_registry)
        self._cfg = model.cfg
        return model
    

class TransformConfig(Config):
    def __init__(self, transform_cfgs: Iterable[Dict[str, Any]]) -> None:
        super(TransformConfig, self).__init__()
        self._cfg = [{"type": None} for _ in transform_cfgs]
        for i, cfg in enumerate(transform_cfgs):
            self._cfg[i].update(cfg)
    
    def build(self, transform_registry: Dict[str, Module]) -> Compose:
        transforms = []
        for cfg in self._cfg:
            transform = build_module_from_cfg(
                "transform", 
                cfg,
                transform_registry,
            )
            transforms.append(transform)
        
        for i, transform in enumerate(transforms):
            self._cfg[i].update(transform.cfg)
        
        return Compose(transforms)
        

class DatasetConfig(Config):
    def __init__(self, dataset_cfg: Dict[str, Any]) -> None:
        super(DatasetConfig, self).__init__()
        dataset_cfg = copy.deepcopy(dataset_cfg)
        transform_cfgs = dataset_cfg.pop("transform", None)
        if transform_cfgs:
            transform_config = TransformConfig(transform_cfgs)
            self._add_child("transform", transform_config)
        self._cfg.update(dataset_cfg)
    
    def build(
            self, 
            dataset_registry: Dict[str, Dataset], 
            transform_registry: Dict[str, Module]
        ) -> Dataset:
        if self._children:
            transform = self._children["transform"].build(transform_registry)
        else:
            transform = None
        param_dict = {}
        param_dict.update(self._cfg)
        param_dict.update({"transform": transform})
        dataset = build_module_from_cfg("dataset", param_dict, dataset_registry)
        self._cfg.update(dataset.cfg)
        return dataset


class DataloaderConfig(Config):
    def __init__(self, dataloader_cfg: Dict[str, Any]) -> None:
        super(DataloaderConfig, self).__init__()
        dataloader_cfg = copy.deepcopy(dataloader_cfg)
        dataset_cfg = dataloader_cfg.pop("dataset")
        dataset_config = DatasetConfig(dataset_cfg)
        self._add_child("dataset", dataset_config)
        self._cfg.update(dataloader_cfg)
    
    def build(
            self,
            dataset_registry: Dict[str, Dataset],
            transform_registry: Dict[str, Module]
        ) -> DataLoader:
        dataset = self._children["dataset"].build(dataset_registry, transform_registry)
        param_dict = {}
        param_dict.update(self._cfg)
        param_dict.update({"dataset": dataset})
        dataloader = DataLoader(**param_dict)
        return dataloader
    

class MetricConfig(Config):
    def __init__(self, metric_config: Dict[str, Any]) -> None:
        super(MetricConfig, self).__init__()
        self._cfg.update(metric_config)

    def build(self, metric_registry: Dict[str, Any]) -> object:
        metric = build_module_from_cfg("metric", self._cfg, metric_registry)
        return metric
    

class TrainConfig(Config):
    