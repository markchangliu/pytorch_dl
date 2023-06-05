# Author: Chang Liu
# Some codes are refered from https://github.com/open-mmlab/mmcv/tree/1.x/mmcv/runner


"""Runner module."""


import torch
from statistics import mean
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Callable, Tuple, Union

from pytorch_dl.core.logging import get_logger


_logger = get_logger(__name__)


class EpochBasedRunner():
    def __init__(
            self,
            model: Module,
            optimizer: Optimizer,
            loss_func: Callable[..., Tensor],
            metric_funcs: Dict[str, Callable[..., float]],
            data_loaders: List[DataLoader],
            workflows: List[Tuple[str, int]]
        ) -> None:
        param_dict = {
            "model": model,
            "optimizer": optimizer,
            "loss_func": loss_func,
            "metric_funcs": metric_funcs,
            "data_loaders": data_loaders,
            "workflows": workflows
        }
        self._param_check(param_dict)


    def _param_check(self, param_dict: Dict[str, Any]) -> None:
        model = param_dict["model"]
        optimizer = param_dict["optimizer"]
        loss_func = param_dict["loss_func"]
        metric_funcs = param_dict["metric_funcs"]
        data_loaders = param_dict["data_loaders"]
        workflows = param_dict["workflows"]

        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func

        assert isinstance(metric_funcs, dict), \
            ("`eval_metrics` should be a dict.")
        self.metric_funcs = metric_funcs
        
        assert isinstance(data_loaders, type) and isinstance(workflows, list), \
            ("The types of `data_loaders` and `workflows` should be" 
             "list.")
        assert len(data_loaders) == len(workflows), \
            ("`data_loaders` and `workflows` should have the same length")
        self.data_loaders = data_loaders
        self.workflows = workflows

    
    def _train_one_epoch(
            self,
            data_loader: DataLoader,
        ) -> float:
        iter_losses = [],
        for batch_idx, (X, y) in enumerate(data_loader):
            y_pred = self.model(X)
            iter_loss = self.loss_func(y_pred, y)
            iter_losses.append(iter_loss.item())
            iter_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        epoch_loss = mean(iter_losses)
        return epoch_loss

    
    def train(
            self, 
            data_loader: DataLoader,
            num_epoch: int
        ) -> None:
        self.model.train()
        for i in range(num_epoch):
            epoch_loss = self._train_one_epoch(data_loader)
            _logger.info(
                f"Train epoch: {i + 1}/{num_epoch}, train loss: {epoch_loss:.2f}."
            )
    

    def _val_one_epoch(
            self, 
            data_loader: DataLoader
        ) -> Tuple[float, Dict[str, float]]:
        iter_losses = []
        iter_metrics = {k: [] for k, v in self.metric_funcs.items()}
        for batch_idx, (X, y) in enumerate(data_loader):
            y_pred = self.model(y)
            iter_loss = self.loss_func(y_pred, y)
            iter_losses.append(iter_loss.item())
            for k, metric_func in self.metric_funcs.items():
                iter_metrics[k].append(metric_func(y_pred, y))
        epoch_loss = mean(iter_losses)
        epoch_metrics = {k: 0 for k, v in self.metric_funcs.items()}
        for k in epoch_metrics.keys():
            epoch_metrics[k] = mean(iter_metrics[k])
        return epoch_loss, epoch_metrics

    
    @torch.no_grad()
    def val(
            self, 
            data_loader: DataLoader,
            num_epochs: int
        ) -> None:
        epoch_loss, epoch_metrics = self._val_one_epoch(data_loader)
        _logger.info(f"Val epoch, val loss: {epoch_loss:.2f}.")
        for k, epoch_metric in epoch_metrics.items():
            _logger.info(f"{k}: {epoch_metric:.2f}")

    
    @torch.no_grad()
    def inference(
            self,
            data_loader: DataLoader,
            num_epochs: int
        ) -> List[Any]:
        results = []
        for batch_idx, (X, y) in enumerate(data_loader):
            y_pred = self.model(X)
            results.append(y_pred)
        return results