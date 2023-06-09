# Author: Chang Liu
# Some codes are refered from https://github.com/open-mmlab/mmcv/tree/1.x/mmcv/runner


"""Runner module."""


import time
import torch
from statistics import mean
from torch import Tensor
from torch.nn import Module, DataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Callable, Tuple, Optional, Union

from pytorch_dl.core.logging import get_logger


_logger = get_logger(__name__)


class SingleNodeRunner():
    def __init__(
            self,
            model: Union[Module, DataParallel],
            optimizer: Optimizer,
            loss_func: Callable[..., Tensor],
            metric_funcs: Dict[str, Callable[..., float]],
            data_loaders: List[DataLoader],
            workflows: List[Tuple[str, int]],
            is_data_parallel: Optional[bool] = False,
            device_ids: Optional[List[int]] = None,
            output_device: Optional[int] = None,
            train_log_period: Optional[int] = 1
        ) -> None:
        param_dict = {
            "model": model,
            "optimizer": optimizer,
            "loss_func": loss_func,
            "metric_funcs": metric_funcs,
            "data_loaders": data_loaders,
            "workflows": workflows,
            "is_data_parallel": is_data_parallel,
            "device_ids": device_ids,
            "output_device": output_device,
            "train_log_period": train_log_period
        }
        
        self._param_check(param_dict)


    def _param_check(self, param_dict: Dict[str, Any]) -> None:
        model = param_dict["model"]
        optimizer = param_dict["optimizer"]
        loss_func = param_dict["loss_func"]
        metric_funcs = param_dict["metric_funcs"]
        data_loaders = param_dict["data_loaders"]
        workflows = param_dict["workflows"]
        is_data_parellel = param_dict["is_data_parallel"]
        device_ids = param_dict["device_ids"]
        output_device = param_dict["output_device"]
        train_log_period = param_dict["train_log_period"]

        assert torch.cuda.is_available(), \
            ("No cuda device is available")
        
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_log_period = train_log_period

        assert isinstance(metric_funcs, dict), \
            ("`metric_funcs` should be a dict.")
        self.metric_funcs = metric_funcs

        assert isinstance(data_loaders, list) and isinstance(workflows, list), \
            ("The types of `data_loaders` and `workflows` should be " 
             "list.")
        assert len(data_loaders) == len(workflows), \
            ("`data_loaders` and `workflows` should have the same length")
        self.data_loaders = data_loaders
        self.workflows = workflows

        self.is_data_parellel = False if torch.cuda.device_count() == 1 \
            else is_data_parellel

        if self.is_data_parellel:
            if device_ids:
                assert len(device_ids) <= torch.cuda.device_count(), \
                    ("The number of devices in `device_ids` is greater than the number of "
                    f"available device = {torch.cuda.device_count()}.")
            else:
                device_ids = list(range(torch.cuda.device_count()))
            self.device_ids = device_ids

            if output_device:
                assert output_device in device_ids, \
                    (f"Invalid `output_device`={output_device}, which is not in the "
                        f"`device_ids`={device_ids}.")
            else:
                output_device = device_ids[0]
            self.output_device = output_device

            assert isinstance(model, DataParallel), \
                ("`model` is not a `DataParallel` instance")
        else:
            self.device_ids = None
            self.output_device = 0


    def train(
            self,
            data_loader: DataLoader,
            num_epochs: int
        ) -> None:
        num_batches = len(data_loader)

        if self.is_data_parellel:
            model = self.model
        else:
            model = self.model.cuda(self.output_device)
        model.train()
        
        for i in range(num_epochs):
            epoch_loss = 0
            for batch_idx, (X, y) in enumerate(data_loader):
                if self.is_data_parellel:
                    y = y.cuda(self.output_device)
                else:
                    X, y = X.cuda(self.output_device), y.cuda(self.output_device)
                y_pred = model(X)
                iter_loss = self.loss_func(y_pred, y)
                iter_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                epoch_loss += iter_loss.item()
            epoch_loss /= num_batches
            if (i + 1) % self.train_log_period == 0:
                _logger.info(
                    f"Train epoch {i + 1}/{num_epochs}, loss {epoch_loss:.4f}."
                )
    

    @torch.no_grad()
    def val_or_test(
            self,
            data_loader: DataLoader,
            mode: str = "val"
        ) -> Tuple[float, Dict[str, float]]:
        num_batches = len(data_loader)

        if self.is_data_parellel:
            model = self.model
        else:
            model = self.model.cuda(self.output_device)
        model.eval()

        epoch_loss = 0
        epoch_metrics = {k: 0 for k in self.metric_funcs.keys()}
        for batch_idx, (X, y) in enumerate(data_loader):
            if self.is_data_parellel:
                y = y.cuda(self.output_device)
            else:
                X, y = X.cuda(self.output_device), y.cuda(self.output_device)
            y_pred = model(X)
            iter_loss = self.loss_func(y_pred, y)
            epoch_loss += iter_loss.item()
            for metric_name, metric_func in self.metric_funcs.items():
                epoch_metrics[metric_name] += metric_func(y_pred, y)
        epoch_loss /= num_batches
        epoch_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        _logger.info(
            f"{mode.capitalize()} epoch, loss {epoch_loss:.4f}."
        )
        for metric_name, metric_value in epoch_metrics.items():
            _logger.info(f"{metric_name} {metric_value:.4f}.")

        return epoch_loss, epoch_metrics
    

    @torch.no_grad()
    def inference(
            self,
            data_loader: DataLoader
        ) -> Tuple[float, List[Any]]:
        if self.is_data_parellel:
            model = self.model
        else:
            model = self.model.cuda(self.output_device)
        
        inference_times = []
        results = []
        for batch_idx, (X, y) in enumerate(data_loader):
            if self.is_data_parellel:
                y = y.cuda(self.output_device)
            else:
                X, y = X.cuda(self.output_device), y.cuda(self.output_device)
            t1 = time.time()
            y_pred = model(X)
            t2 = time.time()
            inference_times.append(t2 - t1)
            results.append(y_pred)
        avg_inference_time = mean(inference_times)
        return avg_inference_time, results
    