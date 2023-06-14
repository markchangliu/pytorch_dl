# Author: Chang Liu
# Some codes are refered from https://github.com/open-mmlab/mmcv/tree/1.x/mmcv/runner


"""Runner module."""


import os
import time
import torch
from statistics import mean
from torch import Tensor
from torch.nn import Module, DataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Callable, Tuple, Optional, Union

from pytorch_dl.core.logging import get_logger
from pytorch_dl.core.meters import Timer, MetricMeter


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
            iter_log_interval: int = 1,
            train_epoch_log_interval: int = 1,
            meter_win_size: int = 20,
            validate_interval: int = 1,
            checkpoint_interval: int = 5,
            work_dir: str = os.getcwd()
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
            "iter_log_interval": iter_log_interval,
            "train_epoch_log_interval": train_epoch_log_interval,
            "meter_win_size": meter_win_size,
            "validate_interval": validate_interval,
            "checkpoint_interval": checkpoint_interval,
            "work_dir": work_dir
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
        iter_log_interval = param_dict["iter_log_interval"]
        train_epoch_log_interval = param_dict["train_epoch_log_interval"]
        meter_win_size = param_dict["meter_win_size"]
        validate_interval = param_dict["validate_interval"]
        checkpoint_interval = param_dict["checkpoint_interval"]
        work_dir = param_dict["work_dir"]

        assert torch.cuda.is_available(), \
            ("No cuda device is available.")
        
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.iter_log_interval = iter_log_interval
        self.train_epoch_log_interval = train_epoch_log_interval
        self.validate_interval = validate_interval
        self.checkpoint_interval = checkpoint_interval

        assert isinstance(metric_funcs, dict), \
            ("`metric_funcs` should have a format of "
             "`Dict[str, Callable[..., float]]`.")
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
            self.model = model
        else:
            self.device_ids = None
            self.output_device = 0
            self.model = model

        self.timer = Timer()
        self.train_metric_meters = MetricMeter(
            meter_win_size,
            ["loss"]
        )
        self.val_test_metric_meters = MetricMeter(
            meter_win_size,
            ["loss"] + list(self.metric_funcs.keys())
        )

        assert os.path.isdir(work_dir), \
            f"`work_dir`='{work_dir}' is not a directory."
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        self.work_dir = work_dir


    def train(
            self,
            data_loader: DataLoader,
            num_epochs: int
        ) -> None:
        num_batches = len(data_loader)

        self.model.train()

        for i in range(num_epochs):
            self.train_metric_meters.reset()
            for batch_idx, (X, y_gt) in enumerate(data_loader):
                if self.is_data_parellel:
                    y_gt = y_gt.cuda(self.output_device)
                    batch_size = y_gt.shape[0] * len(self.device_ids)
                else:
                    X, y_gt = X.cuda(self.output_device), y_gt.cuda(self.output_device)
                    batch_size = y_gt.shape[0]
                y_pred = self.model(X)
                iter_loss = self.loss_func(y_pred, y_gt)
                iter_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.train_metric_meters.update_info(
                    batch_size,
                    {"loss": iter_loss.item()}
                )
                if (batch_idx + 1) % self.iter_log_interval == 0:
                    iter_info_str = self.train_metric_meters.log_win_avg()
                    _logger.info(
                        f"Train epoch {i + 1}/{num_epochs}, "
                        f"iter {batch_idx + 1}/{num_batches}, "
                        f"{iter_info_str}."
                    )
            if (i + 1) % self.train_epoch_log_interval == 0:
                epoch_info_str = self.train_metric_meters.log_global_avg()
                _logger.info(
                    f"Train epoch {i + 1}/{num_epochs}, "
                    f"{epoch_info_str}"
                )
            

    @torch.no_grad()
    def val_or_test(
            self,
            data_loader: DataLoader,
            mode: str = "val"
        ) -> None:
        num_batches = len(data_loader)

        self.model.eval()

        self.val_test_metric_meters.reset()

        for batch_idx, (X, y_gt) in enumerate(data_loader):
            if self.is_data_parellel:
                y_gt = y_gt.cuda(self.output_device)
                batch_size = y_gt.shape[0] * len(self.device_ids)
            else:
                X, y_gt = X.cuda(self.output_device), y_gt.cuda(self.output_device)
                batch_size = y_gt.shape[0]
            y_pred = self.model(X)
            iter_loss = self.loss_func(y_pred, y_gt)
            iter_metrics = {k: f(y_pred, y_gt)[0] for k, f in self.metric_funcs.items()}
            iter_metrics.update({"loss": iter_loss.item()})
            self.val_test_metric_meters.update_info(
                batch_size,
                iter_metrics
            )
            if (batch_idx + 1) % self.iter_log_interval == 0:
                iter_info_str = self.val_test_metric_meters.log_win_avg()
                _logger.info(
                    f"{mode.capitalize()} epoch, iter {batch_idx}/{num_batches}, "
                    f"{iter_info_str}."
                )
        epoch_info_str = self.val_test_metric_meters.log_global_avg()
        _logger.info(
            f"{mode.capitalize()} epoch, {epoch_info_str}."
        )
    

    @torch.no_grad()
    def inference(
            self,
            data_loader: DataLoader
        ) -> Tuple[float, List[Any]]:
        inference_times = []
        results = []
        
        self.model.eval()
        
        for batch_idx, (X, y_gt) in enumerate(data_loader):
            if not self.is_data_parellel:
                X = X.cuda(self.output_device)
            t1 = time.time()
            y_pred = self.model(X)
            t2 = time.time()
            inference_times.append(t2 - t1)
            results.append(y_pred)
        avg_inference_time = mean(inference_times)
        return avg_inference_time, results
    

    def run_workflows(self) -> Any:
        if not self.is_data_parellel:
            self.model = self.model.cuda(self.output_device)

        for ((work, num_epoch), data_loader) in zip(
            self.workflows, 
            self.data_loaders
            ):
            if work == "train":
                self.train(data_loader, num_epoch)
            elif work == "val":
                self.val_or_test(data_loader, "val")
            elif work == "test":
                self.val_or_test(data_loader, "test")
            elif work == "inference":
                self.inference(data_loader)
        
        self.model = self.model.cpu()

    
    def _save_checkpoint(
            self,
            checkpoint_name: str
        ) -> None:
        assert checkpoint_name.endswith(".tar"), \
            (f"`checkpoint_name`='{checkpoint_name}' does "
             f"not ends with an extension of '.tar'.")
        if self.is_data_parellel:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        checkpoint = {
            "model_state": model_state_dict,
            "optimizer_state": self.optimizer.state_dict()
        }
        checkpoint_path = os.path.join(self.work_dir, checkpoint_name)
        torch.save(checkpoint, checkpoint_path)

    
    def _load_checkpoint(
            self,
            checkpoint_path: str
        ) -> None:
        assert os.path.exists(checkpoint_path), \
            f"`checkpoint_path`='{checkpoint_path}' does not exists."
        assert os.path.isfile(checkpoint_path), \
            f"`checkpoint_path`={checkpoint_path} is not a file."
        checkpoint = torch.load(checkpoint_path)
        if self.is_data_parellel:
            self.model.module.load_state_dict(
                checkpoint["model_state"],
                map_location=f"cuda:{self.output_device}"
            )
        else:
            self.model.load_state_dict(
                checkpoint["model_state"],
                map_loacation=f"cuda:{self.output_device}"
            )
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])