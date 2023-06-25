# Author: Chang Liu
# Some codes are refered from https://github.com/open-mmlab/mmcv/tree/1.x/mmcv/runner


"""Single gpu runner module."""


import os
import time
import torch
from statistics import mean
from torch import Tensor
from torch.nn import Module, DataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from torch.utils.data import DataLoader
from typing import Optional, Callable, Tuple, Any, Dict, List, Union

from pytorch_dl.core.logging import get_logger
from pytorch_dl.core.meters import Timer, MetricMeter


_logger = get_logger(__name__)
    

class _SingleNodeRunner():
    def __init__(
            self,
            work_dir: Optional[str] = None,
            device_ids: Optional[List[int]] = None,
            output_device: Optional[int] = None,
        ) -> None:
        assert torch.cuda.is_available(), "No cuda device is available."
        
        if work_dir is None:
            work_dir = os.getcwd()
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        self.work_dir = work_dir

        if device_ids:
            assert set(device_ids) == set(range(torch.cuda.device_count())), \
                (f"`device_ids`={device_ids} is not consistent with "
                f"available devices {list(range(torch.cuda.device_count()))}")
        else:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids

        if output_device:
            assert output_device in self.device_ids, \
                f"`output_device` is not in available device ids {self.device_ids}."
        else:
            output_device = self.device_ids[0]
        self.output_device = output_device


    def _save_checkpoint(
            self,
            model: Module,
            optimizer: Optimizer,
            scheduler: Scheduler,
            checkpoint_path: str
        ) -> None:
        _logger.info(f"Saving checkpoint at '{checkpoint_path}'...")
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)
        _logger.info(f"Saving checkpoint completes.")

    
    def _train_one_epoch(
            self,
            model: Module,
            optimizer: Optimizer,
            scheduler: Scheduler,
            data_loader: DataLoader,
            loss_func: Callable[[Any, Any], Tensor],
            metric_funcs: Dict[str, Callable[[Any, Any], float]],
            metric_meter: MetricMeter,
            epoch_idx: Optional[int],
            num_epoches: Optional[int],
            iter_log_interval: int = 1,
            is_data_parallel: bool = False
        ) -> None:
        num_batches = len(data_loader)
        num_samples = 0
        total_num_samples = len(data_loader.dataset)

        metric_meter.reset()
        model.train()

        for batch_idx, (X, y_gt) in enumerate(data_loader):
            num_samples += y_gt.shape[0]
            if not is_data_parallel:
                X = X.cuda(self.output_device)
            y_gt = y_gt.cuda(self.output_device)
            y_pred = model(X)
            iter_loss = loss_func(y_pred, y_gt)
            iter_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            iter_metrics = {
                k: f(y_pred, y_gt)[0] for k, f in metric_funcs.items()
            }
            iter_metrics.update({"loss": iter_loss.item()})
            metric_meter.update_info(num_samples, iter_metrics)

            if (batch_idx + 1) % iter_log_interval == 0:
                iter_info = metric_meter.log_win_avg()
                _logger.info(
                    f"Train epoch {epoch_idx + 1}/{num_epoches}, "
                    f"iter {num_samples}/{total_num_samples}, "
                    f"{iter_info}."
                )
        
        scheduler.step()

    
    @torch.no_grad()
    def _test_one_epoch(
            self,
            model: Module,
            data_loader: DataLoader,
            loss_func: Callable[[Any, Any], Tensor],
            metric_funcs: Dict[str, Callable[[Any, Any], float]],
            metric_meter: MetricMeter,
            mode: str,
            iter_log_interval: int = 1,
            is_data_parallel: bool = False
        ) -> None:
        num_batches = len(data_loader)
        num_samples = 0
        total_num_samples = len(data_loader.dataset)

        metric_meter.reset()
        model.eval()

        for batch_idx, (X, y_gt) in enumerate(data_loader):
            num_samples += y_gt.shape[0]
            if not is_data_parallel:
                X = X.cuda(self.output_device)
            y_gt = y_gt.cuda(self.output_device)
            y_pred = model(X)
            iter_loss = loss_func(y_pred, y_gt)
            iter_metrics = {
                k: f(y_pred, y_gt)[0] for k, f in metric_funcs.items()
            }
            iter_metrics.update({"loss": iter_loss.item()})
            metric_meter.update_info(num_samples, iter_metrics)
        
            if (batch_idx + 1) % iter_log_interval == 0:
                iter_info = metric_meter.log_win_avg()
                _logger.info(
                    f"{mode.capitalize()} epoch, "
                    f"iter {num_samples}/{total_num_samples}, "
                    f"{iter_info}."
                )
        
        epoch_info = metric_meter.log_global_avg()
        _logger.info(
            f"{mode.capitalize()} epoch, "
            f"{epoch_info}."
        )


    def train(
            self,
            num_epoches: int,
            model: Union[Module, DataParallel],
            optimizer: Optimizer,
            scheduler: Scheduler,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            loss_func: Callable[[Any, Any], Tensor],
            metric_funcs: Dict[str, Callable[[Any, Any], float]],
            meter_win_size: int = 20,
            iter_log_interval: int = 1,
            train_epoch_log_interval: int = 1,
            val_interval: int = 1,
            checkpoint_interval: int = 5,
            is_data_parallel: bool = False
        ) -> None:
        _logger.info(f"Training process spthts...")

        num_batches = len(train_dataloader)
        total_num_samples = len(train_dataloader.dataset)
        train_metric_meter = MetricMeter(
            meter_win_size,
            ["loss"] + list(metric_funcs.keys())
        )
        val_metric_meter = MetricMeter(
            meter_win_size,
            ["loss"] + list(metric_funcs.keys())
        )

        for epoch_idx in range(num_epoches):
            self._train_one_epoch(
                model,
                optimizer,
                scheduler,
                train_dataloader,
                loss_func,
                metric_funcs,
                train_metric_meter,
                epoch_idx,
                num_epoches,
                iter_log_interval,
                is_data_parallel
            )
            
            if (epoch_idx + 1) % train_epoch_log_interval == 0:
                epoch_info = train_metric_meter.log_global_avg()
                _logger.info(
                    f"Train epoch {epoch_idx + 1}/{num_epoches}, "
                    f"{epoch_info}."
                )

            if (epoch_idx + 1) % val_interval == 0:
                self._test_one_epoch(
                    model,
                    val_dataloader,
                    loss_func,
                    metric_funcs,
                    val_metric_meter,
                    "val",
                    iter_log_interval,
                    is_data_parallel
                )

            if (epoch_idx + 1) % checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    self.work_dir, 
                    f"epoch_{epoch_idx}.pth"
                )
                self._save_checkpoint(
                    model.module if is_data_parallel else model,
                    optimizer,
                    f"epoch_{epoch_idx}.pth"
                )
        
        epoch_info = train_metric_meter.log_global_avg()
        _logger.info(
            f"Train epoch {epoch_idx + 1}/{num_epoches}, {epoch_info}."
        )

        self._test_one_epoch(
            model,
            val_dataloader,
            loss_func,
            metric_funcs,
            val_metric_meter,
            "val",
            iter_log_interval,
            is_data_parallel
        )

        self._save_checkpoint(
            model.module if is_data_parallel else model,
            optimizer,
            os.path.join(self.work_dir, f"epoch_last.pth")
        )

        _logger.info("Training process completess.")


    @torch.no_grad()
    def test(
            self,
            model: Module,
            data_loader: DataLoader,
            loss_func: Callable[[Any, Any], Tensor],
            metric_funcs: Dict[str, Callable[[Any, Any], float]],
            meter_win_size: int = 20,
            iter_log_interval: int = 1,
            is_data_parallel: bool = False
        ) -> None:
        _logger.info(f"Testing process spthts...")

        model.eval()

        test_metric_meter = MetricMeter(
            meter_win_size,
            ["loss"] + list(metric_funcs.keys())
        )

        self._test_one_epoch(
            model,
            data_loader,
            loss_func,
            metric_funcs,
            test_metric_meter,
            "test",
            iter_log_interval,
            is_data_parallel
        )
        
        _logger.info("Testing process completess.")



class SingleGpuRunner(_SingleNodeRunner):
    def __init__(
            self,
            work_dir: Optional[str] = None,
            device_ids: Optional[List[int]] = None,
            output_device: Optional[int] = None,
        ) -> None:
        _logger.info("SingleGpuRunner initialization starts...")
        super(SingleGpuRunner, self).__init__(
            work_dir,
            None,
            output_device
        )
        _logger.info(f"SingleGpuRunner initialization completes.")
    

    def train(
            self, 
            num_epoches: int, 
            model: Module, 
            optimizer: Optimizer, 
            scheduler: Scheduler,
            train_dataloader: DataLoader, 
            val_dataloader: DataLoader, 
            loss_func: Callable[[Any, Any], Tensor], 
            metric_funcs: Dict[str, Callable[[Any, Any], float]], 
            meter_win_size: int = 20, 
            iter_log_interval: int = 1, 
            train_epoch_log_interval: int = 1, 
            val_interval: int = 1, 
            checkpoint_interval: int = 5
        ) -> None:
        return super(SingleGpuRunner, self).train(
            num_epoches, 
            model, 
            optimizer, 
            scheduler,
            train_dataloader, 
            val_dataloader, 
            loss_func, 
            metric_funcs, 
            meter_win_size, 
            iter_log_interval, 
            train_epoch_log_interval, 
            val_interval, 
            checkpoint_interval, 
            False
        )
    

    @torch.no_grad()
    def test(
            self, 
            model: Module, 
            data_loader: DataLoader, 
            loss_func: Callable[[Any, Any], Tensor], 
            metric_funcs: Dict[str, Callable[[Any, Any], float]], 
            meter_win_size: int = 20, 
            iter_log_interval: int = 1
        ) -> None:
        return super(SingleGpuRunner, self).test(
            model, 
            data_loader, 
            loss_func, 
            metric_funcs, 
            meter_win_size, 
            iter_log_interval, 
            False
        )
    

class DataParallelRunner(_SingleNodeRunner):
    def __init__(
            self, 
            work_dir: Optional[str] = None, 
            device_ids: Optional[List[int]] = None, 
            output_device: Optional[int] = None
        ) -> None:
        _logger.info("DataParallelRunner initialization spthts...")
        super(DataParallelRunner, self).__init__(
            work_dir, 
            device_ids, 
            output_device
        )
        _logger.info(f"DataParallelRunner initialization completes.")

    
    def train(
            self, 
            num_epoches: int, 
            model: Module, 
            optimizer: Optimizer, 
            scheduler: Scheduler,
            train_dataloader: DataLoader, 
            val_dataloader: DataLoader, 
            loss_func: Callable[[Any, Any], Tensor], 
            metric_funcs: Dict[str, Callable[[Any, Any], float]], 
            meter_win_size: int = 20, 
            iter_log_interval: int = 1, 
            train_epoch_log_interval: int = 1, 
            val_interval: int = 1, 
            checkpoint_interval: int = 5
        ) -> None:
        return super(DataParallelRunner, self).train(
            num_epoches, 
            model, 
            optimizer, 
            scheduler,
            train_dataloader, 
            val_dataloader, 
            loss_func, 
            metric_funcs, 
            meter_win_size, 
            iter_log_interval, 
            train_epoch_log_interval, 
            val_interval, 
            checkpoint_interval, 
            True
        )
    

    @torch.no_grad()
    def test(
            self, 
            model: Module, 
            data_loader: DataLoader, 
            loss_func: Callable[[Any, Any], Tensor], 
            metric_funcs: Dict[str, Callable[[Any, Any], float]], 
            meter_win_size: int = 20, 
            iter_log_interval: int = 1
        ) -> None:
        return super(DataParallelRunner, self).test(
            model, 
            data_loader, 
            loss_func, 
            metric_funcs, 
            meter_win_size, 
            iter_log_interval, 
            True
        )