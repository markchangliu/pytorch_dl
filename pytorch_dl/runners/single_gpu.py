# Author: Chang Liu


"""Single gpu runner."""


import time
import os
import torch
from statistics import mean
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from typing import Optional, Callable, Any

from pytorch_dl.core.logging import get_logger
from pytorch_dl.core.utils import save_checkpoint


_logger = get_logger(__name__)

__all__ = ["SingleGpuRunner"]

class SingleGpuRunner(object):
    def __init__(
            self, 
            num_epoches: int, 
            work_dir: Optional[str] = None,
            device_id: int = 0, 
            log_iter_interval: int = 5, 
            val_epoch_interval: int = 1, 
            checkpoint_epoch_interval: int = 5
        ) -> None:
        if work_dir is None:
            curr_timestamp = time.time()
            work_dir = os.path.join(os.getcwd(), f"{curr_timestamp:.0f}")

        available_devices = list(range(torch.cuda.device_count()))
        assert device_id in available_devices, \
            f"`device_id` is not in available devices {available_devices}."
        
        self.num_epoches = num_epoches
        self.work_dir = work_dir
        self.device_id = device_id
        self.log_iter_interval = log_iter_interval
        self.val_epoch_interval = val_epoch_interval
        self.checkpoint_epoch_interval = checkpoint_epoch_interval

        self._cfg = {
            "type": type(self).__name__,
            "num_epoches": num_epoches,
            "work_dir": work_dir,
            "device_id": device_id,
            "log_iter_interval": log_iter_interval,
            "val_epoch_interval": val_epoch_interval,
            "checkpoint_epoch_interval": checkpoint_epoch_interval
        }

    
    @torch.no_grad()
    def _test_one_epoch(
            self, 
            model: Module, 
            test_dataloader: DataLoader,
            loss_func: Callable[[Any, Any], Tensor],
            metric: object,
            mode: str,
        ) -> None:
        num_batches = len(test_dataloader)
        num_samples = 0
        total_num_samples = len(test_dataloader.dataset)

        metric.reset()
        model.eval()

        iter_losses = []
        for batch_idx, (X, y_gt) in enumerate(test_dataloader):
            X, y_gt = X.cuda(self.device_id), y_gt.cuda(self.device_id)
            num_samples += y_gt.shape[0]
            y_pred = model(X)
            iter_loss = loss_func(y_pred, y_gt)
            iter_losses.append(iter_loss.item())
            metric.update(y_pred, y_gt)
        
            if (batch_idx + 1) % self.log_iter_interval == 0:
                _logger.info(
                    f"{mode.capitalize()} epoch, "
                    f"iter {num_samples}/{total_num_samples}, "
                    f"loss {iter_loss.item():.4f}."
                )
        
        _logger.info(
            f"{mode.capitalize()} epoch, "
            f"evaluation result:\n{str(metric)}"
        )

        
    def _train_one_epoch(
            self,
            epoch_idx: int, 
            model: Module,
            train_dataloader: DataLoader,
            optimizer: Optimizer,
            scheduler: Scheduler,
            loss_func: Callable[[Any, Any], Tensor],
            metric: object,
        ) -> None:
        num_batches = len(train_dataloader)
        num_samples = 0
        total_num_samples = len(train_dataloader.dataset)

        metric.reset()
        model.train()

        iter_losses = []
        for batch_idx, (X, y_gt) in enumerate(train_dataloader):
            X, y_gt = X.cuda(self.device_id), y_gt.cuda(self.device_id)
            num_samples += y_gt.shape[0]
            y_pred = model(X)
            iter_loss = loss_func(y_pred, y_gt)
            iter_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            iter_losses.append(iter_loss.item())
            metric.update(y_pred, y_gt)
            if (batch_idx + 1) % self.log_iter_interval == 0:
                _logger.info(
                    f"Train epoch {epoch_idx + 1}/{self.num_epoches}, "
                    f"iter {num_samples}/{total_num_samples}, "
                    f"loss = {iter_loss.item():.4f}."
                )
        
        scheduler.step()

        _logger.info(
            f"Train epoch {epoch_idx + 1}/{self.num_epoches}, "
            f"loss = {mean(iter_losses):.4f}."
        )

    
    def train(
            self,
            model: Module,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader,
            optimizer: Optimizer,
            scheduler: Scheduler,
            loss_func: Callable[[Any, Any], Tensor],
            metric: object,
        ) -> None:
        _logger.info(f"Training process starts...")

        model = model.cuda(self.device_id)
        
        for epoch_idx in range(self.num_epoches):
            self._train_one_epoch(
                epoch_idx,
                model,
                train_dataloader,
                optimizer,
                scheduler,
                loss_func,
                metric
            )

            if (epoch_idx + 1) % self.val_epoch_interval == 0:
                self._test_one_epoch(model, val_dataloader, loss_func, metric,"val",)
            
            if (epoch_idx + 1) % self.checkpoint_epoch_interval == 0:
                checkpoint_path = os.path.join(
                    self.work_dir, 
                    f"epoch_{epoch_idx}.pth"
                )
                save_checkpoint(model, optimizer, scheduler, checkpoint_path)
        
        self._test_one_epoch(model, val_dataloader, loss_func, metric,"val",)
        save_checkpoint(model, optimizer, scheduler, os.path.join(self.work_dir, "epoch_last.pth"))
        _logger.info("Model training complete.")


    @torch.no_grad()
    def test(
            self,
            model: Module,
            test_dataloader: DataLoader,
            loss_func: Callable[[Any, Any], Tensor],
            metric: object,
        ) -> None:
        _logger.info(f"Testing process spthts...")

        model = model.cuda(self.device_id)
        model.eval()

        self._test_one_epoch(model, test_dataloader, loss_func, metric, "test",)
        
        _logger.info("Testing process completess.")