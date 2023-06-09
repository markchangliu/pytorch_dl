# Author: Chang Liu


"""Classification problem model evaluation metrics."""


import torch
from statistics import mean
from torch import Tensor
from torch.nn import Module
from typing import List, Union, Dict, Any, Tuple, Optional


def _check_y(
        y_pred: Union[List[Tensor], Tensor], 
        y_gt: Union[List[Tensor], Tensor],
    ) -> Tuple[Tensor, Tensor]:
        assert type(y_pred) is type(y_gt), \
            ("`y_pred` and `y_gt` must be in the same type.")
        
        if isinstance(y_pred, list):
            y_pred = torch.cat(y_pred)
            y_gt = torch.cat(y_gt)

        if len(y_pred.shape) > 1:
            assert len(y_pred) == 2, \
                (f"`y_pred`'s shape is {y_pred.shape}, which is neither "
                 "(N, ) nor (N, 1).")
            _, y_pred = torch.max(y_pred, dim=1, keepdim=False)
        
        if len(y_gt.shape) > 1:
            assert y_gt.shape[1] == 1, \
                (f"`y_gt`'s shape is {y_gt.shape}, which is neither " 
                 "(N, ) nor (N, 1).")
            _, y_gt = torch.max(y_gt, dim=1, keepdim=False)

        assert y_pred.numel() == y_gt.numel(), \
            (f"`y_pred` and `y_gt` have different number of elements,"
             f"`y_pred` has {y_pred.numel()} while `y_gt` has {y_gt.numel()}.")

        return y_pred, y_gt


def _get_tp_fp_fn(
        y_pred: Tensor,
        y_gt: Tensor,
        mode: str,
        labels: List[int],
        pos_label: int
    ) -> Tuple[int, int, int, Optional[Dict[int, List[int]]]]:
    assert len(y_pred.shape) == 1 and len(y_gt.shape) == 1, \
        (f"`y_pred` and `y_gt` must have a shape of (N, ), "
         f"but got {y_pred.shape} and {y_gt.shape}.")
    
    assert y_pred.numel() == y_gt.numel(), \
            (f"`y_pred` and `y_gt` have different number of elements,"
             f"`y_pred` has {y_pred.numel()} while `y_gt` has {y_gt.numel()}.")
    
    valid_modes = ["binary", "micro", "macro"]
    assert mode in valid_modes, \
        (f"`mode`={mode} is not one of {valid_modes}.")
    
    assert pos_label in labels, \
        (f"`pos_label`={pos_label} is not in `labels`={labels}.")
    
    if mode == "binary":
        tp = (y_pred == pos_label)[y_gt == pos_label].sum().item()
        fp = (y_pred == pos_label)[y_gt != pos_label].sum().item()
        fn = (y_pred != pos_label)[y_gt == pos_label].sum().item()
    else:
        cls_metric_dict = {}
        for l in labels:
            tp_l = (y_pred == l)[y_gt == l].sum().item()
            fp_l = (y_pred == l)[y_gt != l].sum().item()
            fn_l = (y_pred != l)[y_gt == l].sum().item()
            cls_metric_dict[l] == [tp_l, fp_l, fn_l]
    
    if mode == "micro":
        tp, fp, fn = 0, 0, 0
        for l, (tp_l, fp_l, fn_l) in cls_metric_dict.items():
            tp += tp_l
            fp += fp_l
            fn += fn_l
        return tp, fp, fn, cls_metric_dict
    elif mode == "macro":
        return -1, -1, -1, cls_metric_dict
    elif mode == "binary":
        return tp, fp, fn, None


class Accuracy(Module):
    def __init__(self) -> None:
        super(Accuracy, Module).__init__()
        
    
    def forward(
            self,
            y_pred: Union[List[Tensor], Tensor],
            y_gt: Union[List[Tensor], Tensor]
        ) -> float:
        y_pred, y_gt = _check_y(y_pred, y_gt)
        num_samples = y_pred[0]
        num_correctness = (y_pred == y_gt).sum()
        accuracy = num_correctness / num_samples
        return accuracy


class Precision(Module):
    def __init__(
            self, 
            mode: str, 
            labels: List[int],
            pos_label: Optional[int] = 1
        ) -> None:
        super(Precision, self).__init__()
        
        self.labels = labels
        
        assert pos_label in labels, \
            (f"`pos_label`={pos_label} is not in `labels`={labels}.")
        self.pos_label = pos_label

        valid_modes = ["binary", "micro", "macro"]
        assert mode in valid_modes, \
            (f"`mode`={mode} is not one of {valid_modes}.")
        self.mode = mode
    
    
    def forward(
            self, 
            y_pred: Union[List[Tensor], Tensor], 
            y_gt: Union[List[Tensor], Tensor]
        ) -> Tuple[float, Optional[Dict[int, float]]]:
        y_pred, y_gt = _check_y(y_pred, y_gt)
        tp, fp, fn, cls_metric_dict = _get_tp_fp_fn(
            y_pred,
            y_gt,
            self.mode,
            self.labels,
            self.pos_label,
        )

        if self.mode == "micro" or self.mode == "macro":
            cls_precisions = {}
            for l, (tp_l, fp_l, fn_l) in cls_metric_dict:
                 cls_precisions[l] = tp_l / (tp_l+ fp_l)

        if self.mode == "binary":
            precision = tp / (tp + fp)
            return precision, None
        elif self.mode == "micro":
            precision = tp / (tp + fp)
            return precision, cls_precisions
        elif self.mode == "macro":
            precision = mean(list(cls_precisions.values()))
            return precision, cls_precisions


class Recall(Module):
    def __init__(
            self,
            mode: str,
            labels: List[int],
            pos_label: int = 1
        ) -> None:
        super(Recall, self).__init__()
        self.labels = labels
        
        assert pos_label in labels, \
            (f"`pos_label`={pos_label} is not in `labels`={labels}.")
        self.pos_label = pos_label

        valid_modes = ["binary", "micro", "macro"]
        assert mode in valid_modes, \
            (f"`mode`={mode} is not one of {valid_modes}.")
        self.mode = mode

    
    def forward(
            self,
            y_pred: Union[List[Tensor], Tensor], 
            y_gt: Union[List[Tensor], Tensor]
        ) -> Tuple[float, Optional[Dict[int, float]]]:
        y_pred, y_gt = _check_y(y_pred, y_gt)

        tp, fp, fn, cls_metric_dict = _get_tp_fp_fn(
            y_pred,
            y_gt,
            self.mode,
            self.labels,
            self.pos_label,
        )

        if self.mode == "micro" or self.mode == "macro":
            cls_recalls = {}
            for l, (tp_l, fp_l, fn_l) in cls_metric_dict:
                 cls_recalls[l] = tp_l / (tp_l + fn_l)

        if self.mode == "binary":
            recall = tp / (tp + fn)
            return recall, None
        elif self.mode == "micro":
            recall = tp / (tp + fn)
            return recall, cls_recalls
        elif self.mode == "macro":
            recall = mean(list(cls_recalls.values()))
            return recall, cls_recalls
        

class CrossEntropyLoss(Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    
    def _param_check_forward(
            self, 
            param_dict: Dict[str, Any]
        ) -> Tuple[Tensor, Tensor]:
        logit_pred = param_dict["logit_pred"]
        y_gt = param_dict["y_gt"]
        assert type(logit_pred) is type(y_gt), \
            ("`logit_pred` and `y_gt` must be in the same type.")
        
        if isinstance(logit_pred, list):
            logit_pred = torch.cat(logit_pred)
            y_gt = torch.cat(y_gt)

        assert len(logit_pred.shape) == 2, \
            (f"`logit_pred` should shape of (N, C), but got "
             f"{logit_pred.shape} instead.")
        
        assert len(y_gt.shape) == 1 or (len(y_gt.shape) == 2 and y_gt.shape[1] == 1), \
            (f"`y_gt` should have a shape of either (N, 1) or "
             f"(N), but got {y_gt.shape}")
        
        assert logit_pred.shape[0] == y_gt.shape[0], \
            (f"`logit_pred` and `y_gt` should have the same number of samples, "
             f"but got {logit_pred.shape[0]} and {y_gt.shape[0]}.")
        
        y_gt = y_gt.contiguous().view(-1)
        
        return logit_pred, y_gt


    def forward(
            self, 
            logit_pred: Union[List[Tensor], Tensor], 
            y_gt: Union[List[Tensor], Tensor],
        ) -> None:
        param_dict = {
            "logit_pred": logit_pred,
            "y_gt": y_gt
        }
        logit_pred, y_gt = self._param_check_forward(param_dict)
        
        num_samples = y_gt.numel()
        max_logit, _ = torch.max(logit_pred, dim=1, keepdim=True)
        logit_pred -= max_logit
        exp_score_pred = torch.exp(logit_pred)
        prob_pred = exp_score_pred / torch.sum(exp_score_pred, dim=1, keepdim=True)
        prob_pred_at_label_idx = prob_pred[torch.arange(num_samples), y_gt]
        cross_entropy = torch.log(prob_pred_at_label_idx)
        loss = - torch.sum(cross_entropy, dim=0)
        loss /= num_samples
        return loss