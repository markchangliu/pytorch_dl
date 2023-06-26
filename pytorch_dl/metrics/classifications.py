# Author: Chang Liu


"""Classification problem model evaluation metrics."""


import torch
from statistics import mean
from textwrap import dedent
from torch import Tensor
from torch.nn import Module
from typing import Iterable, Union, Dict, Any, Tuple, Optional


class ClassifierMetric(object):
    def __init__(
            self,
            mode: str,
            label_names: Iterable[str],
            pos_label_name: Optional[int] = None,
        ):
        super(ClassifierMetric, self).__init__()
        valid_modes = ["binary", "micro", "macro"]
        assert mode in valid_modes, \
            (f"`mode` '{mode}' is not one of {valid_modes}.")
        assert pos_label_name in label_names, \
            (f"`pos_label_name` '{pos_label_name}' is not one of the labels "
             f"'{label_names}'.")
        assert mode == "binary" and pos_label_name is None, \
            (f"`pos_label_name` should not be None when `mode` is 'binary'.")
        
        self.label_names = label_names
        self.pos_label_name = pos_label_name
        self.label_ids = list(range(len(self.label_names)))
        self.pos_label_id = self.label_names.index(self.pos_label_name)
        self.mode = mode

        self._record_dict = {"total": 0}
        if self.mode == "binary":
            self._record_dict.update(
                {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
            )
        else:
            for i in self.label_ids:
                self._record_dict.update(
                    {
                        self.label_names[i]: {
                            "count": 0, 
                            "tp": 0, 
                            "fp": 0, 
                            "tn": 0, 
                            "fn": 0
                        }
                    }
                )

        self._cfg = {
            "type": type(self).__name__,
            "label_names": self.label_names,
            "pos_label_name": self.pos_label_name,
            "mode": mode
        }

    
    def __repr__(self) -> str:
        total = self._record_dict["total"]
        accuracy, precision, recall = self.get_metrics()
        repr_str = f"Mode: {self.mode}\n"
        repr_str += (
            f"{'':^10s}|{'Count':^10s}|{'Accuracy':^10s}|{'Precision':^10s}|"
            f"{'Recall':^10s}\n"
        )
        if self.mode != "binary":
            for l in self.label_names:
                record_dict = self._record_dict[l]
                count = record_dict["count"]
                tp = record_dict["tp"]
                fp = record_dict["fp"]
                tn = record_dict["tn"]
                fn = record_dict["fn"]
                accuracy_l = (tp + tn) / count
                precision_l = tp / (tp + fp)
                recall_l = tp / (tp + fn)
                repr_str += (
                    f"{l.capitalize():<10s}|{count:^10d}|{accuracy_l:^10.4f}|"
                    f"{precision_l:^10.4f}|{recall_l:^10.4f}\n"
                )
        repr_str += (
            f"{'Overall':<10s}|{total:^10s}|{accuracy:^10.4f}"
            f"{precision:^10.4f}|{recall:^10.4f}"
        )
        return repr_str

    
    def update(
            self, 
            y_pred: Tensor, 
            y_gt: Tensor
        ) -> None:
        if len(y_pred.shape) > 1:
            assert len(y_pred.shape) == 2, \
                (f"`y_pred`'s shape is {y_pred.shape}, which is neither "
                 "(N, ) nor (N, C).")
            _, y_pred = torch.max(y_pred, dim=1, keepdim=False)
        
        if len(y_gt.shape) > 1:
            assert y_gt.shape[1] == 1, \
                (f"`y_gt`'s shape is {y_gt.shape}, which is neither " 
                 "(N, ) nor (N, 1).")
            _, y_gt = torch.max(y_gt, dim=1, keepdim=False)

        assert y_pred.numel() == y_gt.numel(), \
            (f"`y_pred` and `y_gt` have different number of elements,"
             f"`y_pred` has {y_pred.numel()} while `y_gt` has {y_gt.numel()}.")

        self._record_dict["total"] += y_pred.shape[0]
        if self.mode == "binary":
            self._record_dict["tp"] += \
                (y_gt == self.pos_label_id)[y_pred == self.pos_label_id].sum().item()
            self._record_dict["fp"] += \
                (y_gt != self.pos_label_id)[y_pred == self.pos_label_id].sum().item()
            self._record_dict["tn"] += \
                (y_gt != self.pos_label_id)[y_pred != self.pos_label_id].sum().item()
            self._record_dict["fn"] += \
                (y_gt == self.pos_label_id)[y_pred != self.pos_label_id].sum().item()
        else:
            for l in self.label_ids:
                label_name = self.label_names[l]
                record_dict = self._record_dict[label_name]
                record_dict["count"] += (y_gt == l).sum().item()
                record_dict["tp"] += (y_gt == l)[y_pred == l].sum().item()
                record_dict["fp"] += (y_gt != l)[y_pred == l].sum().item()
                record_dict["tn"] += (y_gt != l)[y_pred != l].sum().item()
                record_dict["fn"] += (y_gt == l)[y_pred != l].sum().item()
    
    def get_metrics(self) -> Tuple[float, float, float]:
        if self.mode == "binary":
            tp = self._record_dict["tp"]
            fp = self._record_dict["fp"]
            tn = self._record_dict["tn"]
            fn = self._record_dict["fn"]
            total = self._record_dict["total_count"]
            accuracy = (tp + tn) / total
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
        elif self.mode == "micro":
            total = self._record_dict["total"]
            tp, fp, tn, fn = 0, 0, 0, 0
            for l in self.label_names:
                record_dict = self._record_dict[l]
                tp += record_dict["tp"]
                fp += record_dict["fp"]
                tn += record_dict["tn"]
                fn += record_dict["fn"]
            accuracy = (tp + tn) / total
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
        elif self.mode == "macro":
            total = self._record_dict["total"]
            accuracy, precision, recall = 0, 0, 0
            for l in self.label_names:
                record_dict = self._record_dict[l]
                count_l = record_dict["count"]
                tp_l = record_dict["tp"]
                fp_l = record_dict["fp"]
                tn_l = record_dict["tn"]
                fn_l = record_dict["fn"]
                accuracy += (tp_l + tn_l) / count_l
                precision += tp_l / (tp_l + fp_l)
                recall += tp_l / (tp_l + fn_l)
            accuracy /= len(self.label_ids)
            precision /= len(self.label_ids)
            recall /= len(self.label_ids)
        return accuracy, precision, recall
    

def _check_y(
        y_pred: Union[Iterable[Tensor], Tensor], 
        y_gt: Union[Iterable[Tensor], Tensor],
    ) -> Tuple[Tensor, Tensor]:
        assert type(y_pred) is type(y_gt), \
            ("`y_pred` and `y_gt` must be in the same type.")
        
        if isinstance(y_pred, Iterable):
            y_pred = torch.cat(y_pred)
            y_gt = torch.cat(y_gt)

        if len(y_pred.shape) > 1:
            assert len(y_pred.shape) == 2, \
                (f"`y_pred`'s shape is {y_pred.shape}, which is neither "
                 "(N, ) nor (N, C).")
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
        labels: Iterable[int],
        pos_label: int
    ) -> Tuple[int, int, int, Optional[Dict[int, Iterable[int]]]]:
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
            cls_metric_dict[l] = [tp_l, fp_l, fn_l]
    
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


############### Evaluation metrics ###############

class Accuracy(Module):
    def __init__(self) -> None:
        super(Accuracy, self).__init__()
        
    
    def forward(
            self,
            y_pred: Union[Iterable[Tensor], Tensor],
            y_gt: Union[Iterable[Tensor], Tensor]
        ) -> Tuple[float, None]:
        y_pred, y_gt = _check_y(y_pred, y_gt)
        num_samples = y_pred.shape[0]
        num_correctness = (y_pred == y_gt).sum().item()
        accuracy = num_correctness / num_samples
        return accuracy, None


class Precision(Module):
    def __init__(
            self, 
            mode: str, 
            labels: Iterable[int],
            pos_label: Optional[int] = 1
        ) -> None:
        super(Precision, self).__init__()
        
        self.label_ids = labels
        
        assert pos_label in labels, \
            (f"`pos_label`={pos_label} is not in `labels`={labels}.")
        self.pos_label = pos_label

        valid_modes = ["binary", "micro", "macro"]
        assert mode in valid_modes, \
            (f"`mode`={mode} is not one of {valid_modes}.")
        self.mode = mode
    
    
    def forward(
            self, 
            y_pred: Union[Iterable[Tensor], Tensor], 
            y_gt: Union[Iterable[Tensor], Tensor]
        ) -> Tuple[float, Optional[Dict[int, float]]]:
        y_pred, y_gt = _check_y(y_pred, y_gt)
        tp, fp, fn, cls_metric_dict = _get_tp_fp_fn(
            y_pred,
            y_gt,
            self.mode,
            self.label_ids,
            self.pos_label,
        )
        if self.mode == "micro" or self.mode == "macro":
            cls_precisions = {}
            for l, (tp_l, fp_l, fn_l) in cls_metric_dict.items():
                 cls_precisions[l] = tp_l / (tp_l + fp_l + 1e-5)

        if self.mode == "binary":
            precision = tp / (tp + fp)
            return precision, None
        elif self.mode == "micro":
            precision = tp / (tp + fp)
            return precision, cls_precisions
        elif self.mode == "macro":
            precision = mean(Iterable(cls_precisions.values()))
            return precision, cls_precisions


class Recall(Module):
    def __init__(
            self,
            mode: str,
            labels: Iterable[int],
            pos_label: int = 1
        ) -> None:
        super(Recall, self).__init__()
        self.label_ids = labels
        
        assert pos_label in labels, \
            (f"`pos_label`={pos_label} is not in `labels`={labels}.")
        self.pos_label = pos_label

        valid_modes = ["binary", "micro", "macro"]
        assert mode in valid_modes, \
            (f"`mode`={mode} is not one of {valid_modes}.")
        self.mode = mode

    
    def forward(
            self,
            y_pred: Union[Iterable[Tensor], Tensor], 
            y_gt: Union[Iterable[Tensor], Tensor]
        ) -> Tuple[float, Optional[Dict[int, float]]]:
        y_pred, y_gt = _check_y(y_pred, y_gt)

        tp, fp, fn, cls_metric_dict = _get_tp_fp_fn(
            y_pred,
            y_gt,
            self.mode,
            self.label_ids,
            self.pos_label,
        )

        if self.mode == "micro" or self.mode == "macro":
            cls_recalls = {}
            for l, (tp_l, fp_l, fn_l) in cls_metric_dict.items():
                 cls_recalls[l] = tp_l / (tp_l + fn_l + 1e-5)

        if self.mode == "binary":
            recall = tp / (tp + fn)
            return recall, None
        elif self.mode == "micro":
            recall = tp / (tp + fn)
            return recall, cls_recalls
        elif self.mode == "macro":
            recall = mean(Iterable(cls_recalls.values()))
            return recall, cls_recalls
        

############### Loss functions ###############

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
        
        if isinstance(logit_pred, Iterable):
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
            logit_pred: Union[Iterable[Tensor], Tensor], 
            y_gt: Union[Iterable[Tensor], Tensor],
        ) -> Tensor:
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