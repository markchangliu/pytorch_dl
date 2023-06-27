# Author: Chang Liu


"""Classification problem model evaluation metrics."""


import copy
import torch
from statistics import mean
from textwrap import dedent
from torch import Tensor
from torch.nn import Module
from typing import Iterable, Union, Dict, Any, Tuple, Optional

from pytorch_dl.core.logging import get_logger


_logger = get_logger(__name__)

__all__ = ["ClassifierMetric"]


class ClassifierMetric(object):
    def __init__(
            self,
            label_names: Iterable[str],
            pos_label_name: Optional[str] = None,
            mode: str = "macro"
        ):
        super(ClassifierMetric, self).__init__()
        valid_modes = ["binary", "micro", "macro"]
        assert mode in valid_modes, \
            (f"`mode` '{mode}' is not one of {valid_modes}.")
        
        if pos_label_name is None:
            pos_label_name = label_names[0]
        else:
            assert pos_label_name in label_names, \
                (f"`pos_label_name` '{pos_label_name}' is not one of the labels "
                f"'{label_names}'.")
            
        if mode == "binary" and pos_label_name is None:
            _logger.info(
                f"`pos_label_name` is not set for binary mode, the first "
                f"element in `label_names`, '{label_names[0]}', is used as default."
            )
            pos_label_name = label_names[0]
        
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
            "label_names": self.label_names,
            "pos_label_name": self.pos_label_name,
            "mode": self.mode
        }

    
    def __repr__(self) -> str:
        total = self._record_dict["total"]
        accuracy, precision, recall = self.get_metrics()
        repr_str = f"Mode: {self.mode}\n"
        repr_str += (
            f"{'':<15s}|{'Count':<15s}|{'Accuracy':<15s}|{'Precision':<15s}|"
            f"{'Recall':<15s}\n"
        )
        if self.mode != "binary":
            for l in self.label_names:
                record_dict = self._record_dict[l]
                count = record_dict["count"]
                tp = record_dict["tp"]
                fp = record_dict["fp"]
                tn = record_dict["tn"]
                fn = record_dict["fn"]
                accuracy_l = (tp + tn) / (tp + fp + tn + fn)
                precision_l = tp / (tp + fp)
                recall_l = tp / (tp + fn)
                repr_str += (
                    f"{l.capitalize():<15s}|{count:<15d}|{accuracy_l:<15.4f}|"
                    f"{precision_l:<15.4f}|{recall_l:<15.4f}\n"
                )
        repr_str += (
            f"{'Overall':<15s}|{total:<15d}|{accuracy:<15.4f}|"
            f"{precision:<15.4f}|{recall:<15.4f}"
        )
        return repr_str

    
    @property
    def cfg(self) -> Dict[str, Any]:
        return copy.deepcopy(self._cfg)


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
    
    
    def reset(self) -> None:
        self._record_dict["total"] = 0
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
            accuracy = (tp + tn) / (tp + fp + tn + fn)
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
                accuracy += (tp_l + tn_l) / (tp_l + fp_l + tn_l + fn_l)
                precision += tp_l / (tp_l + fp_l)
                recall += tp_l / (tp_l + fn_l)
            accuracy /= len(self.label_ids)
            precision /= len(self.label_ids)
            recall /= len(self.label_ids)
        return accuracy, precision, recall