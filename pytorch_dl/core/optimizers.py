# Author: Chang Liu
# Some codes are refered from https://github.com/facebookresearch/pycls


"""Optimizers."""


from torch.nn import Module, DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Tuple, Union


# def _unwrap_model(
#         model: Union[Module, DataParallel, DistributedDataParallel]
#     ) -> Module:
#     if isinstance(model, (DataParallel, DistributedDataParallel)):
#         return model.module
#     else:
#         return model


def build_optimizer_scheduler(
        model: Module,
        lr: int
    ) -> Tuple[SGD, CosineAnnealingLR]:
    optimizer = SGD(
        model.parameters(),
        lr,
        momentum=0.9,
        weight_decay=5e-4
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=50,
        eta_min=1e-6
    )
    return optimizer, scheduler
