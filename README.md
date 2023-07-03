# Paper Result Reproduction

## ResNet

### CIFAR-10

| Architecture | Official | My Actualization |
|--------------|----------|------------------|
| ResNet-20/18 | 91.66%   | 95.82%           |


Notably, the "ResNet-20" adopt by the official paper does not have the same architecture as the typical ResNet as what we commonly used today.

My Implementation details:

* Trained 50 epoches with mini-batch size 128 on a single GPU.

* SGD optimizer with lr = 0.1, momentum = 0.9, weight decay = 1e-4

* CosineAnnealingLR learning rate scheduler with T_max = 50, eta_min = 1e-6

* Cross entropy loss function.
