# Author: Chang Liu
# Some codes are refered from https://github.com/facebookresearch/pycls


"""Image transformation module.

All transformation operations' input and output are PIL Image object.
"""


import copy
import PIL.Image as pil_image
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from PIL.Image import Image
from random import randint
from torchvision.transforms import Compose, ToTensor
from typing import Optional, Tuple, List, Dict, Any


class ResizePad(nn.Module):
    def __init__(
            self,
            new_size: Optional[Tuple[int, int]] = None,
            new_ratio: Optional[Tuple[float, float]] = None
        ) -> None:
        super(ResizePad, self).__init__()
        assert new_size or new_ratio, (
            "You must provide one of `new_size` and `new_ratio`."
        )
        assert new_size and new_ratio, (
            "You cannot provide both of `new_size` and `new_ratio`."
        )
        if new_size:
            self.new_size = new_size
            self.new_ratio = None
        elif new_ratio:
            self.new_size = None
            self.new_ratio = new_ratio
    
    def forward(self, img: Image) -> Image:
        h, w = img.height, img.width
        if self.new_size:
            new_h, new_w = self.new_size
        elif self.new_ratio:
            new_h_ratio, new_w_ratio = self.new_ratio
            new_h, new_w = int(new_h_ratio * h), int(new_w_ratio * w)
        else:
            pass
        rsz_ratio = min(new_h / h, new_w / w)
        rsz_h = int(rsz_ratio * h)
        rsz_w = int(rsz_ratio * w)
        img = F.resize(img, (rsz_h, rsz_w))
        pad_left = (new_w - rsz_w) // 2
        pad_right = new_w - pad_left - rsz_w
        pad_top = (new_h - rsz_h) // 2
        pad_bottom = new_h - pad_top - rsz_h
        img = F.pad(img, (pad_left, pad_top, pad_right, pad_bottom))
        return img
    

class RandomCrop(nn.Module):
    def __init__(
            self,
            w_crop_ratio: float,
            h_crop_ratio: float
        ) -> None:
        super(RandomCrop, self).__init__()
        assert w_crop_ratio < 1 and h_crop_ratio < 1, (
            "`w_crop_ratio` and `h_crop_ratio` should be smaller than 1."
        )
        self.w_crop_ratio = w_crop_ratio
        self.h_crop_ratio = h_crop_ratio
    

    def forward(self, img: Image) -> Image:
        w, h = img.size
        new_x1 = randint(0, int(w * self.w_crop_ratio / 2))
        new_x2 = randint(int((1 - self.w_crop_ratio / 2) * w), w)
        new_y1 = randint(0, int(h * self.h_crop_ratio / 2))
        new_y2 = randint(int((1 - self.h_crop_ratio / 2) * h), h)
        w_new = new_x2 - new_x1
        h_new = new_y2 - new_y1
        return F.crop(img, new_y1, new_x1, h_new, w_new)
    

class FixedCrop(nn.Module):
    def __init__(
            self,
            w_crop_ratio: float,
            h_crop_ratio: float
        ):
        super(FixedCrop, self).__init__()
        assert w_crop_ratio < 1 and h_crop_ratio < 1, (
            "`w_crop_ratio` and `h_crop_ratio` should be smaller than 1."
        )
        self.w_crop_ratio = w_crop_ratio
        self.h_crop_ratio = h_crop_ratio
    

    def forward(self, img: Image) -> Image:
        w, h = img.size
        new_x1 = int(w * self.w_crop_ratio / 2)
        new_x2 = int(w * (1 - self.w_crop_ratio / 2))
        new_y1 = int(h * self.h_crop_ratio / 2)
        new_y2 = int(h * (1 - self.h_crop_ratio / 2))
        w_new = new_x2 - new_x1
        h_new = new_y2 - new_y1
        return F.crop(img, new_y1, new_x1, h_new, w_new)
    

def build_transforms(
        transform_cfgs: Dict[str, Any]
    ) -> Compose:
    supported_transforms = {
        "ResizePad": ResizePad,
        "RandomCrop": RandomCrop,
        "FixedCrop": FixedCrop,
        "ToTensor": ToTensor
    }
    transform_cfgs = copy.deepcopy(transform_cfgs)
    transform_types = transform_cfgs.pop("types")
    transforms = []
    for transform_type in transform_types:
        assert transform_type in supported_transforms.keys(), \
            (f"Transform type '{transform_type}' is not one of the "
             f"supported types {list(supported_transforms.keys())}.")
        transform_cfg = transform_cfgs.get(transform_type, None)
        if transform_cfg:
            transforms.append(
                supported_transforms[transform_type](**transform_cfg)
            )
        else:
            transforms.append(
                supported_transforms[transform_type]()
            )
    transforms = Compose(transforms)