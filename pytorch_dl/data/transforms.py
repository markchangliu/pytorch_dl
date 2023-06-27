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


__all__ = ["ResizePad", "RandomCrop", "FixedCrop", "ToTensor"]


class Transform(nn.Module):
    def __init__(self):
        super(Transform, self).__init__()
        self._cfg = {}
    
    @property
    def cfg(self) -> Dict[str, Any]:
        return copy.deepcopy(self._cfg)


class ResizePad(Transform):
    def __init__(
            self,
            new_size: Optional[Tuple[int, int]] = None,
            new_ratio: Optional[Tuple[float, float]] = None
        ) -> None:
        super(ResizePad, self).__init__()
        assert new_size or new_ratio, (
            "You must provide one of `new_size` and `new_ratio`."
        )
        if new_size:
            self.new_size = new_size
            self.new_ratio = None
        elif new_ratio:
            self.new_size = None
            self.new_ratio = new_ratio
        
        self._cfg.update({
            "type": type(self).__name__,
            "new_size": self.new_size,
            "new_ratio": self.new_ratio
        })
    
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
    

class Pad(Transform):
    def __init__(self, pad_size: Tuple[int, int]):
        super(Pad, self).__init__()
        self.pad_size = pad_size

    def forward(self, img: Image) -> Image:
        w, h = img.size
        pad_w, pad_h = self.pad_size[0], self.pad_size[1]
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        img = F.pad(img, (pad_left, pad_top, pad_right, pad_bottom))
        return img
    

class RandomCrop(Transform):
    def __init__(
            self,
            crop_ratio: Optional[Tuple[float, float]] = None,
            output_size: Optional[Tuple[int, int]] = None,
        ) -> None:
        super(RandomCrop, self).__init__()
        assert crop_ratio or output_size, \
            f"You mush provide either `crop_ratio` or `output_size`."
        
        if crop_ratio:
            self.crop_ratio = crop_ratio
            self.output_size = None
        else:
            self.output_size = output_size
            self.crop_ratio = None

        self._cfg.update({
            "type": type(self).__name__,
            "crop_ratio": self.crop_ratio,
            "output_size": self.output_size
        })
    

    def forward(self, img: Image) -> Image:
        w, h = img.size
        if self.crop_ratio:
            w_crop_ratio, h_crop_ratio = self.crop_ratio[0], self.crop_ratio[1]
            new_x1 = randint(0, int(w * w_crop_ratio / 2))
            new_x2 = randint(int((1 - w_crop_ratio / 2) * w), w)
            new_y1 = randint(0, int(h * h_crop_ratio / 2))
            new_y2 = randint(int((1 - h_crop_ratio / 2) * h), h)
            w_new = new_x2 - new_x1
            h_new = new_y2 - new_y1
            return F.crop(img, new_y1, new_x1, h_new, w_new)
        else:
            output_w, output_h = self.output_size[0], self.output_size[1]
            new_x1 = randint(0, w - output_w)
            new_y1 = randint(0, h - output_h)
            return F.crop(img, new_y1, new_x1, output_h, output_w)
    

class FixedCrop(Transform):
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

        self._cfg.update({
            "type": type(self).__name__,
            "w_crop_ratio": self.w_crop_ratio,
            "h_crop_ratio": self.h_crop_ratio
        })
    

    def forward(self, img: Image) -> Image:
        w, h = img.size
        new_x1 = int(w * self.w_crop_ratio / 2)
        new_x2 = int(w * (1 - self.w_crop_ratio / 2))
        new_y1 = int(h * self.h_crop_ratio / 2)
        new_y2 = int(h * (1 - self.h_crop_ratio / 2))
        w_new = new_x2 - new_x1
        h_new = new_y2 - new_y1
        return F.crop(img, new_y1, new_x1, h_new, w_new)