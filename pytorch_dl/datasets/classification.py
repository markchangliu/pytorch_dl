# Author: Chang Liu
# Some codes are refered from https://github.com/facebookresearch/pycls


"""Classification datasets."""


import copy
import os
import PIL.Image as pil_image
from PIL.Image import Image
from pytorch_dl.core.io import gen_img_paths
from torch.utils.data import Dataset
from typing import Tuple, List, Callable, Optional, Dict


class ImgFolder(Dataset):
    def __init__(
            self, 
            img_dir: str,
            transforms: Optional[List[Callable[..., Image]]] = None
        ) -> None:
        super(ImgFolder, self).__init__()
        self._dataset = []
        self._cls_name_idx_dict = {}
        self._cls_idx_name_dict = {}
        self.transforms = transforms
        self._construct_ds(img_dir)
    

    def _construct_ds(
            self, 
            img_dir: str
        ) -> None:
        cls_idx = 0
        for dir_name in sorted(os.listdir(img_dir)):
            cls_name = dir_name
            cls_folder_path = os.path.join(img_dir, dir_name)
            if os.path.isdir(cls_folder_path):
                self._cls_idx_name_dict[cls_idx] = cls_name
                self._cls_name_idx_dict[cls_name] = cls_idx
                img_paths = gen_img_paths(cls_folder_path)
                for img_path in img_paths:
                    self._dataset.append((img_path, cls_idx))
                cls_idx += 1
            else:
                self._cls_idx_name_dict[-1] = "NA"
                self._cls_name_idx_dict["NA"] = -1
                img_path = os.path.join(img_dir, dir_name)
                self._dataset.append((img_path, -1))

    
    def __len__(self) -> int:
        return len(self._dataset)


    def __getitem__(
            self, 
            index: int
        ) -> Tuple[Image, int]:
        img_path, cls_idx = self._dataset[index]
        img = pil_image.open(img_path).convert("RGB")
        if self.transforms:
            for tf in self.transforms:
                img = tf(img)
        return img, cls_idx
    

    def get_cls_name_idx_dict(self) -> Dict[str, int]:
        return copy.deepcopy(self._cls_name_idx_dict)
    

    def get_cls_idx_name_dict(self) -> Dict[int, str]:
        return copy.deepcopy(self._cls_idx_name_dict)
