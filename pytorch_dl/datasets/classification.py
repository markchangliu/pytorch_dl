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


class ImgFolderDataset(Dataset):
    def __init__(
            self, 
            img_dir: str,
            categories: List[str],
            transforms: Optional[List[Callable[..., Image]]] = None
        ) -> None:
        super(ImgFolderDataset, self).__init__()
        self._dataset = []
        self.transforms = transforms
        self._construct_ds(img_dir, categories)
    

    def _construct_ds(
            self, 
            img_dir: str,
            categories: List[str]
        ) -> None:
        categories.sort()
        sub_dirs = os.listdir(img_dir)
        sub_dirs.sort()
        io_class_names = []
        for sub_dir in sub_dirs:
            sub_dir_path = os.path.join(img_dir, sub_dir)
            if os.path.isdir(sub_dir_path):
                io_class_names.append(sub_dir)
        io_class_names.sort()
        assert set(categories) == set(io_class_names), ("The categories provided"
            "at config is not equivalent to the folder names under the img_dir '{0}'"
            .format(img_dir))
        self._cls_name_idx_dict = {cat: i for i, cat in enumerate(categories)}
        self._cls_idx_name_dict = {i: cat for i, cat in enumerate(categories)}

        for i, class_name in enumerate(categories):
            class_folder_path = os.path.join(img_dir, class_name)
            img_paths = gen_img_paths(class_folder_path)
            for img_path in img_paths:
                self._dataset.append((img_path, i))


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


class TrainTestImgFolderDataset(ImgFolderDataset):
    def __init__(
            self,
            img_dir: str,
            categories: List[str],
            transforms: Optional[List[Callable[..., Image]]] = None
        ) -> None:
        super(TrainTestImgFolderDataset, self).__init__(
            img_dir,
            categories,
            transforms
        )


class InferenceImgFolderDataset(ImgFolderDataset):
    def __init__(
            self, 
            img_dir: str, 
            categories: List[str],
            transforms: Optional[List[Callable[..., Image]]] = None
        ) -> None:
        super(InferenceImgFolderDataset, self).__init__(
            img_dir, categories, transforms
        )


    def _construct_ds(
            self, 
            img_dir: str, 
            categories: List[str]
        ) -> None:
        categories.sort()
        self._cls_name_idx_dict = {cat: i for i, cat in enumerate(categories)}
        self._cls_idx_name_dict = {i: cat for i, cat in enumerate(categories)}
        img_paths = gen_img_paths(img_dir)
        for img_path in img_paths:
            self._dataset.append((img_path, -1)) 