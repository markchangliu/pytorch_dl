# Author: Chang Liu
# Some codes are refered from https://github.com/facebookresearch/pycls


"""Classification datasets.

All datasets support two loading libraries -- opencv and PIL.

If the loading method is PIL, data output format is a PIL Image object
with "RGB" mode.

If the loading method is opencv, data output format is HWC/RGB/np.int8
numpy.ndarray in the range [0, 255].
"""


import copy
import numpy as np
import os
import PIL.Image as pil_image
from PIL.Image import Image
from pytorch_dl.core.io import gen_img_paths, gen_pickle_data
from torch.utils.data import Dataset
from typing import Tuple, List, Callable, Optional, Dict


############## Image folder dataset ##############

class ImgFolderDataset(Dataset):
    def __init__(
            self, 
            img_dir: str,
            class_names: List[str],
        ) -> None:
        super(ImgFolderDataset, self).__init__()
        self._dataset = []
        self._construct_ds(img_dir, class_names)
    

    def _construct_ds(
            self, 
            img_dir: str,
            class_names: List[str]
        ) -> None:
        class_names.sort()
        sub_dirs = os.listdir(img_dir)
        sub_dirs.sort()
        io_class_names = []
        for sub_dir in sub_dirs:
            sub_dir_path = os.path.join(img_dir, sub_dir)
            if os.path.isdir(sub_dir_path):
                io_class_names.append(sub_dir)
        io_class_names.sort()
        assert set(class_names) == set(io_class_names), ("The class_names provided"
            "at config is not equivalent to the folder names under the 'img_dir={0}'"
            .format(img_dir))
        self._cls_name_idx_dict = {cat: i for i, cat in enumerate(class_names)}
        self._cls_idx_name_dict = {i: cat for i, cat in enumerate(class_names)}

        for i, class_name in enumerate(class_names):
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
        return img, cls_idx
    

    def get_cls_name_idx_dict(self) -> Dict[str, int]:
        return copy.deepcopy(self._cls_name_idx_dict)
    

    def get_cls_idx_name_dict(self) -> Dict[int, str]:
        return copy.deepcopy(self._cls_idx_name_dict)


class TrainTestImgFolderDataset(ImgFolderDataset):
    pass


class InferenceImgFolderDataset(ImgFolderDataset):

    def _construct_ds(
            self, 
            img_dir: str, 
            class_names: List[str]
        ) -> None:
        class_names.sort()
        self._cls_name_idx_dict = {
            class_name: i for i, class_name in enumerate(class_names)
        }
        self._cls_idx_name_dict = {
            i: class_name for i, class_name in enumerate(class_names)
        }
        img_paths = gen_img_paths(img_dir)
        for img_path in img_paths:
            self._dataset.append((img_path, -1))


############## Pickle file dataset ##############


class PickleDataset(Dataset):
    def __init__(
            self,
            data_pickle_paths: List[str],
            class_names: List[str],
            img_size: Tuple[int, int]
        ) -> None:
        self.img_size = img_size
        self._construct_ds(
            data_pickle_paths,
            class_names,
            img_size
        )
        super(PickleDataset, self).__init__()
        
    
    def _construct_ds(
            self,
            data_pickle_paths: List[str],
            class_names: List[str],
            img_size: Tuple[int, int]
        ) -> None:
        data = []
        labels = []
        data_dicts = gen_pickle_data(data_pickle_paths, ["data", "labels"])
        for data, label in data_dicts:
            data.append(data)
            labels.extend(label)
        data = np.vstack(data)
        data_img_size = np.prod(data[0].shape)
        assert data_img_size == np.prod(img_size) * 3, (
            "The size of image loaded from pickle is not equal"
            "to the provided 'img_size={0}'".format(img_size)
        )
        data = data.reshape(
            (-1, 3, img_size[0], img_size[1])
        )
        # Transpose to HWC to support subsequent PIL Image opening
        data = data.transpose((0, 2, 3, 1))
        assert len(labels) == data.shape[0], (
            "The number of labels and image arraies are different."
        )
        self._data = np.ascontiguousarray(data)
        self._labels = labels

        class_names.sort()
        assert len(set(class_names)) == len(set(labels)), (
            "The number of class names in meta pickle file is different"
            "from the number of labels in data pickle files."
        )
        self._cls_name_idx_dict = {
            class_name: i for i, class_name in enumerate(class_names)
        }
        self._cls_idx_name_dict = {
            i: class_name for i, class_name in enumerate(class_names)
        }

    
    def __len__(self) -> int:
        return len(self._labels)
    

    def __getitem__(
            self, 
            index
        ) -> Tuple[Image, int]:
        img_arr = self._data[index]
        img = pil_image.fromarray(img_arr, mode="RGB")
        class_idx = self._labels[index]
        return img, class_idx
    

    def get_cls_name_idx_dict(self) -> Dict[str, int]:
        return copy.deepcopy(self._cls_name_idx_dict)
    

    def get_cls_idx_name_dict(self) -> Dict[int, str]:
        return copy.deepcopy(self._cls_idx_name_dict)
    

class TrainTestPickleDataset(PickleDataset):
    pass


class InferencePickleDataset(PickleDataset):
    def _construct_ds(
            self, 
            data_pickle_paths: List[str], 
            class_names: List[str], 
            img_size: Tuple[int, int]
        ) -> Tuple[Image, int]:
        data_dicts = gen_pickle_data(data_pickle_paths, ["data"])
        for data, label in data_dicts:
            data.append(data)
        data = np.vstack(data)
        data_img_size = np.prod(data[0].shape)
        assert data_img_size == np.prod(img_size) * 3, (
            "The size of image loaded from pickle is not equal"
            "to the provided 'img_size={0}'".format(img_size)
        )
        data = data.reshape(
            (-1, 3, img_size[0], img_size[1])
        )
        # Transpose to HWC to support subsequent PIL Image opening
        data = data.transpose((0, 2, 3, 1))
        self._data = np.ascontiguousarray(data)
        self._labels = [-1] * data.shape[0]
        self._cls_name_idx_dict = {
            class_name: i for i, class_name in enumerate(class_names)
        }
        self._cls_idx_name_dict = {
            i: class_name for i, class_name in enumerate(class_names)
        }