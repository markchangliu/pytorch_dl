# Author: Chang Liu


"""I/O operations."""


import os
import pickle
from typing import Generator, List, Any


def gen_img_paths(
        img_dir: str
    ) -> Generator[str, None, None]:
    for root, subdirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(root, file)
                yield img_path


def gen_pickle_data(
        pickle_files: List[str],
        get_keys: List[str]
    ) -> Any:
    for pickle_file in pickle_files:
        with open(pickle_file, "rb") as f:
            data_dict = pickle.load(f, encoding="latin1")
            yield [data_dict[k] for k in get_keys]
