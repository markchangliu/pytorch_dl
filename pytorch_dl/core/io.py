# Author: Chang Liu


"""I/O operations."""


import os
from typing import Generator


def gen_img_paths(
        img_dir: str
    ) -> Generator[str, None, None]:
    for root, subdirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(root, file)
                yield img_path