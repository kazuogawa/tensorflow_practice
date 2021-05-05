import pathlib
from typing import List

import tensorflow as tf

from domain.data_source import DataSource
from domain.data_source.flowers import Flowers
from repository.repository import Repository


class FlowersRepositroy(Repository):
    def _get_flowers_df(self) -> Flowers:
        data_root_orig = tf.keras.utils.get_file(
            origin="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
            fname="flower_photos",
            untar=True,
        )
        data_root = pathlib.Path(data_root_orig)
        all_image_paths: List[str] = list(
            map(lambda path: str(path), list(data_root.glob("*/*")))
        )
        image_tensors: List[tf.Tensor] = list(
            map(lambda path: self.image_path_to_image_tensor(path), all_image_paths)
        )

    def image_path_to_image_tensor(self, path: str) -> tf.Tensor:
        return tf.image.decode_image(tf.io.read_file(path))

    def get_data(self) -> DataSource:
        return self._get_flowers_df()
