from typing import Optional

import tensorflow as tf

from domain.data_source.data_source import DataSource


class Preprocessor:
    def resize_image(
        self,
        data_source: DataSource,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> DataSource:
        return data_source

    def normalization(self, data_source: DataSource) -> DataSource:
        return data_source

    def run(self, data_source: DataSource) -> DataSource:
        return self.normalization(self.resize_image(data_source))
