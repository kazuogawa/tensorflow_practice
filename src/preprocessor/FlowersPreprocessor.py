from typing import Optional

import tensorflow as tf

from domain.data_source.data_source import DataSource
from preprocessor.preprocessor import Preprocessor


class FlowersPreprocessor(Preprocessor):
    def resize_image(
        self,
        data_source: DataSource,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> DataSource:
        pass

    def normalization(self, data_source: DataSource) -> DataSource:
        """
        本来はsklearn.preprocessing.minmax_scaleとか使うべきだろうが、今回は0-255の範囲なので、255で割ることにする
        :return:
        """
        pass

    def execute(self, data_source: DataSource) -> DataSource:
        pass
