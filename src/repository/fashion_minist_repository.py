import tensorflow as tf

from domain.data_source import DataSource
from domain.fashion_minist import FashionMinist
from repository.repository import Repository


class FashionMinistRepositroy(Repository):
    def _get_minist_df(self) -> FashionMinist:
        (x_train, y_train), (
            x_test,
            y_test,
        ) = tf.keras.datasets.fashion_mnist.load_data()
        return FashionMinist(x_train, y_train, x_test, y_test)

    def _get_fashion_minist_df_mapped_mini_max_normalization(self) -> FashionMinist:
        """
        本来はsklearn.preprocessing.minmax_scaleとか使うべきだろうが、今回は0-255の範囲なので、255で割ることにする
        :return:
        """
        minist = self._get_minist_df()
        return FashionMinist(
            x_train=minist.x_train / 255.0,
            y_train=minist.y_train,
            x_test=minist.x_test / 255.0,
            y_test=minist.y_test,
        )

    def get_data(self) -> DataSource:
        return self._get_fashion_minist_df_mapped_mini_max_normalization()
