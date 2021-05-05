import tensorflow as tf

from domain.data_source.data_source import DataSource
from domain.data_source.fashion_mnist import FashionMnist
from repository.repository import Repository


class FashionMnistRepositroy(Repository):
    def get_data(self) -> DataSource:
        (x_train, y_train), (
            x_test,
            y_test,
        ) = tf.keras.datasets.fashion_mnist.load_data()
        return FashionMnist(x_train, y_train, x_test, y_test)
