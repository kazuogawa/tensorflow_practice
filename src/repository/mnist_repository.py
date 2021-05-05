import tensorflow as tf

from domain.data_source.data_source import DataSource
from domain.data_source.mnist import Mnist
from repository.repository import Repository


class MnistRepositroy(Repository):
    def get_data(self) -> DataSource:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        return Mnist(x_train, y_train, x_test, y_test)
