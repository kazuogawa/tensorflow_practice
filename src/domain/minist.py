from dataclasses import dataclass

from numpy import ndarray

from domain.data_source import DataSource


@dataclass
class Minist(DataSource):
    x_train: ndarray
    y_train: ndarray
    x_test: ndarray
    y_test: ndarray
