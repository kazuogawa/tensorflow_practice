from abc import ABC

from numpy import ndarray


class DataSource(ABC):
    x_train: ndarray
    y_train: ndarray
    x_test: ndarray
    y_test: ndarray
    label_dict: dict
