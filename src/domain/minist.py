from dataclasses import dataclass

from numpy import ndarray

from domain.data_source import DataSource


@dataclass
class Minist(DataSource):
    x_train: ndarray
    y_train: ndarray
    x_test: ndarray
    y_test: ndarray
    label_dict = {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9",
    }
