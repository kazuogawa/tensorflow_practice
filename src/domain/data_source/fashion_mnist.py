from dataclasses import dataclass, field
from typing import List

from numpy import ndarray

from domain.data_source.data_source import DataSource


@dataclass
class FashionMnist(DataSource):
    x_train: ndarray
    y_train: ndarray
    x_test: ndarray
    y_test: ndarray
    label_dict = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot",
    }
