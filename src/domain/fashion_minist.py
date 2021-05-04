from dataclasses import dataclass, field
from typing import List

from numpy import ndarray

from domain.data_source import DataSource


@dataclass
class FashionMinist(DataSource):
    x_train: ndarray
    y_train: ndarray
    x_test: ndarray
    y_test: ndarray
    label_name = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
