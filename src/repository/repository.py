from abc import ABC

from domain.data_source.data_source import DataSource


class Repository(ABC):
    def get_data(self) -> DataSource:
        pass
