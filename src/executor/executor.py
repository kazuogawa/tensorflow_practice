from abc import ABC

from tensorflow.python.keras.engine.sequential import Sequential

from domain.data_source import DataSource


class Executor(ABC):
    def generate_model(self, data_source: DataSource) -> Sequential:
        pass

    def run(self):
        pass
