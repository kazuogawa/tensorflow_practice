from abc import ABC

from tensorflow.python.keras.engine.sequential import Sequential


class Executor(ABC):
    def generate_model(self) -> Sequential:
        pass

    def run(self):
        pass
