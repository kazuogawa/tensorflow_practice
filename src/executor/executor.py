from abc import ABC

import matplotlib.pyplot as plt
from tensorflow.python.keras.engine.sequential import Sequential

from repository.repository import Repository


class Executor(ABC):
    def __init__(self, repository: Repository):
        self.repository = repository
        self.data_source = repository.get_data()

    def generate_model(self) -> Sequential:
        pass

    def run(self):
        pass

    def plot_sample_image(self):
        """
        pixelの配列の値をplotして確認するための処理
        :return:
        """
        plt.imshow(self.data_source.x_train[0])
        plt.colorbar()
        plt.grid(False)
        plt.show()

    def plot_image_and_label(self):
        """
        pixelの配列の値とlabelを25個plotして確認するための処理
        :return:
        """
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.data_source.x_train[i], cmap=plt.cm.binary)
            plt.xlabel(self.data_source.label_dict[self.data_source.y_train[i]])
        plt.show()
