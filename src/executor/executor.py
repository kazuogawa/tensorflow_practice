from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.engine.sequential import Sequential

from preprocessor.preprocessor import Preprocessor
from repository.repository import Repository


class Executor(ABC):
    def __init__(self, repository: Repository, preprocessor: Preprocessor):
        self.repository = repository
        self.preprocessor = preprocessor
        self.data_source = preprocessor.run(repository.get_data())

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

    def plot_image(self, i, predictions_array, true_label, img):
        predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = "blue"
        else:
            color = "red"

        plt.xlabel(
            "{} {:2.0f}% ({})".format(
                self.data_source.label_dict[predicted_label],
                100 * np.max(predictions_array),
                self.data_source.label_dict[true_label],
            ),
            color=color,
        )

    def plot_value_array(self, i, predictions_array, true_label):
        predictions_array, true_label = predictions_array[i], true_label[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color("red")
        thisplot[true_label].set_color("blue")

    def plot_x_and_probability_and_ans(self, x, y, prediction):
        num_rows = 5
        num_cols = 3
        num_images = num_rows * num_cols
        plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
            self.plot_image(i, prediction, y, x)
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
            self.plot_value_array(i, prediction, y)
        plt.show()
