import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.python.keras.engine.sequential import Sequential

from executor.executor import Executor


class FashionMinist(Executor):
    """
    https://www.tensorflow.org/tutorials/keras/classification?hl=ja
    実行するたびに結構変わる
    313/313 - 0s - loss: 0.3543 - accuracy: 0.8694
    """

    def generate_model(self) -> Sequential:
        width = len(self.data_source.x_train[0][0])
        height = len(self.data_source.x_train[0])
        return tf.keras.Sequential(
            [
                # 28x28の784の位置次元並列に変更
                tf.keras.layers.Flatten(input_shape=(width, height)),
                # 128個のノードにする
                tf.keras.layers.Dense(128, activation="relu"),
                # 過学習防止に突っ込んでみたが、あってもなくても変わらん
                # tf.keras.layers.Dropout(0.2),
                # 合計が1になる10個の確率の配列を返す。0-9の10種類の服を出力するため
                tf.keras.layers.Dense(10, activation="softmax"),
            ]
        )

    def fitting(self, model: Sequential) -> Sequential:
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(self.data_source.x_train, self.data_source.y_train, epochs=10)
        # loss, accuracyを出力してくれる
        model.evaluate(self.data_source.x_test, self.data_source.y_test, verbose=2)
        return model

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

    def run(self):
        model = self.generate_model()
        fittted_model = self.fitting(model)
        y_pred = fittted_model.predict(self.data_source.x_test)
        self.plot_x_and_probability_and_ans(
            self.data_source.x_test, self.data_source.y_test, y_pred
        )
