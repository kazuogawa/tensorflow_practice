import tensorflow as tf
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

    def run(self):
        model = self.generate_model()
        fitted_model = self.fitting(model)
        y_pred = fitted_model.predict(self.data_source.x_test)
        self.plot_x_and_probability_and_ans(
            self.data_source.x_test, self.data_source.y_test, y_pred
        )
