import tensorflow as tf
from tensorflow import Tensor
from tensorflow.python.keras.engine.sequential import Sequential

from domain.data_source import DataSource
from executor.executor import Executor
from repository.fashion_minist_repository import FashionMinistRepositroy


class FashionMinist(Executor):
    """
    https://www.tensorflow.org/tutorials/keras/classification?hl=ja
    """

    def __init__(self, repository: FashionMinistRepositroy):
        self.repository = repository
        self.data_source = repository.get_data()

    def generate_model(self) -> Sequential:
        width = len(self.data_source.x_train[0][0])
        height = len(self.data_source.x_train[0])
        return tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(width, height)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                # 0-10
                tf.keras.layers.Dense(10),
            ]
        )

    def fitting(self, model: Sequential) -> Sequential:
        # 予測値はロジットや対数オッズ比で出力される
        predictions = model(self.data_source.x_train[:1]).numpy()
        # 確率に変換
        probability = tf.nn.softmax(predictions).numpy()
        # 損失関数。下記の書き方をすればそれぞれの標本についてクラスごとに損失のスカラを返す
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # loss確認する場合はコメント外す
        # loss = loss_fn(self.data_source.y_train[:1], predictions).numpy()
        model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
        model.fit(self.data_source.x_train, self.data_source.y_train, epochs=5)
        model.evaluate(self.data_source.x_test, self.data_source.y_test, verbose=2)
        return model

    def run(self):
        print(self.data_source.x_train.shape)
