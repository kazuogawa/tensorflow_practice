import tensorflow as tf
from tensorflow import Tensor
from tensorflow.python.keras.engine.sequential import Sequential

from domain.data_source import DataSource
from repository.minist_repository import MinistRepositroy


class QuickStartBeginner:
    """
    https://www.tensorflow.org/tutorials/quickstart/beginner?hl=ja
    """

    def __init__(self, repository: MinistRepositroy):
        self.repository = repository

    def generate_model(self, minist: DataSource) -> Sequential:
        # TODO: めっちゃドメインに依存してて抽象化できないなー。。。dictとかで管理する方が良さそう・・・
        width = len(minist.x_train[0][0])
        height = len(minist.x_train[0])
        return tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(width, height)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                # 0-10
                tf.keras.layers.Dense(10),
            ]
        )

    def fitting(self, model: Sequential, minist: DataSource) -> Sequential:
        # 予測値はロジットや対数オッズ比で出力される
        predictions = model(minist.x_train[:1]).numpy()
        # 確率に変換
        probability = tf.nn.softmax(predictions).numpy()
        # 損失関数。下記の書き方をすればそれぞれの標本についてクラスごとに損失のスカラを返す
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # loss確認する場合はコメント外す
        # loss = loss_fn(minist.y_train[:1], predictions).numpy()
        model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
        model.fit(minist.x_train, minist.y_train, epochs=5)
        model.evaluate(minist.x_test, minist.y_test, verbose=2)
        return model

    def run(self):
        minist: DataSource = self.repository.get_data()
        model = self.generate_model(minist)
        fitted_model = self.fitting(model, minist)
        probability_model = tf.keras.Sequential(
            [fitted_model, tf.keras.layers.Softmax()]
        )

        predict: Tensor = probability_model(minist.x_test[:5])
        print(predict)
        print(tf.math.argmax(predict, 1))
        print("ans: ", minist.y_test[:5])
