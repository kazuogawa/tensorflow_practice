import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow import Tensor
from tensorflow.python.keras.engine.sequential import Sequential

from executor.executor import Executor


class QuickStartBeginner(Executor):
    """
    https://www.tensorflow.org/tutorials/quickstart/beginner?hl=ja
    """

    def generate_model(self) -> Sequential:
        width = len(self.data_source.x_train[0][0])
        height = len(self.data_source.x_train[0])
        return tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(width, height)),
                tf.keras.layers.Dense(128, activation="relu"),
                # https://keras.io/ja/layers/core/
                # https://techblog.gmo-ap.jp/2017/11/09/%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E5%85%A5%E9%96%80%E8%80%85%E3%81%8Ckeras%E3%81%A7%E3%83%9E%E3%83%AB%E3%83%81%E3%83%AC%E3%82%A4%E3%83%A4%E3%83%BC%E3%83%91%E3%83%BC%E3%82%BB%E3%83%97%E3%83%88/
                # 過学習を避けるために出力結果を間引いているらしい
                # TODO:除いた場合とあった場合でどれくらい違うのか試す
                tf.keras.layers.Dropout(0.2),
                # 数字の0から9の10種類のため
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
        # loss = loss_fn(minist.y_train[:1], predictions).numpy()
        model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
        model.fit(self.data_source.x_train, self.data_source.y_train, epochs=5)
        model.evaluate(self.data_source.x_test, self.data_source.y_test, verbose=2)
        return model

    def run(self):
        model = self.generate_model()
        fitted_model = self.fitting(model)
        probability_model = tf.keras.Sequential(
            [fitted_model, tf.keras.layers.Softmax()]
        )
        y_pred: Tensor = probability_model(self.data_source.x_test)
        self.plot_x_and_probability_and_ans(
            self.data_source.x_test, self.data_source.y_test, y_pred
        )
