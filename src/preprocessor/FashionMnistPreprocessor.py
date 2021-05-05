from domain.data_source.data_source import DataSource
from domain.data_source.fashion_mnist import FashionMnist
from preprocessor.preprocessor import Preprocessor


class FashionMnistPreprocessor(Preprocessor):
    def normalization(self, data_source: DataSource) -> DataSource:
        """
        本来はsklearn.preprocessing.minmax_scaleとか使うべきだろうが、今回は0-255の範囲なので、255で割ることにする
        :return:
        """
        return FashionMnist(
            x_train=data_source.x_train / 255.0,
            y_train=data_source.y_train,
            x_test=data_source.x_test / 255.0,
            y_test=data_source.y_test,
        )
