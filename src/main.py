from args.args import get_args
from executor.executor import Executor
from executor.fashion_mnist_executor import FashionMnistExecutor
from executor.mnist_executor import MnistExecutor
from preprocessor.FashionMnistPreprocessor import FashionMnistPreprocessor
from preprocessor.FlowersPreprocessor import FlowersPreprocessor
from preprocessor.MnistPreprocessor import MnistPreprocessor
from preprocessor.preprocessor import Preprocessor
from repository.fashion_mnist_repository import FashionMnistRepositroy
from repository.mnist_repository import MnistRepositroy
from repository.repository import Repository

if __name__ == "__main__":
    args = get_args()
    executor_name = str(args.executor_name)
    if executor_name == "mnist":
        repository: Repository = MnistRepositroy()
        preprocessor: Preprocessor = MnistPreprocessor()
        executor: Executor = MnistExecutor(repository, preprocessor)
    elif executor_name == "fashion_mnist":
        repository: Repository = FashionMnistRepositroy()
        preprocessor: Preprocessor = FashionMnistPreprocessor()
        executor: Executor = FashionMnistExecutor(repository, preprocessor)
    elif executor_name == "flowers":
        repository: Repository = FashionMnistRepositroy()
        preprocessor: Preprocessor = FlowersPreprocessor()
        executor: Executor = FashionMnistExecutor(repository, preprocessor)
    else:
        raise Exception("{} not exit".format(executor_name))
    executor.run()
