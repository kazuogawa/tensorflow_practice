from args.args import get_args
from executor.executor import Executor
from executor.fashion_minist import FashionMinist
from executor.quickstart_beginner import QuickStartBeginner
from repository.fashion_minist_repository import FashionMinistRepositroy
from repository.minist_repository import MinistRepositroy

if __name__ == "__main__":
    args = get_args()
    executor_name = str(args.executor_name)
    if executor_name == "minist":
        repository = MinistRepositroy()
        executor: Executor = QuickStartBeginner(repository)
    elif executor_name == "fashion_minist":
        repository = FashionMinistRepositroy()
        executor: Executor = FashionMinist(repository)
    else:
        raise Exception("{} not exit".format(executor_name))
    executor.run()
