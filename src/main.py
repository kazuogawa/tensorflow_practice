from executor.quickstart_beginner import QuickStartBeginner
from repository.minist_repository import MinistRepositroy

if __name__ == "__main__":
    minist_repository = MinistRepositroy()
    quickstart_beginner_executor = QuickStartBeginner(minist_repository)
    quickstart_beginner_executor.run()
