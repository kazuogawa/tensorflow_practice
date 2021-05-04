import argparse


def get_args():
    parser = argparse.ArgumentParser(description="want to run executor name")
    parser.add_argument("executor_name", type=str, help="executor name")
    return parser.parse_args()
