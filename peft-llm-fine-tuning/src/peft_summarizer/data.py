import argparse
from datasets import load_dataset

DATASET_NAME = "knkarthick/samsum"


def load_samsum(name: str = DATASET_NAME):
    return load_dataset(name)


def inspect_dataset() -> None:
    ds = load_samsum()
    print(ds)
    print("\nColumns:", ds["train"].column_names)
    print("\nExample:")
    print(ds["train"][0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()

    if args.inspect:
        inspect_dataset()
    else:
        inspect_dataset()
