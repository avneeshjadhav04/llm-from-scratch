"""Prepare data: download FineWeb-Edu and save tokenized binaries."""
import argparse
from config import Config
from data.dataset import prepare_fineweb_edu


def main():
    parser = argparse.ArgumentParser(description="Prepare data for LLM training")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--num_tokens", type=int, default=None)
    parser.add_argument("--train_split", type=float, default=None, help="Train/validation split ratio")
    args = parser.parse_args()

    config = Config()
    num_tokens = args.num_tokens or config.num_tokens_to_train
    data_dir = args.data_dir
    train_split = args.train_split if args.train_split is not None else config.train_split

    print("Preparing FineWeb-Edu dataset...")
    prepare_fineweb_edu(data_dir, num_tokens=num_tokens, train_split=train_split)
    print("Data preparation complete!")


if __name__ == "__main__":
    main()
