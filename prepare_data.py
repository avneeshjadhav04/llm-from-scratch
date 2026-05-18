"""Prepare data: download FineWeb-Edu and save tokenized binaries."""
import argparse
from config import Config
from data.dataset import prepare_fineweb_edu


def main():
    parser = argparse.ArgumentParser(description="Prepare data for LLM training")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--num_tokens", type=int, default=None)
    args = parser.parse_args()

    config = Config()
    num_tokens = args.num_tokens or config.num_tokens_to_train
    data_dir = args.data_dir

    print("Preparing FineWeb-Edu dataset...")
    prepare_fineweb_edu(data_dir, num_tokens=num_tokens)
    print("Data preparation complete!")


if __name__ == "__main__":
    main()
