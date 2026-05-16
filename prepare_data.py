"""Prepare data: download corpus, train BPE tokenizer, and save tokenized binaries."""
import os
import argparse
from config import Config
from data.tokenizer import BPETokenizer
from data.dataset import prepare_wikitext2, load_text_files, tokenize_and_save_corpus


def main():
    parser = argparse.ArgumentParser(description="Prepare data for LLM training")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--vocab_size", type=int, default=None)
    args = parser.parse_args()

    config = Config()
    vocab_size = args.vocab_size or config.vocab_size
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)

    # Download dataset
    print("Preparing dataset...")
    corpus_dir = prepare_wikitext2(data_dir)
    text = load_text_files(corpus_dir)
    print(f"Loaded corpus with {len(text):,} characters")

    # Train tokenizer
    tokenizer_path = os.path.join(data_dir, "tokenizer")
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(text)
    tokenizer.save(tokenizer_path)

    # Tokenize and save corpus
    output_path = os.path.join(data_dir, "corpus.bin")
    tokenize_and_save_corpus(text, tokenizer, output_path, train_split=config.train_split)

    print("Data preparation complete!")


if __name__ == "__main__":
    main()
