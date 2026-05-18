"""Dataset loading and batching for LLM training."""
import os
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader


def prepare_fineweb_edu(output_dir: str = "data", num_tokens: int = 10_000_000) -> str:
    """Download FineWeb-Edu dataset and save as tokenized binary."""
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "corpus_train.bin")
    val_path = os.path.join(output_dir, "corpus_val.bin")

    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"Tokenized corpus already exists at {output_dir}")
        return output_dir

    print("Loading FineWeb-Edu dataset (sample)...")
    # Use a small subset for 10M tokens target
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)

    from data.tokenizer import Tokenizer
    tokenizer = Tokenizer()

    all_tokens = []
    target_tokens = num_tokens
    train_split = 0.95

    print(f"Tokenizing up to {target_tokens:,} tokens...")
    for i, example in enumerate(ds):
        text = example["text"]
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)

        if len(all_tokens) >= target_tokens:
            break

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1} documents, {len(all_tokens):,} tokens...")

    all_tokens = np.array(all_tokens[:target_tokens], dtype=np.uint16)
    n = int(len(all_tokens) * train_split)
    train_ids = all_tokens[:n]
    val_ids = all_tokens[n:]

    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    print(f"Saved {len(train_ids):,} train tokens to {train_path}")
    print(f"Saved {len(val_ids):,} val tokens to {val_path}")
    return output_dir


class TextDataset(Dataset):
    """Dataset that loads pre-tokenized data and serves chunks of fixed length."""

    def __init__(self, data_path: str, seq_len: int):
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx:idx+self.seq_len].astype(np.int64))
        y = torch.from_numpy(self.data[idx+1:idx+self.seq_len+1].astype(np.int64))
        return x, y


def get_dataloaders(data_dir: str, seq_len: int, batch_size: int, num_workers: int = 2):
    """Create train and validation dataloaders from pre-saved binary files."""
    train_path = os.path.join(data_dir, "corpus_train.bin")
    val_path = os.path.join(data_dir, "corpus_val.bin")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Corpus binaries not found. Run 'python prepare_data.py' first."
        )

    train_dataset = TextDataset(train_path, seq_len)
    val_dataset = TextDataset(val_path, seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader
