"""Dataset loading and batching for LLM training."""
import os
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def prepare_fineweb_edu(output_dir: str = "data", num_tokens: int = 200_000_000) -> str:
    """Download FineWeb-Edu dataset and save as tokenized binary.
    
    Memory-efficient implementation: streams documents, tokenizes in batches,
    and writes directly to disk to avoid loading all tokens into RAM.
    """
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "corpus_train.bin")
    val_path = os.path.join(output_dir, "corpus_val.bin")
    temp_path = os.path.join(output_dir, "corpus_temp.bin")

    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"Tokenized corpus already exists at {output_dir}")
        return output_dir

    print("Loading FineWeb-Edu dataset (sample)...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)

    from data.tokenizer import Tokenizer
    tokenizer = Tokenizer()

    target_tokens = num_tokens
    train_split = 0.95
    write_batch_size = 1_000_000  # ~2MB buffer

    print(f"Tokenizing up to {target_tokens:,} tokens (memory-efficient batched write)...")
    batch_tokens = []
    total_written = 0
    doc_count = 0

    with open(temp_path, "wb") as f:
        pbar = tqdm(total=target_tokens, unit="tok", desc="Tokenizing")
        for example in ds:
            text = example["text"]
            tokens = tokenizer.encode(text)
            batch_tokens.extend(tokens)
            doc_count += 1

            # Flush batch to disk when buffer is full
            while len(batch_tokens) >= write_batch_size and total_written < target_tokens:
                to_write = min(write_batch_size, target_tokens - total_written)
                chunk = np.array(batch_tokens[:to_write], dtype=np.uint16)
                chunk.tofile(f)
                total_written += to_write
                batch_tokens = batch_tokens[to_write:]
                pbar.update(to_write)

            if total_written >= target_tokens:
                break

        # Flush remaining tokens
        if batch_tokens and total_written < target_tokens:
            to_write = min(len(batch_tokens), target_tokens - total_written)
            chunk = np.array(batch_tokens[:to_write], dtype=np.uint16)
            chunk.tofile(f)
            total_written += to_write
            pbar.update(to_write)

        pbar.close()

    print(f"  Processed {doc_count:,} documents, wrote {total_written:,} tokens to temp file")

    # Split into train/val using memmap (zero RAM copy)
    full = np.memmap(temp_path, dtype=np.uint16, mode="r")
    n = int(total_written * train_split)

    # Write train split
    full[:n].tofile(train_path)
    # Write val split
    full[n:total_written].tofile(val_path)

    # Cleanup temp file
    del full
    os.remove(temp_path)

    train_size = n
    val_size = total_written - n
    print(f"Saved {train_size:,} train tokens to {train_path}")
    print(f"Saved {val_size:,} val tokens to {val_path}")
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
