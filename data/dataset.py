"""Dataset loading and batching for LLM training."""
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data.tokenizer import BPETokenizer


class TextDataset(Dataset):
    """Dataset that loads pre-tokenized data and serves chunks of fixed length."""

    def __init__(self, data_path: str, seq_len: int):
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx:idx+self.seq_len].astype(np.int64))
        y = torch.from_numpy(self.data[idx+1:idx+self.seq_len+1].astype(np.int64))
        return x, y


def prepare_wikitext2(output_dir: str = "data") -> str:
    """Download and extract Wikitext-2 raw dataset."""
    os.makedirs(output_dir, exist_ok=True)
    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
    zip_path = os.path.join(output_dir, "wikitext-2-v1.zip")
    data_dir = os.path.join(output_dir, "wikitext-2")

    # If data dir already exists with text files, skip download
    if os.path.isdir(data_dir) and any(
        f.endswith((".txt", ".tokens")) for f in os.listdir(data_dir)
    ):
        return data_dir

    import zipfile

    # Validate existing zip — remove if corrupt so we re-download
    if os.path.exists(zip_path):
        try:
            with zipfile.ZipFile(zip_path, "r") as _:
                pass  # Valid zip
        except zipfile.BadZipFile:
            print("Existing zip is corrupt. Re-downloading...")
            os.remove(zip_path)

    # Download with retry
    if not os.path.exists(zip_path):
        import requests
        for attempt in range(1, 4):
            try:
                print(f"Downloading Wikitext-2 (attempt {attempt}/3)...")
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                with open(zip_path, "wb") as f:
                    f.write(response.content)
                # Validate immediately after download
                with zipfile.ZipFile(zip_path, "r") as _:
                    pass
                break
            except Exception as e:
                print(f"  Download failed: {e}")
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                if attempt == 3:
                    raise RuntimeError(
                        "Failed to download Wikitext-2 after 3 attempts. "
                        "Check your internet connection or the URL availability."
                    )

    # Extract
    print("Extracting Wikitext-2...")
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(output_dir)

    return data_dir


def load_text_files(data_dir: str) -> str:
    """Load all .txt and .tokens files from a directory and concatenate them."""
    texts = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".txt") or fname.endswith(".tokens"):
            with open(os.path.join(data_dir, fname), "r", encoding="utf-8") as f:
                texts.append(f.read())
    return "\n\n".join(texts)


def tokenize_and_save_corpus(
    text: str,
    tokenizer: BPETokenizer,
    output_path: str,
    train_split: float = 0.9,
) -> None:
    """Tokenize text and save as binary files for train/val."""
    print("Encoding corpus...")
    token_ids = tokenizer.encode(text)
    token_ids = np.array(token_ids, dtype=np.uint16)

    n = int(len(token_ids) * train_split)
    train_ids = token_ids[:n]
    val_ids = token_ids[n:]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    train_path = output_path.replace(".bin", "_train.bin")
    val_path = output_path.replace(".bin", "_val.bin")

    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    print(f"Saved {len(train_ids)} train tokens to {train_path}")
    print(f"Saved {len(val_ids)} val tokens to {val_path}")


def get_dataloaders(data_dir: str, seq_len: int, batch_size: int):
    """Create train and validation dataloaders from pre-saved binary files."""
    train_path = os.path.join(data_dir, "corpus_train.bin")
    val_path = os.path.join(data_dir, "corpus_val.bin")

    train_dataset = TextDataset(train_path, seq_len)
    val_dataset = TextDataset(val_path, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader
