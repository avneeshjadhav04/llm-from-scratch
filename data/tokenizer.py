"""Tokenizer wrapper using tiktoken (GPT-2 encoding)."""
import tiktoken
import os


class Tokenizer:
    """Lightweight wrapper around tiktoken for GPT-2 encoding."""

    def __init__(self, encoding_name: str = "gpt2"):
        self.enc = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.enc.n_vocab
        self.eot_token = self.enc.eot_token

    def encode(self, text: str, allowed_special: set = None) -> list:
        if allowed_special is None:
            allowed_special = {"<|endoftext|>"}
        return self.enc.encode(text, allowed_special=allowed_special)

    def decode(self, tokens: list) -> str:
        return self.enc.decode(tokens)

    def encode_single(self, text: str) -> list:
        return self.encode(text)

    def save(self, path: str):
        """tiktoken encodings are built-in; no-op save for API compatibility."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(self.enc.name)

    def load(self, path: str):
        """tiktoken encodings are built-in; no-op load for API compatibility."""
        pass
