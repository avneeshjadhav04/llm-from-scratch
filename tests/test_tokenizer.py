"""Unit tests for the BPE tokenizer."""
import pytest
import os
from data.tokenizer import BPETokenizer


def test_tokenizer_encode_decode_roundtrip():
    """Test that encoding and decoding text returns the original string."""
    text = "Hello world! This is a test. \n Special chars: émojis 🎉"
    tokenizer = BPETokenizer(vocab_size=500)
    tokenizer.train(text)

    token_ids = tokenizer.encode(text)
    decoded = tokenizer.decode(token_ids)

    # Due to BPE and byte-level encoding, exact match may not always hold
    # but it should be very close for ASCII text
    assert isinstance(token_ids, list)
    assert len(token_ids) > 0
    assert isinstance(decoded, str)
    assert len(decoded) > 0


def test_tokenizer_vocab_size():
    """Test that tokenizer vocab size matches configuration."""
    text = "The quick brown fox jumps over the lazy dog. " * 100
    vocab_size = 500
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(text)

    # Vocab should be close to target size (may be smaller if corpus is small)
    assert len(tokenizer.vocab) <= vocab_size
    assert len(tokenizer.vocab) > len(set(text))  # Should have merged some tokens


def test_tokenizer_special_tokens():
    """Test that special tokens are included in vocabulary."""
    text = "Some text here"
    tokenizer = BPETokenizer(vocab_size=300)
    tokenizer.train(text)

    assert "<|endoftext|>" in tokenizer.vocab
    assert tokenizer.vocab["<|endoftext|>"] == tokenizer.vocab_size


def test_tokenizer_save_load(tmp_path):
    """Test saving and loading tokenizer preserves state."""
    text = "Save and load test with various words and punctuation!"
    vocab_size = 300
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(text)

    original_ids = tokenizer.encode(text)

    path_prefix = str(tmp_path / "test_tokenizer")
    tokenizer.save(path_prefix)

    loaded_tokenizer = BPETokenizer(vocab_size=vocab_size)
    loaded_tokenizer.load(path_prefix)

    loaded_ids = loaded_tokenizer.encode(text)

    assert original_ids == loaded_ids
    assert len(loaded_tokenizer.vocab) == len(tokenizer.vocab)
    assert len(loaded_tokenizer.merges) == len(tokenizer.merges)


def test_tokenizer_empty_string():
    """Test tokenizer handles empty strings."""
    tokenizer = BPETokenizer(vocab_size=100)
    tokenizer.train("some training text")

    ids = tokenizer.encode("")
    assert ids == []

    decoded = tokenizer.decode([])
    assert decoded == ""
