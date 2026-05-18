"""Unit tests for the BPE tokenizer."""
import pytest
from data.tokenizer import Tokenizer


def test_tokenizer_encode_decode_roundtrip():
    """Test that encoding and decoding text returns the original string."""
    text = "Hello world! This is a test. Special chars: émojis"
    tokenizer = Tokenizer()
    token_ids = tokenizer.encode(text)
    decoded = tokenizer.decode(token_ids)
    assert isinstance(token_ids, list)
    assert len(token_ids) > 0
    assert decoded == text


def test_tokenizer_vocab_size():
    """Test that tokenizer reports correct vocab size."""
    tokenizer = Tokenizer()
    assert tokenizer.vocab_size == 50257


def test_tokenizer_empty_string():
    """Test tokenizer handles empty strings."""
    tokenizer = Tokenizer()
    ids = tokenizer.encode("")
    assert ids == []
    decoded = tokenizer.decode([])
    assert decoded == ""


def test_tokenizer_special_token():
    """Test special token handling."""
    tokenizer = Tokenizer()
    text = "<|endoftext|>"
    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    assert len(ids) == 1
    assert ids[0] == tokenizer.eot_token
