"""Unit tests for the Transformer model."""
import pytest
import torch
from config import Config
from model.transformer import Transformer, CausalSelfAttention, TransformerBlock


def get_test_config():
    return Config(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        max_seq_len=32,
        dropout=0.0,
        batch_size=4,
        grad_accum_steps=1,
        learning_rate=1e-3,
        weight_decay=0.01,
        max_steps=100,
        warmup_steps=10,
        eval_interval=50,
        eval_iters=10,
        checkpoint_interval=100,
        train_split=0.9,
        num_tokens_to_train=10000,
        max_new_tokens=50,
        temperature=1.0,
        top_k=10,
        device="cpu",
        dtype="float32",
        compile_model=False,
        data_dir="data",
        checkpoint_dir="checkpoints",
        log_dir="logs",
        dataset_name="wikitext-2",
    )


def test_model_parameter_count():
    """Test that model reports parameter count correctly."""
    config = get_test_config()
    model = Transformer(config)
    n_params = model.get_num_params()
    assert n_params > 0

    # Rough estimate: embeddings + blocks + head
    expected_min = config.vocab_size * config.d_model  # token embeddings
    assert n_params > expected_min


def test_model_forward_pass_shape():
    """Test that forward pass produces correct output shapes."""
    config = get_test_config()
    model = Transformer(config)
    batch_size = 4
    seq_len = 16

    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    logits, loss = model(idx)

    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert loss is None  # No targets provided


def test_model_forward_pass_with_targets():
    """Test forward pass with target labels."""
    config = get_test_config()
    model = Transformer(config)
    batch_size = 4
    seq_len = 16

    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, loss = model(idx, targets)

    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert loss is not None
    assert loss.ndim == 0  # Scalar loss


def test_model_generation():
    """Test text generation from a prompt."""
    config = get_test_config()
    model = Transformer(config)
    batch_size = 1
    prompt_len = 5
    max_new = 10

    idx = torch.randint(0, config.vocab_size, (batch_size, prompt_len))
    generated = model.generate(idx, max_new_tokens=max_new, temperature=1.0, top_k=None)

    assert generated.shape == (batch_size, prompt_len + max_new)


def test_transformer_block_output_shape():
    """Test that a transformer block preserves input shape."""
    config = get_test_config()
    block = TransformerBlock(config)
    batch_size = 2
    seq_len = 8
    x = torch.randn(batch_size, seq_len, config.d_model)
    out = block(x)
    assert out.shape == x.shape


def test_attention_output_shape():
    """Test that attention produces correct output shape."""
    config = get_test_config()
    attn = CausalSelfAttention(config)
    batch_size = 2
    seq_len = 8
    x = torch.randn(batch_size, seq_len, config.d_model)
    out = attn(x)
    assert out.shape == x.shape


def test_gradient_flow():
    """Test that gradients flow through the model."""
    config = get_test_config()
    model = Transformer(config)
    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))

    logits, loss = model(idx, targets)
    loss.backward()

    # Check that all parameters have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert not torch.all(param.grad == 0), f"Parameter {name} has zero gradient"
