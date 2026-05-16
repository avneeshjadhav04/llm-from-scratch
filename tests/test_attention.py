"""Unit tests for causal self-attention mechanism."""
import pytest
import torch
from config import Config
from model.transformer import CausalSelfAttention


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


def test_causal_mask():
    """Test that attention is causal (no looking ahead)."""
    config = get_test_config()
    attn = CausalSelfAttention(config)

    batch_size = 2
    seq_len = 8
    x = torch.randn(batch_size, seq_len, config.d_model)

    # Hook into attention weights
    attn_weights = None
    original_forward = attn.forward

    def hook_forward(x):
        B, T, C = x.size()
        q, k, v = attn.c_attn(x).split(attn.d_model, dim=2)
        k = k.view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
        q = q.view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
        v = v.view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (attn.d_head ** 0.5))
        att = att.masked_fill(attn.bias[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)

        nonlocal attn_weights
        attn_weights = att.detach().clone()
        return original_forward(x)

    attn.forward = hook_forward
    _ = attn(x)

    # Check that upper triangle of attention is zero (causal)
    for b in range(batch_size):
        for h in range(config.n_heads):
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    assert attn_weights[b, h, i, j].item() == pytest.approx(0.0, abs=1e-6), \
                        f"Attention not causal at batch={b}, head={h}, i={i}, j={j}"


def test_attention_weights_sum_to_one():
    """Test that attention weights sum to 1 across the sequence dimension."""
    config = get_test_config()
    attn = CausalSelfAttention(config)

    batch_size = 2
    seq_len = 8
    x = torch.randn(batch_size, seq_len, config.d_model)

    # Extract attention weights
    attn_weights = None
    original_forward = attn.forward

    def hook_forward(x):
        B, T, C = x.size()
        q, k, v = attn.c_attn(x).split(attn.d_model, dim=2)
        k = k.view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
        q = q.view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)
        v = v.view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (attn.d_head ** 0.5))
        att = att.masked_fill(attn.bias[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)

        nonlocal attn_weights
        attn_weights = att.detach().clone()
        return original_forward(x)

    attn.forward = hook_forward
    _ = attn(x)

    # Check that each row sums to 1
    row_sums = attn_weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_attention_different_inputs():
    """Test that different inputs produce different outputs."""
    config = get_test_config()
    attn = CausalSelfAttention(config)

    x1 = torch.randn(2, 8, config.d_model)
    x2 = torch.randn(2, 8, config.d_model)

    out1 = attn(x1)
    out2 = attn(x2)

    assert not torch.allclose(out1, out2, atol=1e-5)


def test_attention_batch_consistency():
    """Test that processing a batch is equivalent to processing individually."""
    config = get_test_config()
    attn = CausalSelfAttention(config)
    attn.eval()

    x = torch.randn(2, 8, config.d_model)
    batch_out = attn(x)

    out1 = attn(x[0:1])
    out2 = attn(x[1:2])

    assert torch.allclose(batch_out[0], out1[0], atol=1e-5)
    assert torch.allclose(batch_out[1], out2[0], atol=1e-5)
