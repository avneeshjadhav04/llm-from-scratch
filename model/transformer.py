"""Transformer model implemented from scratch using PyTorch."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class LayerNorm(nn.Module):
    """Layer normalization."""

    def __init__(self, ndim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention implemented manually."""

    def __init__(self, config: Config):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.d_model = config.d_model

        # Key, Query, Value projections
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=True)
        # Output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=True)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .view(1, 1, config.max_seq_len, config.max_seq_len)
        )

    def forward(self, x):
        B, T, C = x.size()

        # Compute Q, K, V
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, nh, T, hs)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))  # (B, nh, T, T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, hs)

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, config: Config):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_ff, bias=True)
        self.c_proj = nn.Linear(config.d_ff, config.d_model, bias=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """A single Transformer decoder block."""

    def __init__(self, config: Config):
        super().__init__()
        self.ln_1 = LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.d_model)
        self.mlp = FeedForward(config)

    def forward(self, x):
        # Pre-normalization architecture
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """Decoder-only Transformer language model."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)
        print(f"Number of parameters: {self.get_num_params()/1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.max_seq_len, f"Sequence length {t} exceeds maximum {self.config.max_seq_len}"

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # (1, t)

        tok_emb = self.wte(idx)  # (b, t, d_model)
        pos_emb = self.wpe(pos)  # (1, t, d_model)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (b, t, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate tokens given a prompt."""
        for _ in range(max_new_tokens):
            # Crop to max sequence length
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
