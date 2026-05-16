# LLM-from-Scratch (100M Parameters)

A complete implementation of a **100M parameter decoder-only Transformer** language model built entirely from scratch in PyTorch. This project demonstrates deep understanding of modern LLM architecture at scale, including a custom Byte-Pair Encoding (BPE) tokenizer, manual multi-head causal self-attention, and a full training pipeline with mixed precision, gradient accumulation, gradient checkpointing, and checkpointing.

**Key Highlights:**
- **~107M Parameters**: Comparable to GPT-2 small (124M), trained from scratch.
- **No `nn.Transformer`**: Every component (attention, FFN, LayerNorm, embeddings) is implemented manually.
- **BPE Tokenizer from Scratch**: Custom byte-level BPE with GPT-2 style pre-tokenization.
- **Memory Optimizations**: Gradient checkpointing, gradient accumulation, mixed precision (`torch.amp`), and CPU-compatible fallback.
- **Production-Ready Training**: Cosine LR scheduling, checkpointing, CSV logging with loss curve visualization.
- **Advanced Sampling**: Temperature scaling, top-k, and nucleus (top-p) decoding.
- **Comprehensive Testing**: Unit tests for attention causality, model shapes, tokenizer round-trips, and gradient flow.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Mathematical Foundations](#mathematical-foundations)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Hardware Requirements](#hardware-requirements)
- [Memory Optimizations](#memory-optimizations)
- [Training Tips](#training-tips)
- [Testing](#testing)
- [Results](#results)
- [Future Work](#future-work)

---

## Architecture Overview

This project implements a **decoder-only Transformer** (GPT-style), the architecture that powers modern LLMs like GPT-4, Llama, and Claude.

### Model Specifications (~107M Parameters)

| Hyperparameter | Value |
|---------------|-------|
| `vocab_size` | 10,000 |
| `d_model` | 768 |
| `n_layers` | 14 |
| `n_heads` | 12 |
| `d_ff` | 3,072 |
| `max_seq_len` | 256 |
| `dropout` | 0.1 |
| **~Total Parameters** | **~107M** |

### Architecture Diagram

```
Input Tokens
    |
    v
[Token Embedding] + [Positional Embedding]
    |
    v
[Dropout]
    |
    v
+--> [LayerNorm] --> [Causal Self-Attention] --> [Residual Add] --+
|                                                                  |
|    [LayerNorm] --> [Feed-Forward Network] --> [Residual Add]    |
|                                                                  |
+-- [Repeat 14 times] ---------------------------------------------+
    |
    v
[Final LayerNorm]
    |
    v
[Linear Projection to Vocab] (Weight-tied with embeddings)
    |
    v
Logits -> Softmax -> Next Token Prediction
```

### Key Components

#### 1. Causal Self-Attention
Multi-head scaled dot-product attention with a causal (lower-triangular) mask to ensure the model only attends to past and current tokens.

#### 2. Feed-Forward Network
Two-layer MLP with GELU activation: `FFN(x) = W_2 * GELU(W_1 * x + b_1) + b_2`

#### 3. Pre-Normalization
LayerNorm is applied **before** attention and FFN (unlike original Transformer), which stabilizes training for deep networks.

#### 4. Weight Tying
The input token embedding matrix is shared with the output projection matrix, reducing parameters and improving performance.

#### 5. Gradient Checkpointing
During training, intermediate activations in Transformer blocks are recomputed during the backward pass instead of being stored. This trades ~30% extra compute for ~40% memory savings, enabling a 100M model on a single T4 GPU.

---

## Mathematical Foundations

### Scaled Dot-Product Attention

Given queries **Q**, keys **K**, and values **V**:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Where $d_k$ is the dimension of each attention head. The scaling factor $\sqrt{d_k}$ prevents dot products from growing too large in magnitude, which would push the softmax into regions with extremely small gradients.

### Causal Masking

To enforce autoregressive generation, we apply a causal mask:

$$
\text{MaskedAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

Where $M$ is a mask matrix with $M_{ij} = -\infty$ if $j > i$ (future positions), and $0$ otherwise. This ensures each position can only attend to itself and previous positions.

### Multi-Head Attention

Instead of performing a single attention function, we project into $h$ different subspaces:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$

Where each head is computed independently:

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

### Layer Normalization

LayerNorm normalizes across the feature dimension for each sample independently:

$$
\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

Where $\mu$ and $\sigma^2$ are the mean and variance computed over the last dimension, and $\gamma$, $\beta$ are learned affine parameters.

### GELU Activation

The Gaussian Error Linear Unit (GELU) is a smooth non-linearity used in modern Transformers:

$$
\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
$$

Where $\Phi(x)$ is the cumulative distribution function of the standard normal distribution.

### Byte-Pair Encoding (BPE)

BPE is a subword tokenization algorithm that starts with a vocabulary of individual characters and iteratively merges the most frequent adjacent pairs:

1. **Initialize** vocabulary with all unique characters in the corpus.
2. **Count** all adjacent symbol pairs in the corpus.
3. **Merge** the most frequent pair into a new symbol.
4. **Repeat** until reaching the desired vocabulary size.

This project implements **byte-level BPE**, which handles any Unicode text by falling back to byte sequences for unknown characters, ensuring the tokenizer can encode any input without unknown tokens.

### Learning Rate Schedule

We use a cosine learning rate schedule with linear warmup:

$$
\text{lr}(t) =
\begin{cases}
\text{lr}_{\max} \cdot \frac{t}{T_{\text{warmup}}} & \text{if } t < T_{\text{warmup}} \\
\text{lr}_{\min} + \frac{1}{2}(\text{lr}_{\max} - \text{lr}_{\min})\left(1 + \cos\left(\pi \cdot \frac{t - T_{\text{warmup}}}{T_{\max} - T_{\text{warmup}}}\right)\right) & \text{otherwise}
\end{cases}
$$

---

## Project Structure

```
llm-from-scratch/
├── data/
│   ├── __init__.py
│   ├── tokenizer.py          # BPE tokenizer from scratch
│   └── dataset.py            # Data loading and batching
├── model/
│   ├── __init__.py
│   └── transformer.py        # Manual Transformer implementation
├── utils/
│   ├── __init__.py
│   ├── training.py           # Training loop, LR schedule, checkpointing
│   └── sampling.py           # Temperature, top-k, top-p sampling
├── tests/
│   ├── test_tokenizer.py     # BPE round-trip and save/load tests
│   ├── test_model.py         # Forward pass and gradient tests
│   └── test_attention.py     # Causality and attention weight tests
├── notebooks/
│   └── playground.ipynb      # Interactive generation notebook
├── checkpoints/              # Saved model weights (.gitignored)
├── logs/                     # Training logs and loss curves
├── config.py                 # Unified 100M parameter configuration
├── prepare_data.py           # Download corpus and train tokenizer
├── train.py                  # Main training script
├── generate.py               # Text generation CLI
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/avneeshjadhav04/llm-from-scratch.git
cd llm-from-scratch
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
pytest tests/
```

---

## Usage

### Step 1: Prepare Data & Train Tokenizer

Download the Wikitext-2 corpus and train the BPE tokenizer:

```bash
python prepare_data.py
```

This will:
- Download Wikitext-2 to `data/wikitext-2/`
- Train the BPE tokenizer and save it to `data/tokenizer_*.json`
- Tokenize the corpus and save binary files to `data/corpus_*.bin`

### Step 2: Train the Model

```bash
# GPU training (recommended)
python train.py

# CPU training (slower, for testing)
python train.py --device cpu

# Override specific parameters
python train.py --batch_size 2 --learning_rate 5e-5 --max_steps 50000
```

Training outputs:
- Checkpoints saved to `checkpoints/` every 2,500 steps
- Training logs saved to `logs/100m_training_log.csv`
- Loss curve plot saved to `logs/loss_curve.png`

### Step 3: Generate Text

```bash
python generate.py \
    --checkpoint checkpoints/100m_step_100000.pt \
    --prompt "The future of artificial intelligence is" \
    --max_new_tokens 256 \
    --temperature 0.8 \
    --top_k 50
```

### Interactive Playground

Open `notebooks/playground.ipynb` in Jupyter to experiment with different prompts and sampling parameters interactively.

---

## Hardware Requirements

### Recommended: Google Colab / Kaggle (T4 GPU, 16GB VRAM)

| Setting | Value | Purpose |
|---------|-------|---------|
| `batch_size` | 4 | Fits in 16GB with gradient checkpointing |
| `grad_accum_steps` | 8 | Effective batch size = 32 |
| `dtype` | float16 | Mixed precision halves memory |
| `gradient_checkpointing` | True | Saves ~40% activation memory |
| `compile_model` | True | `torch.compile` for ~1.5x speedup |

### Minimum: Local Machine (CPU)

Training on CPU is possible for testing but extremely slow for the full 100K steps. Use `--device cpu --batch_size 1 --max_steps 100` for smoke tests only.

---

## Memory Optimizations

Training a 100M parameter model on a single GPU requires careful memory management:

1. **Gradient Checkpointing**: Recomputes activations during backward pass. Enabled by default. Adds ~30% compute overhead but saves ~40% memory.
2. **Mixed Precision (FP16)**: Uses `torch.amp.autocast` to run forward/backward in half-precision. Nearly 2x memory savings with minimal accuracy loss.
3. **Gradient Accumulation**: Splits the effective batch size (32) across 8 micro-steps. Each micro-step uses only batch_size=4.
4. **Weight Tying**: Shares input/output embedding matrix. Saves ~7.7M parameters (~7% of total).

---

## Training Tips

### For Google Colab / Kaggle
1. **Enable GPU runtime** before starting (Runtime → Change runtime type → GPU).
2. **Download checkpoints** before the session expires (especially on free tier).
3. **Use `torch.compile`** for ~1.5x speedup (already enabled by default).
4. **Monitor VRAM** with `!nvidia-smi` if you hit OOM errors.
5. **Expected training time**: ~50K steps takes ~6-8 hours on T4.

### If You Hit OOM
```bash
# Reduce batch size and increase accumulation
python train.py --batch_size 2 --grad_accum_steps 16

# Disable torch.compile (saves a little memory)
python train.py --compile false
```

---

## Testing

Run the full test suite:

```bash
pytest tests/ -v
```

### Test Coverage

- **`test_tokenizer.py`**: Validates BPE encode/decode round-trip, vocab size constraints, special token handling, save/load consistency, and empty string edge cases.
- **`test_model.py`**: Verifies parameter counting, forward pass output shapes, loss computation with targets, generation length, transformer block shape preservation, and gradient flow through all parameters.
- **`test_attention.py`**: Confirms causal masking (upper triangle = 0), attention weight normalization (sum = 1), distinct outputs for different inputs, and batch consistency.

---

## Results

### Expected Training Progress

On **Wikitext-2** with the 100M configuration:

| Step | Train Loss | Val Loss | Notes |
|------|------------|----------|-------|
| 0 | ~9.2 | - | Random initialization |
| 1,000 | ~4.8 | ~4.7 | Learns basic word structure |
| 10,000 | ~3.6 | ~3.5 | Learns grammar and syntax |
| 25,000 | ~3.1 | ~3.0 | Coherent short phrases |
| 50,000 | ~2.7 | ~2.6 | Sentence-level coherence |
| 100,000 | ~2.4 | ~2.3 | Paragraph-level coherence |

> These are approximate targets. Actual loss depends on corpus quality and training dynamics.

### Sample Generation (After 50K Steps)

```
Prompt: The future of artificial intelligence is
Output:  the future of artificial intelligence is a topic that has been
studied in the field of artificial intelligence. The first of these
is the use of artificial intelligence in the field of computer science.
The second is the use of artificial neural networks in the field of
machine learning...
```

> With 107M parameters, the model develops meaningful linguistic structure and topical coherence.

---

## Key Design Decisions

1. **Manual Implementation**: No `nn.Transformer`, `nn.MultiheadAttention`, or HuggingFace. Every matrix multiplication and masking operation is explicit for educational value.

2. **Pre-Norm vs Post-Norm**: We use pre-normalization (LayerNorm before sublayers) because it trains more stably for deep networks without careful initialization.

3. **Byte-Level BPE**: Instead of character-level or word-level tokenization, byte-level BPE can represent any Unicode text without unknown tokens, making it robust and real-world applicable.

4. **Weight Tying**: Sharing input/output embeddings reduces parameters by ~7.7M and often improves perplexity.

5. **Mixed Precision**: `torch.amp.autocast` automatically handles loss scaling, allowing larger models/batches without manual tensor type management.

6. **Gradient Checkpointing**: Essential for training 100M parameters on a single consumer GPU. Recomputes activations during backward pass instead of caching them.

---

## Future Work

- [ ] **Larger Datasets**: Support for OpenWebText, FineWeb-Edu, or custom corpora.
- [ ] **Rotary Positional Embeddings (RoPE)**: Replace learned positional embeddings with rotary embeddings (used in Llama).
- [ ] **Grouped Query Attention (GQA)**: Reduce KV cache memory during inference.
- [ ] **LoRA Fine-tuning**: Add parameter-efficient fine-tuning support.
- [ ] **KV-Cache Optimization**: Speed up inference by caching key/value tensors.
- [ ] **Distributed Training**: Add DDP/FSDP support for multi-GPU training.
- [ ] **Chat Template**: Format training data for instruction-following capabilities.

---

## License

This project is released under the MIT License for educational and portfolio purposes.

---

## Acknowledgments

- Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and [llm.c](https://github.com/karpathy/llm.c).
- BPE implementation follows the original [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) paper.
- Transformer architecture based on [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
