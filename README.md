# LLM-from-Scratch

A complete implementation of a decoder-only Transformer language model built entirely from scratch in PyTorch. This project demonstrates deep understanding of modern LLM architecture, including a custom Byte-Pair Encoding (BPE) tokenizer, manual multi-head causal self-attention, and a full training pipeline with mixed precision, gradient accumulation, and checkpointing.

**Key Highlights:**
- **No `nn.Transformer`**: Every component (attention, FFN, LayerNorm, embeddings) is implemented manually.
- **BPE Tokenizer from Scratch**: Custom byte-level BPE with GPT-2 style pre-tokenization.
- **Dual Configuration**: Optimized hyperparameters for both local laptops (GTX 1650, 4GB VRAM) and cloud GPUs (Kaggle/Colab T4, 16GB VRAM).
- **Production-Ready Training**: Gradient accumulation, mixed precision (`torch.cuda.amp`), cosine LR scheduling, checkpointing, and CSV logging with loss curve visualization.
- **Advanced Sampling**: Temperature scaling, top-k, and nucleus (top-p) decoding.
- **Comprehensive Testing**: Unit tests for attention causality, model shapes, tokenizer round-trips, and gradient flow.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Mathematical Foundations](#mathematical-foundations)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Hardware Configurations](#hardware-configurations)
- [Training Tips](#training-tips)
- [Testing](#testing)
- [Results](#results)
- [Future Work](#future-work)

---

## Architecture Overview

This project implements a **decoder-only Transformer** (GPT-style), the architecture that powers modern LLMs like GPT-4, Llama, and Claude.

### Model Specifications

| Mode | Params | `d_model` | Layers | Heads | Seq Len | Vocab | Batch Size |
|------|--------|-----------|--------|-------|---------|-------|------------|
| **Laptop** | ~2M | 128 | 4 | 4 | 128 | 5,000 | 2 (accum 8) |
| **Cloud** | ~45M | 384 | 6 | 6 | 256 | 10,000 | 16 (accum 2) |

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
+-- [Repeat N times] ----------------------------------------------+
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
├── configs/
│   ├── __init__.py
│   ├── laptop.py             # Config for GTX 1650 (4GB)
│   └── cloud.py              # Config for T4/Kaggle (16GB)
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
├── config.py                 # Config loader and CLI parser
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
git clone https://github.com/yourusername/llm-from-scratch.git
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
# Laptop mode (vocab_size=5000)
python prepare_data.py --config laptop

# Cloud mode (vocab_size=10000)
python prepare_data.py --config cloud
```

This will:
- Download Wikitext-2 to `data/wikitext-2/`
- Train the BPE tokenizer and save it to `data/tokenizer_*.json`
- Tokenize the corpus and save binary files to `data/corpus_*.bin`

### Step 2: Train the Model

```bash
# Laptop mode
python train.py --config laptop

# Cloud mode
python train.py --config cloud

# Override specific parameters
python train.py --config laptop --batch_size 4 --learning_rate 1e-4 --max_steps 5000
```

Training outputs:
- Checkpoints saved to `checkpoints/` every `checkpoint_interval` steps
- Training logs saved to `logs/*_training_log.csv`
- Loss curve plot saved to `logs/loss_curve.png`

### Step 3: Generate Text

```bash
python generate.py \
    --config laptop \
    --checkpoint checkpoints/laptop_step_10000.pt \
    --prompt "The future of artificial intelligence is" \
    --max_new_tokens 256 \
    --temperature 0.8 \
    --top_k 40
```

### Interactive Playground

Open `notebooks/playground.ipynb` in Jupyter to experiment with different prompts and sampling parameters interactively.

---

## Hardware Configurations

### Laptop Mode (GTX 1650 Mobile, 4GB VRAM)

| Setting | Value | Purpose |
|---------|-------|---------|
| `batch_size` | 2 | Fits in limited VRAM |
| `grad_accum_steps` | 8 | Effective batch size = 16 |
| `dtype` | float16 | Mixed precision halves memory |
| `d_model` | 128 | Small model (~2M params) |
| `max_seq_len` | 128 | Shorter sequences |
| `compile_model` | False | torch.compile unsupported on old GPUs |

### Cloud Mode (Kaggle/Colab T4, 16GB VRAM)

| Setting | Value | Purpose |
|---------|-------|---------|
| `batch_size` | 16 | Larger batches for stability |
| `grad_accum_steps` | 2 | Effective batch size = 32 |
| `dtype` | float16 | Mixed precision |
| `d_model` | 384 | Medium model (~45M params) |
| `max_seq_len` | 256 | Longer context windows |
| `compile_model` | True | torch.compile for speedup |

---

## Training Tips

### For Laptops (Limited VRAM)
1. **Close all other GPU applications** (browsers, games) before training.
2. **Monitor VRAM** with `nvidia-smi` during training.
3. **Reduce `max_seq_len`** to 64 if you hit OOM errors.
4. **Use gradient accumulation** to maintain effective batch size.
5. **Train overnight** - even 10k steps on a tiny model shows learning.

### For Cloud GPUs
1. **Enable `torch.compile`** for ~1.5-2x speedup.
2. **Use larger batch sizes** for more stable gradients.
3. **Train for 50k+ steps** to see coherent generation.
4. **Download checkpoints** before the session expires.

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

On **Wikitext-2** with the laptop configuration:

| Step | Train Loss | Val Loss | Notes |
|------|------------|----------|-------|
| 0 | ~8.5 | - | Random initialization |
| 500 | ~4.2 | ~4.1 | Learns basic word structure |
| 2000 | ~3.5 | ~3.4 | Learns grammar and syntax |
| 5000 | ~3.1 | ~3.0 | Coherent short phrases |
| 10000 | ~2.8 | ~2.7 | Sentence-level coherence |

### Sample Generation (After 10k Steps, Laptop Mode)

```
Prompt: The future of artificial intelligence is
Output:  the future of artificial intelligence is not yet known. The
first of the first of the first of the world is the first of
a new system of the world. The first of a new system is the
system of the system of a system ...
```

> Note: With only ~2M parameters, the model will show learning but won't produce highly coherent long-form text. The cloud configuration (~45M params) produces significantly better results.

---

## Key Design Decisions

1. **Manual Implementation**: No `nn.Transformer`, `nn.MultiheadAttention`, or HuggingFace. Every matrix multiplication and masking operation is explicit for educational value.

2. **Pre-Norm vs Post-Norm**: We use pre-normalization (LayerNorm before sublayers) because it trains more stably for deep networks without careful initialization.

3. **Byte-Level BPE**: Instead of character-level or word-level tokenization, byte-level BPE can represent any Unicode text without unknown tokens, making it robust and real-world applicable.

4. **Weight Tying**: Sharing input/output embeddings reduces parameters by ~vocab_size * d_model and often improves perplexity.

5. **Mixed Precision**: `torch.cuda.amp` automatically handles loss scaling, allowing larger models/batches without manual tensor type management.

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
