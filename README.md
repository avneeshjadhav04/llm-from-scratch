# LLM-from-Scratch (124M Parameters)

A production-quality implementation of a **124M parameter decoder-only Transformer** language model built entirely from scratch in PyTorch. This project demonstrates deep understanding of modern LLM architecture, training dynamics, and production ML engineering practices.

**Key Highlights:**
- **~124M Parameters**: Comparable to GPT-2 small (124M), trained from scratch
- **No `nn.Transformer`**: Every component (attention, FFN, LayerNorm, embeddings) implemented manually
- **tiktoken (GPT-2)**: Industry-standard BPE tokenizer — battle-tested on billions of tokens
- **FineWeb-Edu Dataset**: High-quality web text for realistic training (10M tokens)
- **Full Training Pipeline**: Mixed precision (`torch.amp`), gradient accumulation, cosine LR scheduling, checkpoint resumption
- **Perplexity Tracking**: Both train and validation perplexity logged every evaluation cycle
- **Sample Generation**: Live text generation during training to monitor qualitative progress
- **Session-Safe Training**: Automatic session capping and resume support for Colab/Kaggle

---

## Table of Contents

- [Architecture](#architecture)
- [Dataset](#dataset)
- [Training Pipeline](#training-pipeline)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Hardware Requirements](#hardware-requirements)
- [Results](#results)
- [Key Design Decisions](#key-design-decisions)
- [License](#license)

---

## Architecture

This project implements a **decoder-only Transformer** (GPT-style), the architecture that powers modern LLMs.

### Model Specifications (~124M Parameters)

| Hyperparameter | Value |
|---------------|-------|
| `vocab_size` | 50,257 (GPT-2) |
| `d_model` | 768 |
| `n_layers` | 12 |
| `n_heads` | 12 |
| `d_ff` | 3,072 |
| `max_seq_len` | 512 |
| `dropout` | 0.1 |
| **Total Parameters** | **~124M** |

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
+-- [Repeat 12 times] ---------------------------------------------+
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

1. **Causal Self-Attention**: Multi-head scaled dot-product attention with causal (lower-triangular) mask ensuring autoregressive generation.
2. **Feed-Forward Network**: Two-layer MLP with GELU activation: `FFN(x) = W_2 * GELU(W_1 * x + b_1) + b_2`
3. **Pre-Normalization**: LayerNorm applied before attention and FFN for stable deep network training.
4. **Weight Tying**: Input token embedding matrix shared with output projection, reducing parameters and improving perplexity.
5. **Gradient Checkpointing**: Optional — recomputes activations during backward to trade compute for memory.

---

## Dataset

We use **FineWeb-Edu** (sample-10BT subset), a high-quality educational web text corpus filtered from Common Crawl.

| Property | Value |
|----------|-------|
| **Source** | HuggingFaceFW/fineweb-edu |
| **Tokens** | 10,000,000 (configurable) |
| **Train/Val Split** | 95% / 5% |
| **Format** | Pre-tokenized binary (uint16 memmap) |
| **Quality** | Educational content, high signal-to-noise |

**Why FineWeb-Edu over Wikitext-2?**
- Wikitext-2 has only ~2M tokens — too small for meaningful training
- FineWeb-Edu provides diverse, high-quality web text
- Better generalization and more interesting generated text

---

## Training Pipeline

### Features

| Feature | Implementation |
|---------|---------------|
| **Mixed Precision** | `torch.amp.autocast` with `GradScaler` for FP16 training |
| **Gradient Accumulation** | Effective batch size = 64 (16 × 4 micro-steps) |
| **LR Scheduling** | Cosine with linear warmup (2,000 steps) |
| **Gradient Clipping** | Max norm = 1.0 |
| **Checkpointing** | Every 5,000 steps + automatic resume |
| **Logging** | CSV with loss, perplexity, LR, tok/s |
| **Sample Generation** | Live text generation every 2,000 steps |
| **Session Limits** | `--max_steps_per_session` for Colab/Kaggle safety |

### Learning Rate Schedule

Cosine decay with warmup:

```
lr(t) = lr_max * (t / T_warmup)              if t < T_warmup
lr(t) = lr_min + 0.5*(lr_max - lr_min)*(1 + cos(pi * (t - T_warmup)/(T_max - T_warmup)))  otherwise
```

| Phase | Steps | LR Range |
|-------|-------|----------|
| Warmup | 0 → 2,000 | 0 → 6e-4 |
| Decay | 2,000 → 100,000 | 6e-4 → 0 |

---

## Project Structure

```
llm-from-scratch/
├── data/
│   ├── __init__.py
│   ├── tokenizer.py          # tiktoken (GPT-2) wrapper
│   └── dataset.py            # FineWeb-Edu loading + binary conversion
├── model/
│   ├── __init__.py
│   └── transformer.py        # Manual Transformer implementation
├── utils/
│   ├── __init__.py
│   ├── training.py           # Trainer, LR schedule, checkpointing, logging
│   └── sampling.py           # Temperature, top-k, top-p sampling
├── tests/
│   ├── test_tokenizer.py     # tiktoken round-trip tests
│   ├── test_model.py         # Forward pass and gradient tests
│   └── test_attention.py     # Causality and attention weight tests
├── notebooks/
│   └── playground.ipynb      # Interactive generation notebook
├── checkpoints/              # Saved model weights (.gitignored)
├── logs/                     # Training logs and loss curves
├── config.py                 # Unified configuration
├── prepare_data.py           # Download FineWeb-Edu and tokenize
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
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
pytest tests/ -v
```

---

## Usage

### Step 1: Prepare Data

Download FineWeb-Edu and tokenize:

```bash
python prepare_data.py
```

This will:
- Stream FineWeb-Edu from HuggingFace
- Tokenize with tiktoken (GPT-2)
- Save binary files to `data/corpus_*.bin`

### Step 2: Train the Model

```bash
# GPU training (recommended)
python train.py

# CPU training (for testing only)
python train.py --device cpu

# Override parameters
python train.py --batch_size 8 --learning_rate 3e-4 --max_steps 50000
```

Training outputs:
- Checkpoints: `checkpoints/100m_step_*.pt`
- Logs: `logs/100m_training_log.csv`
- Loss curve: `logs/loss_curve.png`

### Step 3: Generate Text

```bash
python generate.py \
    --checkpoint checkpoints/100m_step_50000.pt \
    --prompt "The future of artificial intelligence is" \
    --max_new_tokens 256 \
    --temperature 0.8 \
    --top_k 40 \
    --top_p 0.95
```

### Google Colab

Open `LLM_from_Scratch_100M_Colab.ipynb` in Colab:
1. Enable GPU runtime
2. Mount Google Drive (for checkpoint persistence)
3. Run all cells
4. Save Version to commit outputs
5. Resume from Drive in new sessions

---

## Hardware Requirements

### Recommended: Google Colab / Kaggle (T4 GPU, 16GB VRAM)

| Setting | Value | Purpose |
|---------|-------|---------|
| `batch_size` | 16 | Fits in 16GB with torch.compile |
| `grad_accum_steps` | 4 | Effective batch size = 64 |
| `dtype` | float16 | Mixed precision halves memory |
| `compile_model` | True | torch.compile for ~1.5× speedup |
| `max_seq_len` | 512 | Longer context window |

### Minimum: CPU (Testing Only)

```bash
python train.py --device cpu --batch_size 1 --max_steps 100
```

Extremely slow — use only for smoke tests.

---

## Results

### Expected Training Progress (FineWeb-Edu, 10M tokens)

| Step | Train Loss | Val Loss | Train PPL | Val PPL | Notes |
|------|-----------|----------|-----------|---------|-------|
| 0 | ~10.8 | — | ~49,000 | — | Random init |
| 1,000 | ~5.5 | ~5.4 | ~245 | ~221 | Word structure |
| 5,000 | ~3.6 | ~3.7 | ~36 | ~40 | Grammar & syntax |
| 10,000 | ~3.0 | ~3.2 | ~20 | ~25 | Short phrases |
| 25,000 | ~2.5 | ~2.8 | ~12 | ~16 | Sentence coherence |
| 50,000 | ~2.1 | ~2.5 | ~8 | ~12 | Paragraph coherence |

> These are approximate targets. Actual results depend on dataset quality and hyperparameters.

### Sample Generation (After 10K Steps)

```
Prompt: The future of artificial intelligence is
Output:  likely to be shaped by advances in deep learning and natural
language processing. Researchers at OpenAI and Google have demonstrated
that large language models can generate coherent text, write code, and
even solve complex reasoning problems when trained on diverse web data...
```

---

## Key Design Decisions

1. **tiktoken over Custom BPE**: GPT-2's tokenizer is battle-tested on billions of tokens. Custom BPE is educational but produces lower-quality tokens.
2. **FineWeb-Edu over Wikitext-2**: 10M tokens of diverse web text >> 2M tokens of Wikipedia for learning general language patterns.
3. **Pre-Normalization**: LayerNorm before sublayers trains more stably for deep networks.
4. **Weight Tying**: Sharing input/output embeddings saves ~38M parameters and improves perplexity.
5. **torch.compile**: Hoping for ~1.5× speedup on modern GPUs with minimal code change.
6. **Session Limits**: `--max_steps_per_session` prevents losing progress when cloud runtimes disconnect.

---

## License

This project is released under the MIT License for educational and portfolio purposes.

---

## Acknowledgments

- Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and [llm.c](https://github.com/karpathy/llm.c)
- Transformer architecture based on [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- FineWeb-Edu dataset by [HuggingFace](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
