# LLM from Scratch (124M Parameters)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A clean, from-scratch implementation of a **124M-parameter decoder-only Transformer** in PyTorch. No `nn.Transformer`, no shortcuts — every layer is built manually. Trained on **FineWeb-Edu** with mixed precision, gradient accumulation, and automatic checkpoint resume.

## What's Inside

- **Manual Transformer** — attention, FFN, LayerNorm, embeddings all coded from scratch
- **tiktoken (GPT-2)** — battle-tested BPE tokenizer, no custom tokenization
- **FineWeb-Edu Dataset** — 2B tokens of high-quality filtered web text (configurable)
- **Full Training Loop** — FP16 mixed precision, cosine LR schedule, gradient clipping, checkpointing every 5K steps
- **Session-Safe** — resume from any checkpoint, safe for Colab / Kaggle timeouts
- **Text Generation** — temperature, top-k, top-p sampling via CLI

## Quick Start

### 1. Install

```bash
git clone https://github.com/avneeshjadhav04/llm-from-scratch.git
cd llm-from-scratch
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
python prepare_data.py --num_tokens 2000000000
```

Streams FineWeb-Edu from HuggingFace, tokenizes with tiktoken, and writes `data/corpus_train.bin` + `data/corpus_val.bin`.

### 3. Train

```bash
python train.py
```

Training auto-resumes from the latest checkpoint in `checkpoints/`.

Override anything:
```bash
python train.py --batch_size 16 --learning_rate 6e-4 --max_seq_len 1024
```

Key args:
| Flag | Default | Description |
|------|---------|-------------|
| `--batch_size` | 8 | Per-GPU batch (safe for 8–16GB VRAM) |
| `--max_seq_len` | 512 | Context window |
| `--max_steps` | 0 | 0 = auto-compute from `--num_tokens` |
| `--max_steps_per_session` | 0 | Cap steps per run (for Colab timeouts) |

### 4. Generate Text

```bash
python generate.py \
    --checkpoint checkpoints/100m_step_50000.pt \
    --prompt "The future of artificial intelligence is"
```

## Notebooks

| Notebook | Platform |
|----------|----------|
| `LLM_from_Scratch_100M.ipynb` | Generic (local, Kaggle, Lightning AI) |
| `LLM_from_Scratch_100M_Colab.ipynb` | Google Colab (with Drive persistence) |

Both notebooks walk through setup, data prep, training, and checkpoint management step-by-step.

## Model

| Hyperparameter | Value |
|---------------|-------|
| `vocab_size` | 50,257 (GPT-2) |
| `d_model` | 768 |
| `n_layers` | 12 |
| `n_heads` | 12 |
| `d_ff` | 3,072 |
| `max_seq_len` | 512 |
| `dropout` | 0.1 |
| **Parameters** | **~124M** |

Architecture: pre-norm Transformer with causal self-attention, GELU FFN, and weight-tied input/output embeddings.

## Training Config

| Setting | Default | Notes |
|---------|---------|-------|
| `batch_size` | 8 | Safe default; bump to 16 on A100/T4 |
| `grad_accum_steps` | 4 | Effective batch = 32 |
| `learning_rate` | 6e-4 | With cosine decay + 2K warmup |
| `num_tokens_to_train` | 2B | Auto-computes ~122K steps at default settings |
| `dtype` | float16 | Mixed precision via `torch.amp` |
| `compile_model` | True | `torch.compile` for ~1.5× speedup |

## Hardware

| GPU | batch=8, seq=512 | batch=16, seq=512 | batch=4, seq=1024 |
|-----|------------------|-------------------|-------------------|
| T4 (16GB) | ✅ Fits | ✅ Fits | ✅ Fits |
| A100 (40GB) | ✅ Easy | ✅ Easy | ✅ Easy |
| K80 (12GB) | ✅ Fits | ⚠️ OOM risk | ✅ Fits |
| CPU | 🐌 Smoke test only | — | — |

## Results

Training on 2B tokens (~122K steps) with batch=4, seq=1024 on an A100:

| Step | Val Loss | Val PPL |
|------|----------|---------|
| 1K | ~5.4 | ~221 |
| 5K | ~3.7 | ~40 |
| 25K | ~3.1 | ~22 |
| 50K | ~2.9 | ~18 |
| 78K | ~2.6 | ~13 |

> Actual results vary with batch size and sequence length. PPL continues to improve through ~120K steps.

## Project Structure

```
llm-from-scratch/
├── data/
│   ├── dataset.py          # FineWeb-Edu streaming + binary loading
│   └── tokenizer.py        # tiktoken wrapper
├── model/
│   └── transformer.py      # Manual Transformer
├── utils/
│   ├── training.py         # Trainer, checkpoints, LR schedule, logging
│   └── sampling.py         # Temperature + top-k/top-p generation
├── tests/                  # Unit tests (tokenizer, attention, model)
├── config.py               # All defaults in one place
├── prepare_data.py         # One-liner data prep
├── train.py                # Training + resume
├── generate.py             # CLI generation
└── *.ipynb                 # Jupyter notebooks (Colab + generic)
```

## Key Design Decisions

1. **No custom tokenizer** — tiktoken (GPT-2) is battle-tested on billions of tokens.
2. **FineWeb-Edu over Wikitext-2** — 2B tokens of diverse, filtered web text beats 2M tokens of Wikipedia for generalization.
3. **Auto-compute steps** — set `num_tokens`, training figures out `max_steps` automatically. No more hardcoded 100K limit.
4. **Pre-norm + weight tying** — stable training, fewer parameters, better perplexity.
5. **Session limits** — `--max_steps_per_session` caps each run so Colab timeouts don't waste progress.

## License

Project code: **MIT License**

Dataset: **FineWeb-Edu** is released under the [ODC-By v1.0](https://opendatacommons.org/licenses/by/1-0/) license. Copyright © HuggingFace.

## Acknowledgments

- Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT)
- Transformer architecture from [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- FineWeb-Edu dataset by [HuggingFace](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
