"""Unified configuration for LLM-from-scratch."""
import argparse
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for the model and training."""
    # Model architecture (~124M parameters)
    vocab_size: int = 50257          # GPT-2 vocab size
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072
    max_seq_len: int = 512           # Longer context for better learning
    dropout: float = 0.1

    # Training
    batch_size: int = 8              # Safe default for most GPUs (8-16GB)
    grad_accum_steps: int = 4        # Effective batch = 32
    learning_rate: float = 6e-4      # Slightly higher for better convergence on web data
    weight_decay: float = 0.1        # Standard for transformers
    max_steps: int = 0               # 0 = auto-compute from num_tokens_to_train
    max_steps_per_session: int = 0   # 0 = disabled
    warmup_steps: int = 2000
    num_workers: int = 2             # DataLoader workers (0 for CPU-only)
    grad_clip_norm: float = 1.0      # Gradient clipping max norm
    betas: tuple = (0.9, 0.95)       # AdamW beta parameters
    eval_interval: int = 1000
    eval_iters: int = 200
    checkpoint_interval: int = 5000
    log_interval: int = 10           # Print every N steps
    sample_interval: int = 2000      # Generate samples every N steps

    # Data
    train_split: float = 0.95
    num_tokens_to_train: int = 2_000_000_000

    # Generation
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95

    # System
    device: str = "cuda"
    dtype: str = "float16"
    compile_model: bool = True
    gradient_checkpointing: bool = False

    # Paths
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    run_name: str = "100m"           # Prefix for checkpoints and logs


def get_config() -> Config:
    """Return the default configuration."""
    return Config()


def parse_args():
    parser = argparse.ArgumentParser(description="LLM from Scratch")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda)")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max steps (0=auto-compute)")
    parser.add_argument("--max_steps_per_session", type=int, default=None, help="Session step limit (0=disabled)")
    parser.add_argument("--num_tokens", type=int, default=None, help="Override total tokens to train on")
    parser.add_argument("--max_seq_len", type=int, default=None, help="Override max sequence length")
    parser.add_argument("--warmup_steps", type=int, default=None, help="Override warmup steps")
    parser.add_argument("--compile", type=lambda x: x.lower() == "true", default=None, help="Override torch.compile (true/false)")
    return parser.parse_args()


def get_config_from_args():
    args = parse_args()
    config = get_config()
    if args.device is not None:
        config.device = args.device
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.max_steps_per_session is not None:
        config.max_steps_per_session = args.max_steps_per_session
    if args.num_tokens is not None:
        config.num_tokens_to_train = args.num_tokens
    if args.max_seq_len is not None:
        config.max_seq_len = args.max_seq_len
    if args.warmup_steps is not None:
        config.warmup_steps = args.warmup_steps
    if args.compile is not None:
        config.compile_model = args.compile
    return config
