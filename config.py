"""Unified configuration for LLM-from-scratch (100M parameters)."""
import argparse
from dataclasses import dataclass


@dataclass
class Config:
    """Base configuration for the model and training."""
    # Model architecture (~100M parameters)
    vocab_size: int = 10000
    d_model: int = 768
    n_layers: int = 14
    n_heads: int = 12
    d_ff: int = 3072
    max_seq_len: int = 256
    dropout: float = 0.1

    # Training
    batch_size: int = 16
    grad_accum_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 100000
    max_steps_per_session: int = 0  # 0 = disabled; cap max_steps for this session
    warmup_steps: int = 1000
    eval_interval: int = 1000
    eval_iters: int = 100
    checkpoint_interval: int = 2500

    # Data
    train_split: float = 0.9
    num_tokens_to_train: int = 100_000_000

    # Generation
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_k: int = 50

    # System
    device: str = "cuda"
    dtype: str = "float16"
    compile_model: bool = False
    gradient_checkpointing: bool = True
    use_data_parallel: bool = True  # Auto-use nn.DataParallel if multiple GPUs available

    # Paths
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    dataset_name: str = "wikitext-2"


def get_config() -> Config:
    """Return the default configuration."""
    return Config()


def parse_args():
    parser = argparse.ArgumentParser(description="LLM from Scratch")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda)")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max steps")
    parser.add_argument("--max_steps_per_session", type=int, default=None, help="Session step limit (0=disabled)")
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
    if args.compile is not None:
        config.compile_model = args.compile
    return config
