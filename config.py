"""Configuration loader for LLM-from-scratch."""
import argparse
from dataclasses import dataclass


@dataclass
class Config:
    """Base configuration for the model and training."""
    # Model architecture
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    max_seq_len: int
    dropout: float

    # Training
    batch_size: int
    grad_accum_steps: int
    learning_rate: float
    weight_decay: float
    max_steps: int
    warmup_steps: int
    eval_interval: int
    eval_iters: int
    checkpoint_interval: int

    # Data
    train_split: float
    num_tokens_to_train: int

    # Generation
    max_new_tokens: int
    temperature: float
    top_k: int

    # System
    device: str
    dtype: str
    compile_model: bool

    # Paths
    data_dir: str
    checkpoint_dir: str
    log_dir: str
    dataset_name: str


def get_config(name: str = "laptop") -> Config:
    if name == "laptop":
        from configs.laptop import config
    elif name == "cloud":
        from configs.cloud import config
    else:
        raise ValueError(f"Unknown config: {name}")
    return config


def add_config_args(parser: argparse.ArgumentParser):
    """Add config-related arguments to an ArgumentParser."""
    parser.add_argument("--config", type=str, default="laptop", choices=["laptop", "cloud"],
                        help="Configuration to use")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max steps")


def apply_config_overrides(config: Config, args):
    """Apply CLI overrides to a config object."""
    if args.device is not None:
        config.device = args.device
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    return config


def get_config_from_args(args=None):
    """Parse config args and return a Config object."""
    parser = argparse.ArgumentParser(description="LLM from Scratch")
    add_config_args(parser)
    parsed_args = parser.parse_args(args)
    config = get_config(parsed_args.config)
    return apply_config_overrides(config, parsed_args)
