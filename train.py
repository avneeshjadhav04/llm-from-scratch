"""Training script for LLM-from-scratch."""
import os
import re
import torch
import torch.optim as optim
from config import Config, get_config_from_args
from data.dataset import get_dataloaders
from data.tokenizer import Tokenizer
from model.transformer import Transformer
from utils.training import CheckpointManager, Logger, Trainer


def main():
    config = get_config_from_args()

    # Auto-compute max_steps from num_tokens_to_train if not explicitly set
    if config.max_steps == 0:
        tokens_per_step = config.batch_size * config.max_seq_len * config.grad_accum_steps
        config.max_steps = config.num_tokens_to_train // tokens_per_step
        print(f"Auto-computed max_steps: {config.max_steps:,} "
              f"({config.num_tokens_to_train:,} tokens / {tokens_per_step:,} tokens per step)")

    # Device setup
    device = config.device if torch.cuda.is_available() else "cpu"
    config.device = device
    print(f"Using device: {device}")

    # Verify corpus binaries exist
    train_bin = os.path.join(config.data_dir, "corpus_train.bin")
    val_bin = os.path.join(config.data_dir, "corpus_val.bin")
    if not os.path.exists(train_bin) or not os.path.exists(val_bin):
        raise FileNotFoundError(
            f"Corpus binaries not found: {train_bin} or {val_bin}. "
            f"Run 'python prepare_data.py' first to download and tokenize the dataset."
        )

    # Load tokenizer
    tokenizer = Tokenizer()
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Create dataloaders
    train_loader, val_loader = get_dataloaders(
        config.data_dir, config.max_seq_len, config.batch_size, num_workers=config.num_workers
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    if len(train_loader) == 0:
        raise RuntimeError(
            "Train loader is empty. The dataset may be too small for the given batch_size and max_seq_len. "
            "Run 'python prepare_data.py' first or check your data binaries."
        )

    # Create model
    model = Transformer(config).to(device)

    # Compile model if enabled
    if config.compile_model:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )

    # Checkpoint and logging
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    checkpoint_manager = CheckpointManager(config.checkpoint_dir, config_name=config.run_name)
    logger = Logger(config.log_dir, config_name=config.run_name)

    # Resume from checkpoint if available
    start_step = 0
    checkpoint_files = sorted(
        [f for f in os.listdir(config.checkpoint_dir)
         if f.endswith(".pt") and re.search(r'step_(\d+)\.pt$', f)],
        key=lambda x: int(re.search(r'step_(\d+)\.pt$', x).group(1))
    )
    if checkpoint_files:
        latest_checkpoint = os.path.join(config.checkpoint_dir, checkpoint_files[-1])
        start_step, _ = checkpoint_manager.load(latest_checkpoint, model, optimizer, device)
        print(f"Resumed training from step {start_step}")
        start_step += 1

    # Cap total steps for this session
    if config.max_steps_per_session > 0:
        session_cap = start_step + config.max_steps_per_session
        if session_cap < config.max_steps:
            config.max_steps = session_cap
            print(f"Session limit enabled: training up to step {config.max_steps}")

    # Trainer
    trainer = Trainer(
        model, optimizer, config, checkpoint_manager, logger, tokenizer=tokenizer
    )
    trainer.step = start_step
    trainer.train(train_loader, val_loader)

    print("Training complete!")


if __name__ == "__main__":
    main()
