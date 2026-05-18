"""Training script for LLM-from-scratch."""
import os
import torch
import torch.optim as optim
from config import Config, parse_args
from data.dataset import get_dataloaders
from data.tokenizer import Tokenizer
from model.transformer import Transformer
from utils.training import CheckpointManager, Logger, Trainer


def main():
    config = Config()
    args = parse_args()
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
    if args.max_seq_len is not None:
        config.max_seq_len = args.max_seq_len
    if args.warmup_steps is not None:
        config.warmup_steps = args.warmup_steps
    if args.compile is not None:
        config.compile_model = args.compile

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
        config.data_dir, config.max_seq_len, config.batch_size, num_workers=2
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

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
        betas=(0.9, 0.95),
    )

    # Checkpoint and logging
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    checkpoint_manager = CheckpointManager(config.checkpoint_dir, config_name="100m")
    logger = Logger(config.log_dir, config_name="100m")

    # Resume from checkpoint if available
    start_step = 0
    checkpoint_files = sorted(
        [f for f in os.listdir(config.checkpoint_dir) if f.endswith(".pt")],
        key=lambda x: int(x.split("_")[-1].replace(".pt", ""))
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
