"""Training script for LLM-from-scratch."""
import os
import argparse
import torch
import torch.optim as optim
from config import Config, parse_args
from data.dataset import get_dataloaders
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
    if args.compile is not None:
        config.compile_model = args.compile

    # Device setup
    device = config.device if torch.cuda.is_available() else "cpu"
    config.device = device
    print(f"Using device: {device}")

    # Create dataloaders
    train_loader, val_loader = get_dataloaders(config.data_dir, config.max_seq_len, config.batch_size)
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
    checkpoint_manager = CheckpointManager(config.checkpoint_dir)
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

    # Trainer
    trainer = Trainer(model, optimizer, config, checkpoint_manager, logger)
    trainer.step = start_step
    trainer.train(train_loader, val_loader)

    print("Training complete!")


if __name__ == "__main__":
    main()
