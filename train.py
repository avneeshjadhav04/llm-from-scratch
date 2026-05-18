"""Training script for LLM-from-scratch."""
import os
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributed import init_process_group, destroy_process_group

from config import Config, parse_args
from data.dataset import get_dataloaders
from model.transformer import Transformer
from utils.training import CheckpointManager, Logger, Trainer


def ddp_setup(rank: int, world_size: int):
    """Initialize DDP process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _train_single(rank: int, world_size: int, config: Config):
    """Single process training loop (works for both single-GPU and DDP worker)."""
    is_master = rank == 0
    use_ddp = world_size > 1

    # Device setup
    if torch.cuda.is_available():
        device = f"cuda:{rank}"
    else:
        device = "cpu"
    config.device = device

    if is_master:
        print(f"Using device: {device}" + (f" ({world_size} GPUs via DDP)" if use_ddp else ""))

    # Verify corpus binaries exist before creating dataloaders
    train_bin = os.path.join(config.data_dir, "corpus_train.bin")
    val_bin = os.path.join(config.data_dir, "corpus_val.bin")
    if not os.path.exists(train_bin) or not os.path.exists(val_bin):
        raise FileNotFoundError(
            f"Corpus binaries not found: {train_bin} or {val_bin}. "
            f"Run 'python prepare_data.py' first to download and tokenize the dataset."
        )

    # Create dataloaders
    train_loader, val_loader = get_dataloaders(config.data_dir, config.max_seq_len, config.batch_size)
    if is_master:
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    model = Transformer(config).to(device)

    # Wrap in DDP if multi-GPU
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Compile model if enabled
    if config.compile_model:
        if is_master:
            print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    # Checkpoint and logging (only master)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    checkpoint_manager = CheckpointManager(config.checkpoint_dir, config_name="100m")
    logger = Logger(config.log_dir, config_name="100m") if is_master else None

    # Resume from checkpoint (only master loads, then broadcast)
    start_step = 0
    if is_master:
        checkpoint_files = sorted(
            [f for f in os.listdir(config.checkpoint_dir) if f.endswith(".pt")],
            key=lambda x: int(x.split("_")[-1].replace(".pt", ""))
        )
        if checkpoint_files:
            latest_checkpoint = os.path.join(config.checkpoint_dir, checkpoint_files[-1])
            start_step, _ = checkpoint_manager.load(latest_checkpoint, model, optimizer, device)
            print(f"Resumed training from step {start_step}")
            start_step += 1

    if use_ddp:
        start_step_tensor = torch.tensor(start_step, device=device)
        torch.distributed.broadcast(start_step_tensor, src=0)
        start_step = int(start_step_tensor.item())

    # Cap total steps for this session to avoid hitting the Kaggle time limit
    if config.max_steps_per_session > 0:
        session_cap = start_step + config.max_steps_per_session
        if session_cap < config.max_steps:
            config.max_steps = session_cap
            if is_master:
                print(f"Session limit enabled: training up to step {config.max_steps}")

    # Trainer
    trainer = Trainer(model, optimizer, config, checkpoint_manager, logger, rank=rank, world_size=world_size)
    trainer.step = start_step
    trainer.train(train_loader, val_loader)

    if is_master:
        print("Training complete!")


def train_worker(rank: int, world_size: int, config: Config):
    """DDP worker entry point."""
    ddp_setup(rank, world_size)
    try:
        _train_single(rank, world_size, config)
    finally:
        destroy_process_group()


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
    if args.compile is not None:
        config.compile_model = args.compile

    world_size = torch.cuda.device_count()

    if world_size > 1 and config.use_ddp:
        mp.spawn(train_worker, args=(world_size, config), nprocs=world_size, join=True)
    else:
        if world_size > 1 and not config.use_ddp:
            print(f"Note: {world_size} GPUs detected but DDP is disabled. Using single GPU.")
        _train_single(0, 1, config)


if __name__ == "__main__":
    main()
