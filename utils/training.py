"""Training utilities: checkpointing, logging, learning rate scheduling."""
import os
import csv
import math
import time
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
from config import Config


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr=0.0):
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


class CheckpointManager:
    def __init__(self, checkpoint_dir: str, config_name: str = "default"):
        self.checkpoint_dir = checkpoint_dir
        self.config_name = config_name
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, step, model, optimizer, loss):
        # Unwrap DDP before saving so checkpoints are portable
        model_to_save = model.module if hasattr(model, 'module') else model
        checkpoint = {
            "step": step,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }
        path = os.path.join(self.checkpoint_dir, f"{self.config_name}_step_{step}.pt")
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load(self, path, model, optimizer=None, device="cpu"):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state_dict"]

        # Handle DDP module. prefix for cross-device loading
        has_module_prefix = any(k.startswith("module.") for k in state_dict)
        model_is_wrapped = hasattr(model, "module")
        if has_module_prefix and not model_is_wrapped:
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        elif not has_module_prefix and model_is_wrapped:
            state_dict = {"module." + k: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Checkpoint loaded from {path}")
        return checkpoint.get("step", 0), checkpoint.get("loss", float("inf"))


class Logger:
    def __init__(self, log_dir: str, config_name: str = "default"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, f"{config_name}_training_log.csv")
        self.fieldnames = ["step", "train_loss", "val_loss", "lr", "time_ms", "tokens_per_sec"]

        self.losses = []
        self.val_losses = []
        self.steps = []

        if os.path.exists(self.csv_path):
            # Resume: load existing metrics for plotting
            with open(self.csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.steps.append(int(row["step"]))
                    self.losses.append(float(row["train_loss"]))
                    if row.get("val_loss"):
                        self.val_losses.append(float(row["val_loss"]))
        else:
            # Create CSV with headers
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, step, train_loss, val_loss, lr, time_ms, tokens_per_sec):
        row = {
            "step": step,
            "train_loss": f"{train_loss:.4f}",
            "val_loss": f"{val_loss:.4f}" if val_loss is not None else "",
            "lr": f"{lr:.6f}",
            "time_ms": f"{time_ms:.2f}",
            "tokens_per_sec": f"{tokens_per_sec:.2f}" if tokens_per_sec is not None else "",
        }
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

        self.steps.append(step)
        self.losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)

    def plot(self, save_path: str = None):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(self.steps, self.losses, label="Train Loss")
        if self.val_losses:
            # Align val losses with their steps (simplified)
            eval_steps = self.steps[::len(self.steps)//max(len(self.val_losses), 1)][:len(self.val_losses)]
            ax.plot(eval_steps, self.val_losses, label="Val Loss", linestyle="--")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Curve")
        ax.legend()
        ax.grid(True)

        if save_path is None:
            save_path = os.path.join(self.log_dir, "loss_curve.png")
        plt.savefig(save_path)
        print(f"Loss curve saved to {save_path}")
        plt.close()


class Trainer:
    def __init__(self, model, optimizer, config: Config, checkpoint_manager: CheckpointManager, logger: Logger, rank=0, world_size=1):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.logger = logger
        self.step = 0
        self.best_val_loss = float("inf")
        self.rank = rank
        self.world_size = world_size
        self.is_master = rank == 0
        self.use_ddp = world_size > 1 and hasattr(model, 'module')

        # Mixed precision scaler — works for cuda, cuda:0, cuda:1, etc.
        self.scaler = torch.amp.GradScaler('cuda') if config.dtype == "float16" and "cuda" in config.device else None

    def train(self, train_loader, val_loader):
        self.model.train()
        train_iter = iter(train_loader)
        t0 = time.time()
        accumulated_loss = 0.0

        while self.step < self.config.max_steps:
            # Learning rate schedule
            lr = get_lr(self.step, self.config.warmup_steps, self.config.max_steps, self.config.learning_rate)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            # Gradient accumulation
            accumulated_loss = 0.0
            for micro_step in range(self.config.grad_accum_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    x, y = next(train_iter)

                x = x.to(self.config.device)
                y = y.to(self.config.device)

                # Forward pass with autocast
                with torch.amp.autocast(self.config.device, enabled=self.scaler is not None):
                    logits, loss = self.model(x, y)
                    # DDP may return a tensor of losses (one per GPU replica)
                    if loss is not None and loss.dim() > 0:
                        loss = loss.mean()
                    loss = loss / self.config.grad_accum_steps

                # Backward pass — use no_sync during gradient accumulation except last step
                is_last_micro_step = micro_step == self.config.grad_accum_steps - 1
                if self.use_ddp and not is_last_micro_step:
                    with self.model.no_sync():
                        if self.scaler is not None:
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()
                else:
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                accumulated_loss += loss.item()

            # Gradient clipping
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Optimizer step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            # Timing
            t1 = time.time()
            dt = (t1 - t0) * 1000  # ms
            t0 = t1

            tokens_per_sec = (
                self.config.batch_size * self.config.grad_accum_steps * self.config.max_seq_len
            ) / (dt / 1000)

            # Logging (master only)
            if self.step % 10 == 0 and self.is_master:
                print(f"step {self.step:5d} | loss {accumulated_loss:.4f} | lr {lr:.2e} | dt {dt:.2f}ms | tok/s {tokens_per_sec:.2f}")

            # Evaluation (master only logs)
            val_loss = None
            if self.step > 0 and self.step % self.config.eval_interval == 0:
                val_loss = self.evaluate(val_loader)
                self.model.train()
                if self.is_master and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss

            if self.is_master and self.logger is not None:
                self.logger.log(self.step, accumulated_loss, val_loss, lr, dt, tokens_per_sec)

            # Checkpointing (master only)
            if self.step > 0 and self.step % self.config.checkpoint_interval == 0 and self.is_master:
                self.checkpoint_manager.save(self.step, self.model, self.optimizer, accumulated_loss)

            self.step += 1

        # Final checkpoint and plot (master only)
        if self.is_master:
            self.checkpoint_manager.save(self.step, self.model, self.optimizer, accumulated_loss)
            if self.logger is not None:
                self.logger.plot()

    @torch.no_grad()
    def evaluate(self, val_loader):
        self.model.eval()
        losses = []
        for i, (x, y) in enumerate(val_loader):
            if i >= self.config.eval_iters:
                break
            x = x.to(self.config.device)
            y = y.to(self.config.device)
            with torch.amp.autocast(self.config.device, enabled=self.scaler is not None):
                logits, loss = self.model(x, y)
                if loss is not None and loss.dim() > 0:
                    loss = loss.mean()
            losses.append(loss.item())
        mean_loss = sum(losses) / len(losses)

        # Average loss across all DDP ranks
        if self.use_ddp:
            loss_tensor = torch.tensor([mean_loss], device=self.config.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            mean_loss = loss_tensor.item() / self.world_size

        if self.is_master:
            print(f"Validation loss: {mean_loss:.4f}")
        return mean_loss
