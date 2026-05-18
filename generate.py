"""Text generation script for LLM-from-scratch."""
import os
import argparse
import glob
import torch
from config import Config
from data.tokenizer import Tokenizer
from model.transformer import Transformer
from utils.training import CheckpointManager
from utils.sampling import generate_text


def main():
    parser = argparse.ArgumentParser(description="Generate text from LLM")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file (auto-detects latest if not provided)")
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence is", help="Generation prompt")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--device", type=str, default=None, help="Override device")
    parser.add_argument("--max_seq_len", type=int, default=None, help="Override max sequence length")
    args = parser.parse_args()

    config = Config()
    if args.device is not None:
        config.device = args.device
    if args.max_seq_len is not None:
        config.max_seq_len = args.max_seq_len

    device = config.device if torch.cuda.is_available() else "cpu"
    config.device = device

    # Override generation params
    max_new_tokens = args.max_new_tokens or config.max_new_tokens
    temperature = args.temperature or config.temperature
    top_k = args.top_k or config.top_k
    top_p = args.top_p

    # Load tokenizer
    tokenizer = Tokenizer()

    # Create model
    model = Transformer(config).to(device)

    # Load checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        ckpts = sorted(glob.glob("checkpoints/*.pt"))
        if ckpts:
            checkpoint_path = ckpts[-1]
        else:
            raise FileNotFoundError("No checkpoints found. Train first.")

    checkpoint_manager = CheckpointManager(config.checkpoint_dir)
    checkpoint_manager.load(checkpoint_path, model, device=device)

    print(f"Prompt: {args.prompt}")
    print("-" * 60)

    output = generate_text(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device,
    )

    print(output)
    print("-" * 60)


if __name__ == "__main__":
    main()
