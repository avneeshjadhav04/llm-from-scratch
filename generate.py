"""Text generation script for LLM-from-scratch."""
import os
import argparse
import torch
from config import get_config, add_config_args, apply_config_overrides
from data.tokenizer import BPETokenizer
from model.transformer import Transformer
from utils.training import CheckpointManager
from utils.sampling import generate_text


def main():
    parser = argparse.ArgumentParser(description="Generate text from LLM")
    add_config_args(parser)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence is", help="Generation prompt")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=1.0)
    args = parser.parse_args()

    config = get_config(args.config)
    config = apply_config_overrides(config, args)

    device = config.device if torch.cuda.is_available() else "cpu"
    config.device = device

    # Override generation params from CLI if provided
    max_new_tokens = args.max_new_tokens or config.max_new_tokens
    temperature = args.temperature or config.temperature
    top_k = args.top_k or config.top_k

    # Load tokenizer
    tokenizer = BPETokenizer(vocab_size=config.vocab_size)
    tokenizer.load(os.path.join(config.data_dir, "tokenizer"))

    # Create model and load checkpoint
    model = Transformer(config).to(device)
    checkpoint_manager = CheckpointManager(config.checkpoint_dir)
    checkpoint_manager.load(args.checkpoint, model, device=device)

    print(f"Prompt: {args.prompt}")
    print("-" * 60)

    output = generate_text(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=args.top_p,
        device=device,
    )

    print(output)
    print("-" * 60)


if __name__ == "__main__":
    main()
