from config import Config

config = Config(
    # Model architecture (~2M params)
    vocab_size=5000,
    d_model=128,
    n_layers=4,
    n_heads=4,
    d_ff=512,
    max_seq_len=128,
    dropout=0.1,

    # Training (memory-optimized for GTX 1650 4GB)
    batch_size=2,
    grad_accum_steps=8,
    learning_rate=5e-4,
    weight_decay=0.01,
    max_steps=10000,
    warmup_steps=100,
    eval_interval=500,
    eval_iters=50,
    checkpoint_interval=1000,

    # Data
    train_split=0.9,
    num_tokens_to_train=10_000_000,

    # Generation
    max_new_tokens=256,
    temperature=0.8,
    top_k=40,

    # System
    device="cuda",
    dtype="float16",
    compile_model=False,  # torch.compile may not work well on older GPUs

    # Paths
    data_dir="data",
    checkpoint_dir="checkpoints",
    log_dir="logs",
    dataset_name="wikitext-2",
)
