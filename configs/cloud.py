from config import Config

config = Config(
    # Model architecture (~45M params)
    vocab_size=10000,
    d_model=384,
    n_layers=6,
    n_heads=6,
    d_ff=1536,
    max_seq_len=256,
    dropout=0.1,

    # Training (optimized for T4 16GB)
    batch_size=16,
    grad_accum_steps=2,
    learning_rate=3e-4,
    weight_decay=0.01,
    max_steps=50000,
    warmup_steps=500,
    eval_interval=1000,
    eval_iters=100,
    checkpoint_interval=2000,

    # Data
    train_split=0.9,
    num_tokens_to_train=100_000_000,

    # Generation
    max_new_tokens=512,
    temperature=0.8,
    top_k=50,

    # System
    device="cuda",
    dtype="float16",
    compile_model=True,  # torch.compile works great on T4

    # Paths
    data_dir="data",
    checkpoint_dir="checkpoints",
    log_dir="logs",
    dataset_name="wikitext-2",
)
