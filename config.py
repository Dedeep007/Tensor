"""
OPTIMUS Configuration
=====================
Architecture and training hyperparameters for the OPTIMUS causal language model.
Inspired by Qwen2, DeepSeek, and Baichuan architectures.
"""

import os


class OptimusConfig:
    """Model architecture configuration."""

    def __init__(self):
        # --- Tokenizer / Vocabulary ---
        self.vocab_size = 151936            # Qwen2.5 exact vocab size
        self.pad_token_id = 151643          # Qwen <|endoftext|>
        self.eos_token_id = 151645          # Qwen <|im_end|>

        # --- Transformer Core ---
        self.hidden_size = 512              # Embedding dimension (scaled down for 6GB VRAM)
        self.intermediate_size = 1408       # SwiGLU hidden dimension
        self.num_hidden_layers = 8          # Number of transformer blocks
        self.num_attention_heads = 8        # Query heads
        self.num_key_value_heads = 2        # KV heads for Grouped Query Attention
        self.head_dim = self.hidden_size // self.num_attention_heads  # 64

        # --- Positional Encoding ---
        self.max_position_embeddings = 4096 # Maximum sequence length supported
        self.rope_theta = 10000.0           # RoPE base frequency

        # --- Mixture of Experts (MoE) ---
        self.use_moe = True                 # Enable Mixture of Experts
        self.num_experts = 4                # Total routed experts (scaled down)
        self.num_experts_per_tok = 2        # Top-K experts activated per token
        self.num_shared_experts = 1         # Shared experts
        self.expert_intermediate_size = 704 # Per-expert FFN hidden dim
        self.router_aux_loss_coef = 0.01    # Load-balancing auxiliary loss coefficient
        self.router_z_loss_coef = 0.001     # Router z-loss coefficient for stability
        self.moe_layer_freq = 1             # Apply MoE every N layers (1 = all layers)

        # --- Normalization ---
        self.rms_norm_eps = 1e-6            # RMSNorm epsilon

        # --- Regularization ---
        self.dropout = 0.1

    def __repr__(self):
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"OptimusConfig({params})"


class TrainingConfig:
    """Training pipeline configuration."""

    def __init__(self):
        # --- FineWeb-Edu Dataset ---
        self.use_fineweb = True           # Disabled due to severe network streaming latency
        self.fineweb_name = "HuggingFaceFW/fineweb-edu"  # Educational subset (1.3T tokens)
        self.fineweb_subset = "CC-MAIN-2013-20" # Config: "sample-10BT", "sample-100BT", or "default"
        self.val_samples = 64               # Number of pre-fetched validation sequences

        is_modal = os.environ.get("OPTIMUS_MODAL") == "1"

        # --- Data ---
        self.epochs_per_chunk = 1 if is_modal else 5        # Number of times to train on the same chunk before sliding
        self.max_seq_length = 1024                          # Keep sequence length consistent across cloud/local to save VRAM
        self.train_split = 0.9                              # Train/Val split (fallback mode only)
        self.num_workers = 4 if is_modal else 0             # Use CPU workers on Modal
        self.pin_memory = True                              # Enable pin memory for faster GPU transfer

        # --- Optimization ---
        self.batch_size = 4 if is_modal else 1              # 4x micro-batch size (fits comfortably in 24GB VRAM)
        self.gradient_accumulation_steps = 32 if is_modal else 128 # Effective batch size = 128
        self.learning_rate = 3e-4                           # Peak learning rate
        self.min_lr = 1e-5                  # Minimum LR for cosine decay
        self.weight_decay = 0.1             # AdamW weight decay
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_eps = 1e-8
        self.max_grad_norm = 1.0            # Gradient clipping norm

        # --- Schedule (step-based for streaming) ---
        self.max_train_steps = 10_000_000   # Massive step count for continual learning
        self.warmup_steps = 500             # Linear warmup steps

        # --- Mixed Precision ---
        self.use_amp = True                 # Enable Automatic Mixed Precision
        self.amp_dtype = "bfloat16"         # "bfloat16" or "float16"

        # --- torch.compile ---
        self.use_compile = False            # Disabled: Triton-Windows lacks launch_enter_hook required by PyTorch Inductor
        self.compile_mode = "default"

        # --- Validation & Checkpointing ---
        self.val_every_n_steps = 50        # Validate every 500 optimizer steps
        self.patience = 150                  # Very high patience so it doesn't early stop randomly on chunks
        self.checkpoint_dir = os.environ.get("OPTIMUS_CHECKPOINT_DIR", os.path.join(os.path.dirname(__file__), "checkpoints"))

        # --- Logging ---
        self.log_every_n_steps = 10         # Log metrics every N optimizer steps

    def __repr__(self):
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"TrainingConfig({params})"
