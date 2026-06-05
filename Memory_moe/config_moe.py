"""
OPTIMUS_moe Configuration
==========================
Architecture and training hyperparameters for the Memory MoE extension
built on top of the frozen OPTIMUS base model.
"""

import os


class MemoryMoEConfig:
    """Memory MoE architecture configuration."""

    def __init__(self):
        # --- Base Model (frozen) ---
        self.base_hidden_size = 512         # Must match OptimusConfig.hidden_size
        self.base_checkpoint = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "checkpoints", "best_model.pt"
        )

        # --- Memory MoE ---
        self.num_memory_experts = 8         # 8-way Memory MoE
        self.experts_per_token = 2          # Top-2 routing
        self.expert_hidden_size = 512       # Same as base hidden_size
        self.expert_intermediate_size = 1024  # SwiGLU FFN inner dim per expert (~1.57M params/expert)

        # --- Memory Banks ---
        self.memory_bank_size = 8192        # Slots per expert memory bank
        self.memory_heads = 4               # Cross-attention heads for memory retrieval
        self.memory_key_dim = 128           # Dimension of memory address keys (asymmetrical)
        self.memory_softmax_temp = 0.1           # Temperature for memory retrieval softmax (sharpening)
        self.memory_value_dim = 512         # Dimension of memory content values

        # --- Router ---
        self.router_aux_loss_coef = 0.01    # Load-balancing auxiliary loss (updated to 0.01)
        self.router_z_loss_coef = 0.001     # Z-loss for stability

        # --- Expert Diversity & Memory Entropy ---
        self.diversity_loss_coef = 0.0      # Disabled FFN similarity diversity
        self.memory_loss_coef = 0.005       # Maximizes memory usage entropy (peak weight during warmup)
        self.gate_loss_coef = 0.001         # L2 regularization for gate magnitude
        self.gate_bias = 0.5                # Base contribution for memory gate (α = bias + σ(g))
        # Memory entropy schedule
        self.memory_entropy_warmup_steps = 2000   # Steps with full entropy loss
        self.memory_entropy_decay_steps = 8000    # Linear decay to zero by step 10000

        # --- Normalization ---
        self.rms_norm_eps = 1e-6

        # --- Expert Labels (for logging/identification only) ---
        self.expert_labels = [
            "Science", "History", "Coding", "General",
            "Math", "Language", "Logic", "Creative"
        ]

    def __repr__(self):
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"MemoryMoEConfig({params})"


class MoETrainingConfig:
    """Training configuration for the Memory MoE extension."""

    def __init__(self):
        # --- FineWeb-Edu Dataset ---
        self.use_fineweb = False
        self.fineweb_name = "HuggingFaceFW/fineweb-edu"
        self.fineweb_subset = "CC-MAIN-2013-20"
        self.val_samples = 64
        self.samples_per_chunk = 100_000    # 1 lakh samples per chunk

        # --- Data ---
        self.epochs_per_chunk = 3           # Epochs per chunk before sliding
        self.max_seq_length = 1024          # Sequence length
        self.train_split = 0.9
        self.num_workers = 0                # Windows: no multiprocessing workers
        self.pin_memory = True

        # --- Optimization ---
        self.batch_size = 1                 # Micro-batch (6GB VRAM constraint)
        self.gradient_accumulation_steps = 128  # Effective batch = 128
        self.learning_rate = 5e-4           # Higher LR for new params only
        self.min_lr = 1e-5
        self.weight_decay = 0.1
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_eps = 1e-8
        self.max_grad_norm = 1.0

        # --- Schedule ---
        self.max_train_steps = 10_000_000
        self.warmup_steps = 200

        # --- Mixed Precision ---
        self.use_amp = True
        self.amp_dtype = "bfloat16"

        # --- torch.compile ---
        self.use_compile = False
        self.compile_mode = "default"

        # --- Validation & Checkpointing ---
        self.val_every_n_steps = 50
        self.patience = 150
        self.checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")

        # --- Logging ---
        self.log_every_n_steps = 10

    def __repr__(self):
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"MoETrainingConfig({params})"
