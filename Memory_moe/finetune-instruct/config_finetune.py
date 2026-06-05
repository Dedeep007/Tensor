"""
LoRA Instruction Fine-Tuning Configuration
=============================================
Configuration for LoRA adapters and instruction fine-tuning
of OPTIMUS MoE on Open-Orca/OpenOrca.
"""

import os


class LoRAConfig:
    """LoRA adapter configuration."""

    def __init__(self):
        self.rank = 16                      # LoRA rank (r)
        self.alpha = 32                     # LoRA scaling factor (alpha)
        self.dropout = 0.05                 # LoRA dropout
        self.target_modules = [             # Module names to apply LoRA to
            "q_proj", "k_proj", "v_proj", "o_proj",   # Attention projections (base)
            "gate_proj", "up_proj", "down_proj",       # SwiGLU FFN (base + MoE experts)
        ]

    def __repr__(self):
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"LoRAConfig({params})"


class InstructFinetuneConfig:
    """Training configuration for instruction fine-tuning."""

    def __init__(self):
        # --- Paths ---
        self.moe_checkpoint = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "checkpoints", "moe_full_best.pt"
        )
        self.output_dir = os.path.join(os.path.dirname(__file__), "checkpoints")

        # --- Dataset ---
        self.dataset_name = "Open-Orca/OpenOrca"
        self.max_samples = 100_000          # Number of samples to use (None = all)
        self.max_seq_length = 1024          # Max sequence length
        self.val_split = 0.02               # 2% validation split
        self.val_samples = 200              # Max validation samples

        # --- Optimization ---
        self.batch_size = 1                 # Micro-batch size (6GB VRAM)
        self.gradient_accumulation_steps = 64  # Effective batch = 64
        self.learning_rate = 2e-4           # Peak LR for LoRA
        self.min_lr = 1e-5                  # Min LR for cosine decay
        self.weight_decay = 0.01            # Lower weight decay for fine-tuning
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_eps = 1e-8
        self.max_grad_norm = 1.0

        # --- Schedule ---
        self.max_train_steps = 5_000        # Total optimizer steps
        self.warmup_steps = 100             # Linear warmup

        # --- Mixed Precision ---
        self.use_amp = True
        # Use float16 for Nvidia GPUs and enable GradScaler in training loop
        self.amp_dtype = "float16"

        # --- Validation & Checkpointing ---
        self.val_every_n_steps = 100
        self.save_every_n_steps = 500
        self.patience = 20                  # Early stopping patience

        # --- Logging ---
        self.log_every_n_steps = 10

        # --- Loss Masking ---
        self.mask_prompt = True             # Only compute loss on assistant response

    def __repr__(self):
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"InstructFinetuneConfig({params})"
