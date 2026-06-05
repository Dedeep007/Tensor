import os

class TrainingConfigV2:
    """Combined configuration for continual training pipeline v2, including all features from the original config.
    
    This class merges dataset‑specific settings with the full set of training hyper‑parameters
    and utilities originally defined in `config.py`.
    """

    def __init__(self):
        # ---------------------------------------------------------------------
        # Dataset settings (v2 specific)
        # ---------------------------------------------------------------------
        self.dataset_name = "HuggingFaceTB/smollm-corpus"
        self.dataset_config = "fineweb-edu-dedup"
        self.split = "train"

        # ---------------------------------------------------------------------
        # Chunk / sliding‑window settings
        # ---------------------------------------------------------------------
        self.samples_per_chunk = 100_000          # 1 lakh documents per chunk
        self.overlap_ratio = 0.0                  # 20 % overlap between chunks

        # ---------------------------------------------------------------------
        # Checkpoint & logging (uses the renamed folder "chkpoints_v2")
        # ---------------------------------------------------------------------
        self.checkpoint_dir = os.path.join(os.getcwd(), "chkpoints_v2")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.log_file = os.path.join(self.checkpoint_dir, "training.log")

        # ---------------------------------------------------------------------
        # Training hyper‑parameters (mirrored from the original TrainingConfig)
        # ---------------------------------------------------------------------
        self.batch_size = 1                      # micro‑batch size per step
        self.gradient_accumulation_steps = 128
        self.learning_rate = 3e-4
        self.min_lr = 1e-5
        self.weight_decay = 0.1
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_eps = 1e-8
        self.max_grad_norm = 1.0

        # ---------------------------------------------------------------------
        # Mixed‑precision, hardware, and compilation flags
        # ---------------------------------------------------------------------
        self.use_tpu = False
        self.use_amp = True
        self.amp_dtype = "bfloat16"  # bfloat20 is a typo; correct to "bfloat16"
        self.use_compile = False
        self.compile_mode = "default"

        # ---------------------------------------------------------------------
        # Validation, checkpointing and logging frequencies
        # ---------------------------------------------------------------------
        self.val_every_n_steps = 500
        self.patience = 150
        self.log_every_n_steps = 10

        # ---------------------------------------------------------------------
        # Original FineWeb‑Edu related settings (kept for compatibility)
        # ---------------------------------------------------------------------
        self.use_fineweb = False
        self.fineweb_name = "HuggingFaceFW/fineweb-edu"
        self.fineweb_subset = "CC-MAIN-2013-20"
        self.val_samples = 64

        # ---------------------------------------------------------------------
        # Data handling parameters
        # ---------------------------------------------------------------------
        self.epochs_per_chunk = 1
        self.max_seq_length = 1024
        self.train_split = 0.9
        self.num_workers = 0
        self.pin_memory = True

        # ---------------------------------------------------------------------
        # Scheduler / step‑based training schedule
        # ---------------------------------------------------------------------
        self.max_train_steps = 10_000_000
        self.warmup_steps = 500
