"""
OPTIMUS Training Pipeline
==========================
Compute-optimized training with:
    - HuggingFace FineWeb-Edu streaming dataset (1.3T tokens)
    - torch.compile (kernel fusion via Dynamo + Inductor)
    - Automated Mixed Precision (AMP) with bfloat16/float16
    - Gradient accumulation for effective large batch sizes
    - Linear Warmup + Cosine Annealing LR schedule
    - Early stopping on validation loss
    - Best & periodic checkpoint saving
    - Step-level logging (loss, LR, tokens/sec, GPU memory)

Usage:
    conda activate EPOCH
    python train.py
"""

import os
import sys
import time
import math
import json
import argparse
import datetime
from dataset import get_tokenizer, build_dataloaders
from config import OptimusConfig
from configv2 import TrainingConfigV2
from model import OptimusForCausalLM
from scheduler import build_scheduler



# Fix Windows console encoding for emoji logging
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Force tqdm to use \r by pretending we are in a TTY
sys.stderr.isatty = lambda: True
sys.stdout.isatty = lambda: True

import torch
from torch.amp import GradScaler, autocast


# Parse optional command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='OPTIMUS training script')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='Path to checkpoint directory (overrides config)')
    parser.add_argument('--data_file', type=str, default=None, help='Path to a JSONL data file for training (overrides default dataset)')
    parser.add_argument('--val_file', type=str, default=None, help='Path to a JSONL data file for validation')
    return parser.parse_args()

args = parse_args()



# ===========================================================================
# Utility Functions
# ===========================================================================

def get_device(use_tpu=False):
    """Detect best available device."""
    if use_tpu:
        try:
            import torch_xla
            # Using the modern, non-deprecated syntax
            device = torch_xla.device()
            print(f"Successfully connected to: {device}")
            return device
        except ImportError:
            print("⚠️  torch_xla not installed. Falling back to GPU/CPU.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🔥 CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return device
    else:
        print("⚠️  No CUDA device found. Training on CPU (this will be very slow).")
        return torch.device("cpu")


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_num(n):
    """Format large numbers: 1234567 -> '1.23M'."""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def save_checkpoint(model, optimizer, scheduler, scaler, global_step,
                    best_val_loss, val_loss, config, training_config, filepath):
    """Save a training checkpoint atomically."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "val_loss": val_loss,
        "model_config": config.__dict__,
        "training_config": {k: v for k, v in training_config.__dict__.items()},
    }

    # Save GradScaler state if it exists and is enabled
    if scaler is not None and scaler.is_enabled():
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    # Atomic save: write to temp file first, then rename
    tmp_path = filepath + ".tmp"
    if training_config.use_tpu:
        import torch_xla.core.xla_model as xm
        xm.save(checkpoint, tmp_path)
    else:
        torch.save(checkpoint, tmp_path)
    
    # Windows sometimes locks massive files for 10-30 seconds while scanning (Antivirus / OneDrive)
    import time
    for attempt in range(60):
        try:
            # Try to safely remove the old file first to break any soft locks
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except PermissionError:
                    pass
            os.replace(tmp_path, filepath)
            break
        except PermissionError:
            time.sleep(1)
            if attempt == 59:
                raise


def load_checkpoint(filepath, model, optimizer=None, scheduler=None, scaler=None):
    """Load a training checkpoint."""
    checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        state_dict = checkpoint["optimizer_state_dict"]
        # Check for 32-bit vs 8-bit state mismatch to prevent crashes on step()
        mismatch = False
        if "state" in state_dict and len(state_dict["state"]) > 0:
            first_state = next(iter(state_dict["state"].values()))
            is_8bit_state = "state1" in first_state
            is_8bit_optim = "bitsandbytes" in optimizer.__class__.__module__
            
            if is_8bit_optim and not is_8bit_state:
                print("   ⚠️  Switched to 8-bit optimizer, but checkpoint has 32-bit state. Starting optimizer fresh.")
                mismatch = True
            elif not is_8bit_optim and is_8bit_state:
                print("   ⚠️  Switched to 32-bit optimizer, but checkpoint has 8-bit state. Starting optimizer fresh.")
                mismatch = True

        if not mismatch:
            try:
                optimizer.load_state_dict(state_dict)
            except Exception as e:
                print(f"   ⚠️  Could not load optimizer state ({e}). Starting optimizer fresh.")

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint


# ===========================================================================
# Validation
# ===========================================================================

@torch.no_grad()
def validate(model, val_loader, device, amp_dtype):
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, batch in enumerate(val_loader):
        if i >= 50:
            break
            
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        autocast_device = "cuda" if device.type == "cuda" else ("xla" if device.type == "xla" else "cpu")
        with autocast(autocast_device, dtype=amp_dtype, enabled=(device.type in ["cuda", "xla"])):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs["loss"]
        # Count non-padding tokens in labels for proper averaging
        num_tokens = (labels[..., 1:] != -100).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    model.train()
    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss


# ===========================================================================
# Main Training Loop (Step-Based for Streaming)
# ===========================================================================

def train():
    """Main training entry point."""

    # ---- Setup ----
    print("=" * 70)
    print("  ⚡ OPTIMUS Training Pipeline")
    print("  📚 Dataset: FineWeb-Edu (Educational Web Corpus)")
    print("=" * 70)
    print()

    model_config = OptimusConfig()
    train_config = TrainingConfigV2()
    # Override checkpoint directory if supplied via command line
    if args.ckpt_dir:
        train_config.checkpoint_dir = args.ckpt_dir
        os.makedirs(train_config.checkpoint_dir, exist_ok=True)
    device = get_device(train_config.use_tpu)

    # Determine AMP dtype
    if train_config.use_amp and device.type == "cuda":
        if train_config.amp_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            use_scaler = False  # bfloat16 doesn't need GradScaler
            print("⚡ AMP: bfloat16 (no GradScaler needed)")
        else:
            amp_dtype = torch.float16
            use_scaler = True
            print("⚡ AMP: float16 (with GradScaler)")
    elif train_config.use_amp and device.type == "xla":
        amp_dtype = torch.bfloat16
        use_scaler = False
        print("⚡ AMP: bfloat16 natively on TPU (no GradScaler needed)")
    else:
        amp_dtype = torch.float32
        use_scaler = False
        print("⚡ AMP: disabled (using float32)")

    # Enable TF32 on Ampere+ GPUs for faster matmuls
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    # ---- Tokenizer & Data ----
    print("\n📦 Loading tokenizer and data...")
    tokenizer = get_tokenizer()
    # ---------------------------------------------------------------------
    # Data loading – use provided JSONL file if given, otherwise default loader
    # ---------------------------------------------------------------------
    if args.data_file and os.path.exists(args.data_file):
        import json
        from torch.utils.data import Dataset, DataLoader
        class JsonlDataset(Dataset):
            """Simple dataset that reads a JSONL file containing raw text, tokenizes it, 
            and packs tokens into sequences of max_seq_length with no padding waste."""
            def __init__(self, path, tokenizer, max_length):
                import concurrent.futures
                import os
                
                self.samples = []
                token_buffer = []
                eos = tokenizer.eos_token_id
                
                print(f"Reading data from {path}...")
                with open(path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                print(f"Tokenizing {len(lines)} documents using multiple threads...")
                
                def process_line(line):
                    if not line.strip(): return []
                    try:
                        sample = json.loads(line)
                        text = sample.get('text', '').strip()
                        if len(text) > 10:
                            # Add truncation to gracefully handle anomaly documents with 100k+ tokens
                            return tokenizer.encode(
                                text, 
                                add_special_tokens=False,
                                truncation=True,
                                max_length=100000
                            ) + [eos]
                    except:
                        pass
                    return []

                # Using ThreadPoolExecutor works great because HuggingFace tokenizers (Rust backend) release the Python GIL
                with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                    results = executor.map(process_line, lines)
                    
                print(f"Packing tokens into sequences of length {max_length}...")
                for token_ids in results:
                    if token_ids:
                        token_buffer.extend(token_ids)
                        while len(token_buffer) >= max_length:
                            self.samples.append(token_buffer[:max_length])
                            token_buffer = token_buffer[max_length:]

            def __len__(self):
                return len(self.samples)
                
            def __getitem__(self, idx):
                chunk = self.samples[idx]
                input_ids = torch.tensor(chunk, dtype=torch.long)
                attention_mask = torch.ones_like(input_ids)
                labels = input_ids.clone()
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels,
                }
        if args.val_file and os.path.exists(args.val_file):
            train_dataset = JsonlDataset(args.data_file, tokenizer, train_config.max_seq_length)
            val_dataset = JsonlDataset(args.val_file, tokenizer, train_config.max_seq_length)
            
            train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size,
                                      shuffle=True, num_workers=train_config.num_workers,
                                      pin_memory=train_config.pin_memory)
            val_loader = DataLoader(val_dataset, batch_size=train_config.batch_size,
                                    shuffle=False, num_workers=train_config.num_workers,
                                    pin_memory=train_config.pin_memory)
        else:
            json_dataset = JsonlDataset(args.data_file, tokenizer, train_config.max_seq_length)
            
            # Properly split the dataset to avoid training/validation leak
            total_len = len(json_dataset)
            train_len = int(total_len * train_config.train_split)
            val_len = total_len - train_len
            train_ds, val_ds = torch.utils.data.random_split(json_dataset, [train_len, val_len])
            
            train_loader = DataLoader(train_ds, batch_size=train_config.batch_size,
                                      shuffle=True, num_workers=train_config.num_workers,
                                      pin_memory=train_config.pin_memory)
            # Use the split subset for validation
            val_loader = DataLoader(val_ds, batch_size=train_config.batch_size,
                                    shuffle=False, num_workers=train_config.num_workers,
                                    pin_memory=train_config.pin_memory)
    else:
        train_loader, val_loader = build_dataloaders(
            tokenizer, train_config, model_config
        )

    # ---- Model ----
    print("\n🏗️  Building OPTIMUS model...")
    model = OptimusForCausalLM(model_config).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"   Total parameters:     {format_num(total_params)} ({total_params:,})")
    print(f"   Trainable parameters: {format_num(trainable_params)} ({trainable_params:,})")

    # MoE info
    if model_config.use_moe:
        active_params = model.get_num_params(count_active=True)
        moe_layers = sum(1 for layer in model.layers if layer.is_moe)
        print(f"   Active params/token:  {format_num(active_params)} ({active_params:,})")
        print(f"   MoE layers:           {moe_layers}/{model_config.num_hidden_layers}")
        print(f"   Experts:              {model_config.num_experts} routed (top-{model_config.num_experts_per_tok}) + {model_config.num_shared_experts} shared")
        print(f"   Expert FFN dim:       {model_config.expert_intermediate_size}")
        print(f"   Router aux coef:      {model_config.router_aux_loss_coef}")
        print(f"   Router z-loss coef:   {model_config.router_z_loss_coef}")

    # ---- Optimizer ----
    try:
        import bitsandbytes as bnb
        print("\n✨ Using 8-bit AdamW optimizer from bitsandbytes (saves ~75% optimizer VRAM)")
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=train_config.learning_rate,
            betas=(train_config.adam_beta1, train_config.adam_beta2),
            eps=train_config.adam_eps,
            weight_decay=train_config.weight_decay,
        )
    except ImportError:
        print("\n⚠️  bitsandbytes not installed. Falling back to standard 32-bit AdamW.")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config.learning_rate,
            betas=(train_config.adam_beta1, train_config.adam_beta2),
            eps=train_config.adam_eps,
            weight_decay=train_config.weight_decay,
        )

    # ---- Scheduler ----
    total_steps = train_config.max_train_steps
    scheduler = build_scheduler(optimizer, train_config, total_steps)

    # Compute training token budget
    tokens_per_step = (
        train_config.batch_size
        * train_config.gradient_accumulation_steps
        * train_config.max_seq_length
    )
    total_token_budget = total_steps * tokens_per_step

    print(f"\n📊 Training Configuration:")
    print(f"   Micro-batch size:     {train_config.batch_size}")
    print(f"   Gradient accumulation: {train_config.gradient_accumulation_steps}")
    print(f"   Effective batch size: {train_config.batch_size * train_config.gradient_accumulation_steps}")
    print(f"   Sequence length:      {train_config.max_seq_length}")
    print(f"   Tokens/step:          {format_num(tokens_per_step)}")
    print(f"   Total steps:          {format_num(total_steps)}")
    print(f"   Token budget:         {format_num(total_token_budget)}")
    print(f"   Warmup steps:         {train_config.warmup_steps}")
    print(f"   Peak LR:              {train_config.learning_rate}")
    print(f"   Min LR:               {train_config.min_lr}")
    print(f"   Max grad norm:        {train_config.max_grad_norm}")
    print(f"   Val every:            {train_config.val_every_n_steps} steps")
    print(f"   Early stopping:       patience={train_config.patience}")

    # ---- GradScaler (only for float16) ----
    scaler = GradScaler("cuda", enabled=use_scaler) if device.type == "cuda" else GradScaler(enabled=False)

    # ---- Checkpoint directory ----
    os.makedirs(train_config.checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(train_config.checkpoint_dir, "best_model.pt")
    latest_ckpt_path = os.path.join(train_config.checkpoint_dir, "latest_checkpoint.pt")

    # ---- Training State ----
    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0

    if os.path.exists(latest_ckpt_path):
        print(f"\n📂 Resuming from checkpoint: {latest_ckpt_path}")
        ckpt = load_checkpoint(latest_ckpt_path, model, optimizer, scheduler, scaler)
        
        # FIX: PyTorch lr_scheduler.load_state_dict overwrites instance variables.
        # If we extended max_train_steps in config, the checkpoint reverts it!
        # Re-apply the current config values here.
        scheduler.total_steps = train_config.max_train_steps
        scheduler.warmup_steps = train_config.warmup_steps
        
        global_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"   Resumed at step: {global_step} | Best val loss: {best_val_loss:.4f}")
        
    # ---- torch.compile ----
    if train_config.use_compile and device.type == "cuda":
        # Check if Triton is available (required by PyTorch's default Inductor backend)
        triton_available = False
        try:
            import triton
            triton_available = True
        except ImportError:
            pass

        if not triton_available:
            print("\n⚠️  torch.compile skipped: Triton installation not found.")
            print("   Triton is required for torch.compile with the default Inductor backend on GPU.")
            if sys.platform == "win32":
                print("   Note: Triton does not officially support Windows natively. Running without compilation.")
        else:
            print(f"\n🔧 Compiling model with torch.compile(mode='{train_config.compile_mode}')...")
            print("   (First few iterations will be slow due to compilation)")
            try:
                model = torch.compile(model, mode=train_config.compile_mode)
            except Exception as e:
                print(f"⚠️  Failed to compile model: {e}. Running uncompiled.")
    elif train_config.use_compile and device.type != "cuda":
        print("\n⚠️  torch.compile skipped (requires CUDA)")
        
    accumulated_loss = 0.0
    accumulated_aux_loss = 0.0
    micro_step = 0
    tokens_processed = 0
    start_time = time.time()
    stop_training = False

    print(f"\n{'=' * 70}")
    print(f"  🚀 Starting Training — {format_num(total_steps)} optimizer steps")
    print(f"  📚 Streaming from FineWeb-Edu ({train_config.fineweb_subset})")
    print(f"{'=' * 70}\n")

    model.train()

    # -------------------------------------------------------------------------
    # Step-based training loop for streaming datasets.
    #
    # Since FineWeb-Edu is streamed (no fixed size), we don't use traditional
    # epochs. Instead, we iterate through the infinite stream and count
    # optimizer steps. The DataLoader restarts the stream automatically
    # if it reaches the end of the buffer.
    # -------------------------------------------------------------------------
    train_iter = iter(train_loader)
    current_epoch = 1
    
    try:
        while global_step < total_steps and not stop_training:
            # Get next batch from stream (restart if exhausted)
            try:
                batch = next(train_iter)
            except StopIteration:
                if current_epoch < train_config.epochs_per_chunk:
                    print(f"\n  🔄 Epoch {current_epoch}/{train_config.epochs_per_chunk} finished. Restarting current chunk data...")
                    train_iter = iter(train_loader)
                    current_epoch += 1
                    continue
                else:
                    # Stream buffer exhausted — end of chunk!
                    print(f"\n  📦 Reached end of current data chunk (Chunk finished at step {global_step} after {current_epoch} epochs).")
                    # Save the exact progress before exiting so the next chunk resumes seamlessly
                    save_checkpoint(
                        model, optimizer, scheduler, scaler,
                        global_step, best_val_loss, best_val_loss,
                        model_config, train_config, latest_ckpt_path
                    )
                    break
    
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
    
            # Count tokens for throughput measurement
            batch_tokens = (labels[..., 1:] != -100).sum().item()
            tokens_processed += batch_tokens
    
            # ---- Forward pass with AMP ----
            autocast_device = "cuda" if device.type == "cuda" else ("xla" if device.type == "xla" else "cpu")
            with autocast(autocast_device, dtype=amp_dtype, enabled=(train_config.use_amp and device.type in ["cuda", "xla"])):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"]
                # Scale loss for gradient accumulation
                loss = loss / train_config.gradient_accumulation_steps
    
            # ---- Backward pass ----
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            accumulated_loss += loss.item()
            if "aux_loss" in outputs and outputs["aux_loss"] is not None:
                accumulated_aux_loss += outputs["aux_loss"].item() / train_config.gradient_accumulation_steps
            micro_step += 1
    
            # ---- Optimizer step (every gradient_accumulation_steps) ----
            if micro_step % train_config.gradient_accumulation_steps == 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), train_config.max_grad_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), train_config.max_grad_norm
                    )
                    if device.type == "xla":
                        import torch_xla.core.xla_model as xm
                        xm.optimizer_step(optimizer, barrier=True)
                    else:
                        optimizer.step()
                        
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
    
                global_step += 1
    
                # ---- Logging ----
                if global_step % train_config.log_every_n_steps == 0:
                    elapsed = time.time() - start_time
                    current_lr = scheduler.get_last_lr()[0]
                    tps = tokens_processed / elapsed  # tokens per second
    
                    # GPU memory
                    if device.type == "cuda":
                        mem_used = torch.cuda.max_memory_allocated() / 1e9
                        mem_str = f" | GPU: {mem_used:.1f}GB"
                    else:
                        mem_str = ""
    
                    progress = global_step / total_steps * 100
                    aux_str = f" | Aux: {accumulated_aux_loss:.4f}" if model_config.use_moe else ""
                    
                    print(f"Step {global_step}/{total_steps} ({progress:.1f}%) | Loss: {accumulated_loss:.4f}{aux_str} | LR: {current_lr:.2e} | Tok/s: {format_num(int(tps))}{mem_str}")
    
                accumulated_loss = 0.0
                accumulated_aux_loss = 0.0
    
                # ---- Validation ----
                if global_step % train_config.val_every_n_steps == 0:
                    print(f"\n  📝 Validation at step {global_step}...")
                    val_loss = validate(model, val_loader, device, amp_dtype)
                    val_ppl = math.exp(min(val_loss, 20))  # Cap to avoid overflow
                    print(f"     Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
    
                    # Save latest checkpoint
                    save_checkpoint(
                        model, optimizer, scheduler, scaler,
                        global_step, best_val_loss, val_loss,
                        model_config, train_config, latest_ckpt_path
                    )
                    print(f"     💾 Saved latest checkpoint → {latest_ckpt_path}")
    
                    # Check for best model
                    if val_loss < best_val_loss:
                        improvement = best_val_loss - val_loss
                        best_val_loss = val_loss
                        patience_counter = 0
                        save_checkpoint(
                            model, optimizer, scheduler, scaler,
                            global_step, best_val_loss, val_loss,
                            model_config, train_config, best_ckpt_path
                        )
                        print(f"     🏆 New best model! (improved by {improvement:.4f})")
                    else:
                        patience_counter += 1
                        print(f"     ⏳ No improvement ({patience_counter}/{train_config.patience})")
    
                    # Early stopping
                    if patience_counter >= train_config.patience:
                        print(f"\n  🛑 Early stopping triggered after {patience_counter} checks without improvement.")
                        print(f"     Best validation loss: {best_val_loss:.4f}")
                        stop_training = True
                        break
    
                    print()  # Blank line after validation
                    model.train()

    except KeyboardInterrupt:
        print("\n\n🛑 Training interrupted by user. Saving current progress...")
        save_checkpoint(
            model, optimizer, scheduler, scaler,
            global_step, best_val_loss, best_val_loss,
            model_config, train_config, latest_ckpt_path
        )
        print(f"   💾 Saved latest checkpoint → {latest_ckpt_path}")
        print("   You can safely resume later. Goodbye!\n")
        sys.exit(0)

    # ---- Training Complete ----
    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"  ✅ Training Complete!")
    print(f"{'=' * 70}")
    print(f"  Total time:        {datetime.timedelta(seconds=int(total_time))}")
    print(f"  Total steps:       {global_step}")
    print(f"  Tokens processed:  {format_num(tokens_processed)}")
    print(f"  Best val loss:     {best_val_loss:.4f}")
    print(f"  Best val PPL:      {math.exp(min(best_val_loss, 20)):.2f}")
    print(f"  Best checkpoint:   {best_ckpt_path}")
    print(f"  Latest checkpoint: {latest_ckpt_path}")
    print()


# ===========================================================================
# Entry Point
# ===========================================================================

if __name__ == "__main__":
    train()
