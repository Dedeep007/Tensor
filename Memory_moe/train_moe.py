"""
OPTIMUS_moe Training Pipeline
===============================
Train the Memory MoE layer on top of the frozen OPTIMUS base model.

Phase 1: Base frozen, only MoE parameters are trained.

Key differences from base train.py:
    1. Loads frozen base model from best checkpoint
    2. Only optimizes Memory MoE parameters (~51M)
    3. Modular checkpoint saving (router + each expert separately)
    4. Uses FineWeb-Edu streaming or local fallback data

Usage:
    conda activate EPOCH
    cd Memory_moe
    python train_moe.py
"""

import os
import sys
import time
import math
import json
import datetime

# Fix Windows console encoding for emoji logging
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# --- Dual Logger: terminal + log file ---
if os.environ.get("CONTINUAL_TRAINER_ACTIVE") != "1":
    class Logger(object):
        """Writes to both terminal and training_pipeline_moe.log."""
        def __init__(self, filename=None):
            if filename is None:
                filename = os.path.join(os.path.dirname(__file__), "training_pipeline_moe.log")
            self.terminal = sys.stdout
            self.log = open(filename, "a", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger()
    sys.stderr = sys.stdout


# Force tqdm to use \r
sys.stderr.isatty = lambda: True
sys.stdout.isatty = lambda: True

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Ensure training dataset points to Memory_moe/data/fineweb_chunk.txt
os.environ["OPTIMUS_DATA_FILE"] = os.path.join(os.path.dirname(__file__), "data", "fineweb_chunk.txt")

from config_moe import MemoryMoEConfig, MoETrainingConfig
from model_moe import (
    build_optimus_moe,
    save_moe_checkpoints,
    load_moe_checkpoint,
)

# Reuse dataset and scheduler from base OPTIMUS
from dataset import build_dataloaders, get_tokenizer
from scheduler import WarmupCosineScheduler
from eval_factual import evaluate_factual_accuracy


# ===========================================================================
# Utility Functions
# ===========================================================================

def get_device():
    """Detect best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🔥 CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return device
    else:
        print("⚠️  No CUDA device found. Training on CPU (very slow).")
        return torch.device("cpu")


def format_num(n):
    """Format large numbers: 1234567 -> '1.23M'."""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


# ===========================================================================
# Validation
# ===========================================================================

@torch.no_grad()
def validate(model, val_loader, device, amp_dtype):
    """Run validation and return average loss."""
    model.eval()
    # Re-freeze base during eval (it should already be frozen, but be safe)
    model.base_model.eval()

    total_loss = 0.0
    total_tokens = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs["loss"]
        num_tokens = (labels[..., 1:] != -100).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    model.train()
    model.base_model.eval()  # Keep base frozen even in train mode

    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss


# ===========================================================================
# Main Training Loop
# ===========================================================================

def train():
    """Main training entry point for OPTIMUS_moe."""

    print("=" * 70)
    print("  ⚡ OPTIMUS_moe Training Pipeline")
    print("  🧠 Phase 1: Frozen Base + Trainable Memory MoE")
    print("  📚 Dataset: FineWeb-Edu (Educational Web Corpus)")
    print("=" * 70)
    print()

    moe_config = MemoryMoEConfig()
    train_config = MoETrainingConfig()
    device = get_device()

    # Determine AMP dtype
    if train_config.use_amp and device.type == "cuda":
        if train_config.amp_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            use_scaler = False
            print("⚡ AMP: bfloat16 (no GradScaler needed)")
        else:
            amp_dtype = torch.float16
            use_scaler = True
            print("⚡ AMP: float16 (with GradScaler)")
    else:
        amp_dtype = torch.float32
        use_scaler = False
        print("⚡ AMP: disabled (using float32)")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    # ---- Tokenizer & Data ----
    print("\n📦 Loading tokenizer and data...")
    tokenizer = get_tokenizer()
    train_loader, val_loader = build_dataloaders(
        tokenizer, train_config, moe_config
    )

    # ---- Build Model ----
    model = build_optimus_moe(moe_config, device)

    # Verify freeze status
    total, trainable, frozen = model.get_num_params()
    print(f"\n🔒 Freeze verification:")
    print(f"   Base model frozen:  {frozen:,} params (requires_grad=False)")
    print(f"   MoE trainable:     {trainable:,} params (requires_grad=True)")

    # Double-check: ensure no base param has grad
    base_grad_count = sum(1 for p in model.base_model.parameters() if p.requires_grad)
    if base_grad_count > 0:
        print(f"   ⚠️ WARNING: {base_grad_count} base params still have requires_grad=True!")
    else:
        print(f"   ✅ All base params confirmed frozen")

    # ---- Base Factual QA Evaluation ----
    print("\n❓ Evaluating baseline factual QA accuracy on frozen base model...")
    base_factual_acc, _ = evaluate_factual_accuracy(model.base_model, tokenizer, device)
    print(f"   Base model factual QA accuracy: {base_factual_acc:.2%}")

    # ---- Optimizer (only MoE params) ----
    trainable_params = model.get_trainable_params()
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=train_config.learning_rate,
        betas=(train_config.adam_beta1, train_config.adam_beta2),
        eps=train_config.adam_eps,
        weight_decay=train_config.weight_decay,
    )

    # ---- Scheduler ----
    total_steps = train_config.max_train_steps
    min_lr_ratio = train_config.min_lr / train_config.learning_rate
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=train_config.warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=min_lr_ratio,
    )

    # ---- Print training config ----
    tokens_per_step = (
        train_config.batch_size
        * train_config.gradient_accumulation_steps
        * train_config.max_seq_length
    )

    print(f"\n📊 Training Configuration:")
    print(f"   Micro-batch size:     {train_config.batch_size}")
    print(f"   Gradient accumulation: {train_config.gradient_accumulation_steps}")
    print(f"   Effective batch size: {train_config.batch_size * train_config.gradient_accumulation_steps}")
    print(f"   Sequence length:      {train_config.max_seq_length}")
    print(f"   Tokens/step:          {format_num(tokens_per_step)}")
    print(f"   Total steps:          {format_num(total_steps)}")
    print(f"   Warmup steps:         {train_config.warmup_steps}")
    print(f"   Peak LR:              {train_config.learning_rate}")
    print(f"   Min LR:               {train_config.min_lr}")
    print(f"   Max grad norm:        {train_config.max_grad_norm}")
    print(f"   Val every:            {train_config.val_every_n_steps} steps")
    print(f"   Early stopping:       patience={train_config.patience}")

    # ---- GradScaler ----
    scaler = GradScaler("cuda", enabled=use_scaler) if device.type == "cuda" else GradScaler(enabled=False)

    # ---- Checkpoint directory ----
    os.makedirs(train_config.checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(train_config.checkpoint_dir, "moe_full_best.pt")
    latest_ckpt_path = os.path.join(train_config.checkpoint_dir, "moe_full_latest.pt")

    # ---- Training State ----
    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0

    # Resume from MoE checkpoint if exists
    if os.path.exists(latest_ckpt_path):
        print(f"\n📂 Resuming MoE from checkpoint: {latest_ckpt_path}")
        ckpt = load_moe_checkpoint(latest_ckpt_path, model, optimizer, scheduler, scaler)
        scheduler.total_steps = train_config.max_train_steps
        scheduler.warmup_steps = train_config.warmup_steps
        global_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"   Resumed at step: {global_step} | Best val loss: {best_val_loss:.4f}")

    # ---- Training state vars ----
    accumulated_loss = 0.0
    accumulated_ce_loss = 0.0
    accumulated_router_loss = 0.0
    accumulated_mem_loss = 0.0
    accumulated_gate_loss = 0.0
    accumulated_mem_entropy = 0.0
    accumulated_mem_top1 = 0.0
    accumulated_mem_top10 = 0.0
    accumulated_mem_unique = 0.0
    micro_step = 0
    tokens_processed = 0
    start_time = time.time()
    stop_training = False

    print(f"\n{'=' * 70}")
    print(f"  🚀 Starting OPTIMUS_moe Training — {format_num(total_steps)} optimizer steps")
    print(f"  🧠 Phase 1: Frozen Base + Memory MoE")
    print(f"  📚 Streaming from FineWeb-Edu ({train_config.fineweb_subset})")
    print(f"{'=' * 70}\n")

    model.train()
    model.base_model.eval()  # Keep base frozen

    train_iter = iter(train_loader)
    current_epoch = 1

    try:
        while global_step < total_steps and not stop_training:
            # Get next batch
            try:
                batch = next(train_iter)
            except StopIteration:
                if current_epoch < train_config.epochs_per_chunk:
                    print(f"\n  🔄 Epoch {current_epoch}/{train_config.epochs_per_chunk} finished. Restarting current chunk data...")
                    train_iter = iter(train_loader)
                    current_epoch += 1
                    continue
                else:
                    print(f"\n  📦 Reached end of current data chunk (step {global_step} after {current_epoch} epochs).")
                    save_moe_checkpoints(
                        model, optimizer, scheduler, scaler,
                        global_step, best_val_loss, best_val_loss,
                        moe_config, train_config, train_config.checkpoint_dir,
                    )
                    break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            batch_tokens = (labels[..., 1:] != -100).sum().item()
            tokens_processed += batch_tokens

            # ---- Forward pass with AMP ----
            os.environ["OPTIMUS_GLOBAL_STEP"] = str(global_step)
            with autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda" and train_config.use_amp)):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"]
                loss = loss / train_config.gradient_accumulation_steps

            # ---- Backward pass ----
            scaler.scale(loss).backward()
            accumulated_loss += loss.item()
            if "ce_loss" in outputs and outputs["ce_loss"] is not None:
                accumulated_ce_loss += outputs["ce_loss"].item() / train_config.gradient_accumulation_steps
            if "router_loss" in outputs and outputs["router_loss"] is not None:
                accumulated_router_loss += outputs["router_loss"].item() / train_config.gradient_accumulation_steps
            if "memory_loss" in outputs and outputs["memory_loss"] is not None:
                accumulated_mem_loss += outputs["memory_loss"].item() / train_config.gradient_accumulation_steps
            if "gate_loss" in outputs and outputs["gate_loss"] is not None:
                accumulated_gate_loss += outputs["gate_loss"].item() / train_config.gradient_accumulation_steps
                
            if "mem_stats" in outputs and outputs["mem_stats"] is not None:
                stats = outputs["mem_stats"]
                accumulated_mem_entropy += stats["entropy"].item() / train_config.gradient_accumulation_steps
                accumulated_mem_top1 += stats["top1_prob"].item() / train_config.gradient_accumulation_steps
                accumulated_mem_top10 += stats["top10_mass"].item() / train_config.gradient_accumulation_steps
                accumulated_mem_unique += stats["unique_slots"].item() / train_config.gradient_accumulation_steps
            micro_step += 1

            # ---- Optimizer step ----
            if micro_step % train_config.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.get_trainable_params(), train_config.max_grad_norm
                )

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                global_step += 1

                # ---- Logging ----
                if global_step % train_config.log_every_n_steps == 0:
                    elapsed = time.time() - start_time
                    current_lr = scheduler.get_last_lr()[0]
                    tps = tokens_processed / elapsed

                    if device.type == "cuda":
                        mem_used = torch.cuda.max_memory_allocated() / 1e9
                        mem_str = f" | GPU: {mem_used:.1f}GB"
                    else:
                        mem_str = ""

                    progress = global_step / total_steps * 100
                    print(
                        f"Step {global_step}/{total_steps} ({progress:.1f}%) | "
                        f"Loss: {accumulated_loss:.4f} | "
                        f"CE: {accumulated_ce_loss:.4f} | "
                        f"Router: {accumulated_router_loss:.4f} | "
                        f"MemEnt: {accumulated_mem_entropy:.4f} | "
                        f"Gate: {accumulated_gate_loss:.4f} | "
                        f"Top1: {accumulated_mem_top1:.4f} | "
                        f"Top10: {accumulated_mem_top10:.4f} | "
                        f"Uniq: {int(round(accumulated_mem_unique))} | "
                        f"LR: {current_lr:.2e} | "
                        f"Tok/s: {format_num(int(tps))}{mem_str}"
                    )

                accumulated_loss = 0.0
                accumulated_ce_loss = 0.0
                accumulated_router_loss = 0.0
                accumulated_mem_loss = 0.0
                accumulated_gate_loss = 0.0
                accumulated_mem_entropy = 0.0
                accumulated_mem_top1 = 0.0
                accumulated_mem_top10 = 0.0
                accumulated_mem_unique = 0.0

                # ---- Validation ----
                if global_step % train_config.val_every_n_steps == 0:
                    print(f"\n  📝 Validation at step {global_step}...")
                    val_loss = validate(model, val_loader, device, amp_dtype)
                    val_ppl = math.exp(min(val_loss, 20))
                    print(f"     Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")

                    # ---- Factual QA Evaluation ----
                    if global_step % 1000 == 0:
                        print(f"     ❓ Evaluating factual QA accuracy (Step {global_step})...")
                        moe_factual_acc, _ = evaluate_factual_accuracy(model, tokenizer, device)
                        print(f"        Base Acc: {base_factual_acc:.2%} | MoE Acc: {moe_factual_acc:.2%}")

                    # Save latest
                    save_moe_checkpoints(
                        model, optimizer, scheduler, scaler,
                        global_step, best_val_loss, val_loss,
                        moe_config, train_config, train_config.checkpoint_dir,
                    )
                    print(f"     💾 Saved MoE checkpoint → {train_config.checkpoint_dir}")

                    # Check for best
                    if val_loss < best_val_loss:
                        improvement = best_val_loss - val_loss
                        best_val_loss = val_loss
                        patience_counter = 0
                        save_moe_checkpoints(
                            model, optimizer, scheduler, scaler,
                            global_step, best_val_loss, val_loss,
                            moe_config, train_config, train_config.checkpoint_dir,
                            is_best=True,
                        )
                        print(f"     🏆 New best MoE model! (improved by {improvement:.4f})")
                    else:
                        patience_counter += 1
                        print(f"     ⏳ No improvement ({patience_counter}/{train_config.patience})")

                    if patience_counter >= train_config.patience:
                        print(f"\n  🛑 Early stopping triggered after {patience_counter} checks.")
                        print(f"     Best validation loss: {best_val_loss:.4f}")
                        stop_training = True
                        break

                    print()
                    model.train()
                    model.base_model.eval()  # Keep base frozen

    except KeyboardInterrupt:
        print("\n\n🛑 Training interrupted. Saving progress...")
        save_moe_checkpoints(
            model, optimizer, scheduler, scaler,
            global_step, best_val_loss, best_val_loss,
            moe_config, train_config, train_config.checkpoint_dir,
        )
        print(f"   💾 Saved MoE checkpoint → {train_config.checkpoint_dir}")
        print("   You can safely resume later. Goodbye!\n")
        sys.exit(0)

    # ---- Training Complete ----
    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"  ✅ OPTIMUS_moe Training Complete!")
    print(f"{'=' * 70}")
    print(f"  Total time:        {datetime.timedelta(seconds=int(total_time))}")
    print(f"  Total steps:       {global_step}")
    print(f"  Tokens processed:  {format_num(tokens_processed)}")
    print(f"  Best val loss:     {best_val_loss:.4f}")
    print(f"  Best val PPL:      {math.exp(min(best_val_loss, 20)):.2f}")
    print(f"  Checkpoints:       {train_config.checkpoint_dir}")
    print()


# ===========================================================================
# Entry Point
# ===========================================================================

if __name__ == "__main__":
    train()
