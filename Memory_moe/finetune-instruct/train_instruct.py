"""
OPTIMUS_moe LoRA Instruction Fine-Tuning
==========================================
Fine-tune the OPTIMUS MoE model for instruction-following using LoRA
adapters on the Open-Orca/OpenOrca dataset.

Strategy:
    1. Load full OPTIMUS MoE (frozen base + trained MoE layer)
    2. Freeze everything
    3. Apply LoRA adapters to attention + FFN linear layers
    4. Train only LoRA parameters on instruction data
    5. Save lightweight LoRA adapter checkpoints

Usage:
    conda activate EPOCH
    cd Memory_moe/finetune-instruct
    python train_instruct.py
    python train_instruct.py --max_samples 10000 --max_steps 1000
"""

import os
import sys
import math
import time
import argparse

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config_finetune import LoRAConfig, InstructFinetuneConfig
from lora import (
    apply_lora_to_model, get_lora_params, save_lora, load_lora,
    print_lora_summary,
)
from dataset_instruct import build_instruct_dataloaders
from config_moe import MemoryMoEConfig
from model_moe import build_optimus_moe, load_moe_checkpoint
from dataset import get_tokenizer


class Logger(object):
    """Dual logger: writes to both terminal and log file."""
    def __init__(self, filename="finetune_instruct.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


if os.environ.get("CONTINUAL_TRAINER_ACTIVE") != "1":
    sys.stdout = Logger(os.path.join(os.path.dirname(__file__), "training_pipeline_lora.log"))
    sys.stderr = sys.stdout

# Force tqdm-style line buffering, matching the other training scripts
sys.stderr.isatty = lambda: True
sys.stdout.isatty = lambda: True


def get_device():
    """Detect best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🔥 CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return device
    print("⚠️  No CUDA device found. Training on CPU (this will be very slow).")
    return torch.device("cpu")


def format_num(n):
    """Format large numbers: 1234567 -> '1.23M'."""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def get_cosine_lr(step, warmup_steps, max_steps, peak_lr, min_lr):
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return peak_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def validate(model, val_loader, device, amp_dtype):
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda" and amp_dtype != torch.float32)):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        loss = outputs["loss"]
        if loss is not None:
            # Count non-masked tokens
            valid_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    model.train()
    return total_loss / max(total_tokens, 1)


def train(args=None):
    """Main training loop for LoRA instruction fine-tuning."""

    # --- Setup ---
    lora_config = LoRAConfig()
    train_config = InstructFinetuneConfig()

    # Override from args
    if args is not None:
        if args.max_samples is not None:
            train_config.max_samples = args.max_samples
        if args.max_steps is not None:
            train_config.max_train_steps = args.max_steps
        if args.lr is not None:
            train_config.learning_rate = args.lr
        if args.rank is not None:
            lora_config.rank = args.rank
            lora_config.alpha = args.rank * 2  # Keep alpha = 2*rank

    # Output directory is used for LoRA checkpoints
    os.makedirs(train_config.output_dir, exist_ok=True)

    print("=" * 70)
    print("  LoRA Instruction Fine-Tuning for OPTIMUS_moe")
    print("  Dataset: Open-Orca/OpenOrca")
    print("=" * 70)

    device = get_device()
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    # Match the main training scripts: bf16 if supported, otherwise fp16 + GradScaler.
    if train_config.use_amp and device.type == "cuda":
        if train_config.amp_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            use_scaler = False
            print("AMP: bfloat16 (no GradScaler needed)")
        else:
            amp_dtype = torch.float16
            use_scaler = True
            print("AMP: float16 (with GradScaler)")
    else:
        amp_dtype = torch.float32
        use_scaler = False
        print("AMP: disabled (using float32)")

    # --- Load tokenizer ---
    print("\nLoading tokenizer...")
    tokenizer = get_tokenizer()

    # --- Load OPTIMUS MoE model ---
    print("\nLoading OPTIMUS MoE model...")
    moe_config = MemoryMoEConfig()
    model = build_optimus_moe(moe_config, device)

    # Load MoE checkpoint
    moe_ckpt_path = train_config.moe_checkpoint
    if os.path.exists(moe_ckpt_path):
        print(f"\nLoading MoE checkpoint: {moe_ckpt_path}")
        ckpt = load_moe_checkpoint(moe_ckpt_path, model)
        if "global_step" in ckpt:
            print(f"   MoE trained steps: {ckpt['global_step']}")
        if "best_val_loss" in ckpt:
            print(f"   MoE best val loss: {ckpt['best_val_loss']:.4f}")
    else:
        print(f"\nWarning: No MoE checkpoint at {moe_ckpt_path}")
        print("   Using untrained MoE weights")

    # --- Apply LoRA ---
    print(f"\nApplying LoRA (rank={lora_config.rank}, alpha={lora_config.alpha})...")
    num_lora = apply_lora_to_model(model, lora_config)
    print_lora_summary(model)

    # --- Load dataset ---
    train_loader, val_loader = build_instruct_dataloaders(tokenizer, train_config)

    # --- Optimizer (only LoRA params) ---
    lora_params = get_lora_params(model)
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=train_config.learning_rate,
        betas=(train_config.adam_beta1, train_config.adam_beta2),
        eps=train_config.adam_eps,
        weight_decay=train_config.weight_decay,
    )

    print(f"\nOptimizer: AdamW ({len(lora_params)} parameter groups)")
    print(f"   Peak LR:           {train_config.learning_rate}")
    print(f"   Weight decay:      {train_config.weight_decay}")
    print(f"   Grad accumulation: {train_config.gradient_accumulation_steps}")
    print(f"   Effective batch:   {train_config.batch_size * train_config.gradient_accumulation_steps}")
    print(f"   Max steps:         {train_config.max_train_steps}")
    print(f"   Warmup steps:      {train_config.warmup_steps}")
    print(f"   Loss masking:      {'prompt masked' if train_config.mask_prompt else 'full sequence'}")

    # --- Resume from LoRA checkpoint ---
    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0

    resume_path = os.path.join(train_config.output_dir, "lora_latest.pt")
    if os.path.exists(resume_path):
        print(f"\nResuming from: {resume_path}")
        ckpt = load_lora(model, resume_path, optimizer=optimizer)
        global_step = ckpt.get("global_step", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"   Resumed at step: {global_step} | Best val loss: {best_val_loss:.4f}")

    # --- Training Loop ---
    print("\n" + "=" * 70)
    print("  Starting LoRA Instruction Fine-Tuning")
    print("=" * 70 + "\n")

    model.train()
    optimizer.zero_grad()

    # Gradient scaler for mixed precision training
    scaler = GradScaler("cuda", enabled=use_scaler) if device.type == "cuda" else GradScaler(enabled=False)

    accum_loss = 0.0
    accum_ce_loss = 0.0
    accum_tokens = 0
    accum_micro_steps = 0
    micro_step = 0
    epoch = 0
    start_time = time.time()
    while global_step < train_config.max_train_steps:
        epoch += 1
        data_iter = iter(train_loader)

        for batch in data_iter:
            if global_step >= train_config.max_train_steps:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass with AMP
            with autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda" and train_config.use_amp)):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            loss = outputs["loss"]
            if loss is None:
                continue

            # Skip batch if loss is NaN/Inf
            if not torch.isfinite(loss):
                print(f"Warning: non-finite loss detected (step approx {global_step}). Skipping batch.")
                optimizer.zero_grad()
                continue

            # Scale loss for gradient accumulation
            scaled_loss = loss / train_config.gradient_accumulation_steps
            if use_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            accum_loss += loss.item()
            ce_loss = outputs.get("ce_loss")
            if ce_loss is not None:
                accum_ce_loss += ce_loss.item()
            accum_tokens += (labels != -100).sum().item()
            accum_micro_steps += 1
            micro_step += 1

            # Optimizer step after accumulation
            if micro_step % train_config.gradient_accumulation_steps == 0:
                # Unscale gradients before clipping when using GradScaler
                if use_scaler:
                    scaler.unscale_(optimizer)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(lora_params, train_config.max_grad_norm)

                # Update learning rate
                lr = get_cosine_lr(
                    global_step, train_config.warmup_steps,
                    train_config.max_train_steps,
                    train_config.learning_rate, train_config.min_lr,
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                # Step optimizer with scaler when AMP enabled
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # --- Logging ---
                if global_step % train_config.log_every_n_steps == 0:
                    avg_loss = accum_loss / max(accum_micro_steps, 1)
                    avg_ce = accum_ce_loss / max(accum_micro_steps, 1)
                    elapsed = time.time() - start_time
                    tok_per_sec = accum_tokens / max(elapsed, 0.001)

                    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0

                    print(
                        f"Step {global_step}/{train_config.max_train_steps} "
                        f"| Loss: {avg_loss:.4f} "
                        f"| CE: {avg_ce:.4f} "
                        f"| LR: {lr:.2e} "
                        f"| Tok/s: {tok_per_sec/1000:.1f}K "
                        f"| GPU: {gpu_mem:.1f}GB"
                    )

                    accum_loss = 0.0
                    accum_ce_loss = 0.0
                    accum_tokens = 0
                    accum_micro_steps = 0
                    start_time = time.time()

                # --- Validation ---
                if global_step % train_config.val_every_n_steps == 0:
                    print(f"\n  Validation at step {global_step}...")
                    val_loss = validate(model, val_loader, device, amp_dtype)
                    val_ppl = math.exp(min(val_loss, 20))
                    print(f"     Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")

                    # Save latest
                    save_lora(
                        model,
                        os.path.join(train_config.output_dir, "lora_latest.pt"),
                        optimizer=optimizer,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                        config=lora_config,
                    )

                    # Check for best
                    if val_loss < best_val_loss:
                        improvement = best_val_loss - val_loss
                        best_val_loss = val_loss
                        patience_counter = 0
                        save_lora(
                            model,
                            os.path.join(train_config.output_dir, "lora_best.pt"),
                            optimizer=optimizer,
                            global_step=global_step,
                            best_val_loss=best_val_loss,
                            config=lora_config,
                        )
                        print(f"     New best! (improved by {improvement:.4f})")
                    else:
                        patience_counter += 1
                        print(f"     No improvement ({patience_counter}/{train_config.patience})")

                    if patience_counter >= train_config.patience:
                        print(f"\n  Early stopping at step {global_step}")
                        break

                    model.train()
                    print()

                # --- Periodic save ---
                if global_step % train_config.save_every_n_steps == 0:
                    save_lora(
                        model,
                        os.path.join(train_config.output_dir, f"lora_step_{global_step}.pt"),
                        optimizer=optimizer,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                        config=lora_config,
                    )

        if patience_counter >= train_config.patience:
            break

    # --- Final save ---
    print(f"\nTraining complete at step {global_step}")
    save_lora(
        model,
        os.path.join(train_config.output_dir, "lora_final.pt"),
        optimizer=optimizer,
        global_step=global_step,
        best_val_loss=best_val_loss,
        config=lora_config,
    )
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Best validation PPL:  {math.exp(min(best_val_loss, 20)):.2f}")


def main():
    parser = argparse.ArgumentParser(description="LoRA Instruction Fine-Tuning for OPTIMUS_moe")
    parser.add_argument("--max_samples", type=int, default=None, help="Number of OpenOrca samples")
    parser.add_argument("--max_steps", type=int, default=None, help="Max training steps")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--rank", type=int, default=None, help="LoRA rank")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Checkpoints saved.")
