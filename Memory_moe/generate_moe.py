"""
OPTIMUS_moe Generation / Inference
=====================================
Load the OPTIMUS_moe model (frozen base + Memory MoE) and generate text.

Usage:
    conda activate EPOCH
    cd Memory_moe
    python generate_moe.py --prompt "The future of AI is"
    python generate_moe.py --interactive
"""

import os
import sys
import argparse
import time

# Fix Windows console encoding for emoji logging
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F

# Add parent directory for base model imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config_moe import MemoryMoEConfig
from model_moe import build_optimus_moe, load_moe_checkpoint
from dataset import get_tokenizer


# ===========================================================================
# Sampling Utilities (reused from base generate.py)
# ===========================================================================

def top_k_top_p_filter(logits, top_k=50, top_p=0.9):
    """Filter logits using top-k and/or top-p (nucleus) sampling."""
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float("-inf")
    
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = float("-inf")
    
    return logits


def apply_repetition_penalty(logits, generated_ids, penalty=1.2):
    """Apply repetition penalty to discourage repeating tokens."""
    if penalty == 1.0 or len(generated_ids) == 0:
        return logits
    
    unique_ids = set(generated_ids)
    for token_id in unique_ids:
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty
    
    return logits


# ===========================================================================
# Generation with KV-Cache
# ===========================================================================

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=200,
    temperature=1.0,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2,
    device="cuda",
    stream=False,
):
    """
    Autoregressive text generation with OPTIMUS_moe.

    Note: KV-cache is used for the frozen base model layers.
    The Memory MoE layer processes each new token position.
    """
    model.eval()
    
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    generated_ids = input_ids[0].tolist()
    past_key_values = None
    
    # Prefill: process entire prompt
    outputs = model(input_ids=input_ids, past_key_values=None)
    logits = outputs["logits"]
    past_key_values = outputs["past_key_values"]
    
    if stream:
        print(prompt, end="", flush=True)
    
    next_token_logits = logits[0, -1, :].clone()
    
    for step in range(max_new_tokens):
        if temperature > 0:
            next_token_logits = next_token_logits / temperature
        
        next_token_logits = apply_repetition_penalty(
            next_token_logits, generated_ids, repetition_penalty
        )
        
        filtered_logits = top_k_top_p_filter(
            next_token_logits.clone(), top_k=top_k, top_p=top_p
        )
        
        if temperature > 0:
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(filtered_logits, dim=-1, keepdim=True)
        
        next_token_id = next_token.item()
        generated_ids.append(next_token_id)
        
        if stream:
            print(tokenizer.decode([next_token_id]), end="", flush=True)
        
        if next_token_id == tokenizer.eos_token_id:
            break
        
        next_input = next_token.unsqueeze(0)
        outputs = model(input_ids=next_input, past_key_values=past_key_values)
        next_token_logits = outputs["logits"][0, -1, :].clone()
        past_key_values = outputs["past_key_values"]
    
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text


# ===========================================================================
# Model Loading
# ===========================================================================

def load_moe_model(checkpoint_path=None, device="cuda"):
    """
    Load the OPTIMUS_moe model for generation.

    Args:
        checkpoint_path: Path to MoE checkpoint (default: checkpoints/moe_full_best.pt)
        device: Device to load onto

    Returns:
        model: OptimusMoEModel ready for generation
    """
    moe_config = MemoryMoEConfig()

    # Build the full model (loads frozen base automatically)
    model = build_optimus_moe(moe_config, device)

    # Load MoE weights
    if checkpoint_path is None:
        checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", "moe_full_best.pt")

    if os.path.exists(checkpoint_path):
        print(f"📂 Loading MoE checkpoint: {checkpoint_path}")
        ckpt = load_moe_checkpoint(checkpoint_path, model)

        if "global_step" in ckpt:
            print(f"   Step:     {ckpt['global_step']}")
        if "best_val_loss" in ckpt:
            import math
            print(f"   Best Val: {ckpt['best_val_loss']:.4f} (PPL: {math.exp(min(ckpt['best_val_loss'], 20)):.2f})")
    else:
        print(f"⚠️  No MoE checkpoint found at {checkpoint_path}")
        print("   Using randomly initialized MoE weights (untrained)")

    model.eval()
    return model


# ===========================================================================
# CLI Entry Point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="OPTIMUS_moe Text Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--prompt", type=str, default=None, help="Input text prompt")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--stream", action="store_true", help="Stream tokens to stdout")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to MoE checkpoint")
    parser.add_argument("--max_tokens", type=int, default=200, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K filtering")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling threshold")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty")
    
    args = parser.parse_args()
    
    if not args.interactive and args.prompt is None:
        parser.error("Please provide --prompt or use --interactive mode")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"🔥 Device: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  Running on CPU")
    
    tokenizer = get_tokenizer()
    model = load_moe_model(args.checkpoint, device)
    
    if args.interactive:
        print("=" * 60)
        print("  🤖 OPTIMUS_moe Interactive Generation")
        print("=" * 60)
        print(f"  Temperature:        {args.temperature}")
        print(f"  Top-K:              {args.top_k}")
        print(f"  Top-P:              {args.top_p}")
        print(f"  Repetition Penalty: {args.repetition_penalty}")
        print(f"  Max New Tokens:     {args.max_tokens}")
        print(f"  Type 'quit' or 'exit' to stop.")
        print("=" * 60)
        
        while True:
            try:
                prompt = input("\n💬 Prompt: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n👋 Goodbye!")
                break
            
            if not prompt:
                continue
            if prompt.lower() in ("quit", "exit", "q"):
                print("\n👋 Goodbye!")
                break
            
            print("\n🔄 Generating...\n")
            start = time.time()
            
            output = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                device=device,
            )
            
            elapsed = time.time() - start
            num_new_tokens = len(tokenizer.encode(output)) - len(tokenizer.encode(prompt))
            
            print("─" * 60)
            print(output)
            print("─" * 60)
            print(f"  ⏱️  {elapsed:.2f}s | {num_new_tokens} tokens | "
                  f"{num_new_tokens/max(elapsed, 0.001):.1f} tok/s")
    else:
        print(f"💬 Prompt: {args.prompt}\n")
        print("🔄 Generating...\n")
        
        start = time.time()
        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=device,
            stream=args.stream,
        )
        
        elapsed = time.time() - start
        prompt_tokens = len(tokenizer.encode(args.prompt))
        total_tokens = len(tokenizer.encode(output))
        new_tokens = total_tokens - prompt_tokens
        
        print("\n" + "=" * 60)
        if not args.stream:
            print(output)
        print("=" * 60)
        print(f"\n  📊 Stats:")
        print(f"     Prompt tokens:    {prompt_tokens}")
        print(f"     Generated tokens: {new_tokens}")
        print(f"     Time:             {elapsed:.2f}s")
        print(f"     Speed:            {new_tokens/max(elapsed, 0.001):.1f} tokens/sec")


if __name__ == "__main__":
    main()
