"""
OPTIMUS Inference / Generation
================================
Load a trained OPTIMUS checkpoint and generate text autoregressively
with KV-cache, top-k sampling, top-p (nucleus) sampling, temperature
control, and repetition penalty.

Usage:
    conda activate EPOCH
    python generate.py --prompt "The future of AI is"
    python generate.py --prompt "Once upon a time" --max_tokens 256 --temperature 0.8
    python generate.py --interactive
"""

import os
import sys
import argparse

# Fix Windows console encoding for emoji logging
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F
from config import OptimusConfig
from model import OptimusForCausalLM
from dataset import get_tokenizer
import os
from configv2 import TrainingConfigV2

# Load unified configuration for checkpoint paths
cfg = TrainingConfigV2()


# ===========================================================================
# Sampling Utilities
# ===========================================================================

def top_k_top_p_filter(logits, top_k=50, top_p=0.9):
    """
    Filter logits using top-k and/or top-p (nucleus) sampling.
    
    Args:
        logits: (vocab_size,) raw logits for the next token
        top_k: Keep only top-k tokens (0 = disabled)
        top_p: Keep tokens with cumulative probability <= top_p (1.0 = disabled)
    
    Returns:
        Filtered logits with impossible tokens set to -inf.
    """
    # Top-K filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float("-inf")
    
    # Top-P (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift right so that the first token above threshold is kept
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = float("-inf")
    
    return logits


def apply_repetition_penalty(logits, generated_ids, penalty=1.2):
    """
    Apply repetition penalty to discourage repeating tokens.
    
    For tokens that have already appeared:
        - If logit > 0: divide by penalty
        - If logit < 0: multiply by penalty
    """
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
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2,
    device="cuda",
    stream=False,
):
    """
    Autoregressive text generation with KV-cache.
    
    The KV-cache avoids recomputing attention over the entire sequence
    at each step — only the new token's query attends to all cached
    keys and values, giving O(1) compute per token (excluding the LM head).
    
    Args:
        model: OptimusForCausalLM instance
        tokenizer: Qwen tokenizer
        prompt: Input text prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (lower = more deterministic)
        top_k: Top-K filtering (0 = disabled)
        top_p: Nucleus sampling threshold (1.0 = disabled)
        repetition_penalty: Penalty for repeated tokens (1.0 = disabled)
        device: Device to run on
        stream: Whether to stream output to stdout
    
    Returns:
        Generated text string (including the prompt).
    """
    model.eval()
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    generated_ids = input_ids[0].tolist()
    past_key_values = None
    
    # First pass: process the entire prompt (prefill)
    outputs = model(input_ids=input_ids, past_key_values=None)
    logits = outputs["logits"]
    past_key_values = outputs["past_key_values"]
    
    if stream:
        print(prompt, end="", flush=True)
        
    # Get logits for the last position (next token prediction)
    next_token_logits = logits[0, -1, :].clone()
    
    for step in range(max_new_tokens):
        # Apply temperature
        if temperature > 0:
            next_token_logits = next_token_logits / temperature
        
        # Apply repetition penalty
        next_token_logits = apply_repetition_penalty(
            next_token_logits, generated_ids, repetition_penalty
        )
        
        # Apply top-k and top-p filtering
        filtered_logits = top_k_top_p_filter(
            next_token_logits.clone(), top_k=top_k, top_p=top_p
        )
        
        # Sample from the distribution
        if temperature > 0:
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy decoding
            next_token = torch.argmax(filtered_logits, dim=-1, keepdim=True)
        
        next_token_id = next_token.item()
        generated_ids.append(next_token_id)
        
        if stream:
            print(tokenizer.decode([next_token_id]), end="", flush=True)
            
        # Check for EOS
        if next_token_id == tokenizer.eos_token_id:
            break
        
        # Generate next token using KV-cache (only process the new token)
        next_input = next_token.unsqueeze(0)  # (1, 1)
        outputs = model(input_ids=next_input, past_key_values=past_key_values)
        
        next_token_logits = outputs["logits"][0, -1, :].clone()
        past_key_values = outputs["past_key_values"]
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text


# ===========================================================================
# Checkpoint Loading
# ===========================================================================

def load_model_from_checkpoint(checkpoint_path, device="cuda"):
    """
    Load the OPTIMUS model from a training checkpoint.
    
    Args:
        checkpoint_path: Path to the .pt checkpoint file
        device: Device to load the model onto
    
    Returns:
        model: OptimusForCausalLM with loaded weights
        checkpoint: Full checkpoint dict (for inspection)
    """
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Reconstruct config from saved hyperparameters
    config = OptimusConfig()
    if "model_config" in checkpoint:
        for k, v in checkpoint["model_config"].items():
            setattr(config, k, v)
    
    # Build and load model
    model = OptimusForCausalLM(config)
    
    # Handle torch.compile state_dict keys (remove "_orig_mod." prefix)
    state_dict = checkpoint["model_state_dict"]
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k.replace("_orig_mod.", "")
        cleaned_state_dict[clean_key] = v
    
    model.load_state_dict(cleaned_state_dict)
    model = model.to(device)
    model.eval()
    
    # Print checkpoint info
    if "global_step" in checkpoint:
        print(f"   Step:     {checkpoint['global_step']}")
    if "epoch" in checkpoint:
        print(f"   Epoch:    {checkpoint['epoch']}")
    if "val_loss" in checkpoint:
        import math
        val_loss = checkpoint["val_loss"]
        print(f"   Val Loss: {val_loss:.4f} (PPL: {math.exp(min(val_loss, 20)):.2f})")
    if "best_val_loss" in checkpoint:
        print(f"   Best Val: {checkpoint['best_val_loss']:.4f}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Params:   {total_params:,}")
    print()
    
    return model, checkpoint


# ===========================================================================
# Interactive Mode
# ===========================================================================

def interactive_mode(model, tokenizer, device, args):
    """Run interactive text generation in a loop."""
    print("=" * 60)
    print("  🤖 OPTIMUS Interactive Generation")
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
        
        import time
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
              f"{num_new_tokens/elapsed:.1f} tok/s")


# ===========================================================================
# CLI Entry Point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="OPTIMUS Text Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate.py --prompt "The meaning of life is"
  python generate.py --prompt "def fibonacci(n):" --temperature 0.5 --max_tokens 300
  python generate.py --interactive
  python generate.py --checkpoint checkpoints/checkpoint_step_500.pt --prompt "Hello"
        """,
    )
    
    # Required
    parser.add_argument("--prompt", type=str, default=None,
                        help="Input text prompt for generation")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode (loop)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare generation between best_model.pt and latest_checkpoint.pt")
    parser.add_argument("--stream", action="store_true",
                        help="Stream tokens to stdout as they are generated")
    
    # Model
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (default: checkpoints/best_model.pt)")
    
    # Generation parameters
    parser.add_argument("--max_tokens", type=int, default=200,
                        help="Maximum new tokens to generate (default: 200)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7, 0 = greedy)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-K filtering (default: 50, 0 = disabled)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling threshold (default: 0.9, 1.0 = disabled)")
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                        help="Repetition penalty (default: 1.2, 1.0 = disabled)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.interactive and args.prompt is None:
        parser.error("Please provide --prompt or use --interactive mode")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"🔥 Device: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  Running on CPU")
    
    tokenizer = get_tokenizer()

    if args.compare:
        best_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
        latest_path = os.path.join(cfg.checkpoint_dir, "latest_checkpoint.pt")
        
        if not os.path.exists(best_path) or not os.path.exists(latest_path):
            print("❌ Both best_model.pt and latest_checkpoint.pt must exist for --compare mode.")
            sys.exit(1)
            
        print("\n" + "="*60)
        print("🔍 LOADING BEST MODEL")
        print("="*60)
        model_best, _ = load_model_from_checkpoint(best_path, device)
        
        print("\n" + "="*60)
        print("🔍 LOADING LATEST CHECKPOINT")
        print("="*60)
        model_latest, _ = load_model_from_checkpoint(latest_path, device)
        
        if args.interactive:
            print("\n" + "=" * 60)
            print("  🤖 OPTIMUS Interactive Comparison Mode")
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
                
                print("\n[ BEST MODEL ] " + "─"*45)
                out_best = generate(model_best, tokenizer, prompt, args.max_tokens, args.temperature, args.top_k, args.top_p, args.repetition_penalty, device, stream=args.stream)
                if not args.stream: print(out_best)
                else: print()
                
                print("\n[ LATEST CHECKPOINT ] " + "─"*38)
                out_latest = generate(model_latest, tokenizer, prompt, args.max_tokens, args.temperature, args.top_k, args.top_p, args.repetition_penalty, device, stream=args.stream)
                if not args.stream: print(out_latest)
                else: print()
                print("─" * 60)
        else:
            print(f"💬 Prompt: {args.prompt}\n")
            print("\n[ BEST MODEL ] " + "─"*45)
            out_best = generate(model_best, tokenizer, args.prompt, args.max_tokens, args.temperature, args.top_k, args.top_p, args.repetition_penalty, device, stream=args.stream)
            if not args.stream: print(out_best)
            else: print()
            
            print("\n[ LATEST CHECKPOINT ] " + "─"*38)
            out_latest = generate(model_latest, tokenizer, args.prompt, args.max_tokens, args.temperature, args.top_k, args.top_p, args.repetition_penalty, device, stream=args.stream)
            if not args.stream: print(out_latest)
            else: print()
            print("─" * 60)
            
    else:
        # Load single checkpoint
        ckpt_path = args.checkpoint
        if ckpt_path is None:
            # Use the checkpoint directory from the unified config (defaults to chkpoints_v2)
            ckpt_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
        
        if not os.path.exists(ckpt_path):
            print(f"❌ Checkpoint not found: {ckpt_path}")
            print("   Please train the model first with: python train.py")
            sys.exit(1)
        
        model, checkpoint = load_model_from_checkpoint(ckpt_path, device)
        
        # Run generation
        if args.interactive:
            interactive_mode(model, tokenizer, device, args)
        else:
            print(f"💬 Prompt: {args.prompt}\n")
            print("🔄 Generating...\n")
            
            import time
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
            if not args.stream: print(output)
            print("=" * 60)
            print(f"\n  📊 Stats:")
            print(f"     Prompt tokens:    {prompt_tokens}")
            print(f"     Generated tokens: {new_tokens}")
            print(f"     Total tokens:     {total_tokens}")
            print(f"     Time:             {elapsed:.2f}s")
            print(f"     Speed:            {new_tokens/max(elapsed, 0.001):.1f} tokens/sec")


if __name__ == "__main__":
    main()
