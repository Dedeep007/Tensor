"""
OPTIMUS_moe LoRA Instruction Generation
==========================================
Load the OPTIMUS MoE model with LoRA instruction-tuning adapters
and generate responses in chat format.

Usage:
    conda activate EPOCH
    cd Memory_moe/finetune-instruct
    python generate_instruct.py --prompt "What is the capital of France?"
    python generate_instruct.py --interactive
"""

import os
import sys
import argparse
import time

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn.functional as F

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config_finetune import LoRAConfig, InstructFinetuneConfig
from lora import apply_lora_to_model, load_lora, merge_all_lora
from config_moe import MemoryMoEConfig
from model_moe import build_optimus_moe, load_moe_checkpoint
from dataset import get_tokenizer


def format_instruction_prompt(user_message, system_prompt=None):
    """
    Format a user message into ChatML format for generation.

    Returns the prompt up to (and including) '<|im_start|>assistant\n'
    so the model generates the response.
    """
    prompt = ""
    if system_prompt:
        prompt += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


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


@torch.no_grad()
def generate_response(
    model,
    tokenizer,
    user_message,
    system_prompt=None,
    max_new_tokens=256,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2,
    device="cuda",
):
    """
    Generate an instruction-following response.

    Args:
        model: OptimusMoEModel with LoRA adapters
        tokenizer: Qwen2.5 tokenizer
        user_message: The user's question/instruction
        system_prompt: Optional system prompt
        max_new_tokens: Maximum response length
        temperature: Sampling temperature
        top_k: Top-K filtering
        top_p: Nucleus sampling threshold
        repetition_penalty: Repetition penalty
        device: Device

    Returns:
        response_text: Just the assistant's response (no chat template markup)
    """
    model.eval()

    # Format prompt
    prompt = format_instruction_prompt(user_message, system_prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    prompt_len = input_ids.shape[1]

    generated_ids = input_ids[0].tolist()
    past_key_values = None

    # Get special token IDs for stopping
    im_end_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    eos_id = tokenizer.eos_token_id

    # Prefill
    outputs = model(input_ids=input_ids, past_key_values=None)
    logits = outputs["logits"]
    past_key_values = outputs["past_key_values"]

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

        # Stop on EOS or <|im_end|>
        if next_token_id == eos_id:
            break
        if im_end_id and next_token_id == im_end_id[0]:
            break

        next_input = next_token.unsqueeze(0)
        outputs = model(input_ids=next_input, past_key_values=past_key_values)
        next_token_logits = outputs["logits"][0, -1, :].clone()
        past_key_values = outputs["past_key_values"]

    # Decode only the generated response (after the prompt)
    response_ids = generated_ids[prompt_len:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

    return response_text.strip()


def load_instruct_model(lora_path=None, merge=True, device="cuda"):
    """
    Load OPTIMUS MoE with LoRA instruction adapters.

    Args:
        lora_path: Path to LoRA checkpoint (default: checkpoints/lora_best.pt)
        merge: If True, merge LoRA into base for faster inference
        device: Device

    Returns:
        model: Ready for generation
    """
    # Load base + MoE
    moe_config = MemoryMoEConfig()
    model = build_optimus_moe(moe_config, device)

    # Load MoE weights
    train_config = InstructFinetuneConfig()
    moe_ckpt = train_config.moe_checkpoint
    if os.path.exists(moe_ckpt):
        print(f"\nLoading MoE checkpoint: {moe_ckpt}")
        load_moe_checkpoint(moe_ckpt, model)

    # Apply LoRA structure
    lora_config = LoRAConfig()
    apply_lora_to_model(model, lora_config)

    # Load LoRA weights
    if lora_path is None:
        lora_path = os.path.join(os.path.dirname(__file__), "checkpoints", "lora_best.pt")

    if os.path.exists(lora_path):
        print(f"Loading LoRA adapters: {lora_path}")
        ckpt = load_lora(model, lora_path)
        if "global_step" in ckpt:
            print(f"   Trained steps: {ckpt['global_step']}")
        if "best_val_loss" in ckpt:
            import math
            bvl = ckpt["best_val_loss"]
            print(f"   Best val loss: {bvl:.4f} (PPL: {math.exp(min(bvl, 20)):.2f})")
    else:
        print(f"Warning: No LoRA checkpoint at {lora_path}")
        print("   Using untuned LoRA adapters (random initialization)")

    # Merge for faster inference
    if merge:
        merge_all_lora(model)

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="OPTIMUS_moe Instruction Generation (with LoRA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--prompt", type=str, default=None, help="User instruction/question")
    parser.add_argument("--system", type=str, default=None, help="System prompt")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA checkpoint")
    parser.add_argument("--no_merge", action="store_true", help="Don't merge LoRA (slower)")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max response tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K filtering")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty")

    args = parser.parse_args()

    if not args.interactive and args.prompt is None:
        parser.error("Please provide --prompt or use --interactive mode")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = get_tokenizer()
    model = load_instruct_model(
        lora_path=args.lora_path,
        merge=not args.no_merge,
        device=device,
    )

    if args.interactive:
        print("\n" + "=" * 60)
        print("  OPTIMUS_moe Instruction Chat (LoRA)")
        print("=" * 60)
        print(f"  Temperature:        {args.temperature}")
        print(f"  Top-K:              {args.top_k}")
        print(f"  Top-P:              {args.top_p}")
        print(f"  Max tokens:         {args.max_tokens}")
        print(f"  System prompt:      {args.system or '(none)'}")
        print(f"  Type 'quit' or 'exit' to stop.")
        print("=" * 60)

        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("\nGoodbye!")
                break

            start = time.time()
            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                user_message=user_input,
                system_prompt=args.system,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                device=device,
            )
            elapsed = time.time() - start
            num_tokens = len(tokenizer.encode(response))

            print(f"\nAssistant: {response}")
            print(f"  ({num_tokens} tokens, {elapsed:.2f}s, {num_tokens/max(elapsed,0.001):.1f} tok/s)")

    else:
        print(f"\nUser: {args.prompt}")
        print("\nGenerating...\n")

        start = time.time()
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            user_message=args.prompt,
            system_prompt=args.system,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=device,
        )
        elapsed = time.time() - start
        num_tokens = len(tokenizer.encode(response))

        print(f"Assistant: {response}")
        print(f"\n  Stats: {num_tokens} tokens | {elapsed:.2f}s | {num_tokens/max(elapsed,0.001):.1f} tok/s")


if __name__ == "__main__":
    main()
