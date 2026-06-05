"""
Pure PyTorch LoRA (Low-Rank Adaptation) Implementation
========================================================
Implements LoRA adapters for fine-tuning without external dependencies (no PEFT).

LoRA decomposes weight updates into low-rank matrices:
    W' = W + (alpha/r) * B @ A
where:
    W: Original frozen weight (d_out, d_in)
    A: Low-rank down-projection (r, d_in), initialized with Kaiming uniform
    B: Low-rank up-projection (d_out, r), initialized with zeros
    r: Rank of the adaptation
    alpha: Scaling factor

This means at initialization, B @ A = 0, so the model starts identical
to the original.

Usage:
    from lora import apply_lora_to_model, get_lora_params, save_lora, load_lora

    model = build_optimus_moe(...)
    apply_lora_to_model(model, lora_config)
    optimizer = AdamW(get_lora_params(model), lr=2e-4)
    # ... train ...
    save_lora(model, "lora_adapter.pt")
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    LoRA-wrapped linear layer.

    Replaces a standard nn.Linear with:
        output = original_linear(x) + (alpha/r) * dropout(x @ A^T) @ B^T

    The original weight is frozen; only A and B are trainable.
    """

    def __init__(self, original_linear, rank=16, alpha=32, dropout=0.05):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Freeze original weight
        original_linear.weight.requires_grad = False
        if original_linear.bias is not None:
            original_linear.bias.requires_grad = False

        # LoRA matrices
        device = original_linear.weight.device
        dtype = original_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=dtype))

        # Initialize A with Kaiming uniform (same as nn.Linear default)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Dropout on input before LoRA path
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Track whether LoRA has been merged into base weights
        self.merged = False

    def forward(self, x):
        # Original path (frozen)
        result = self.original_linear(x)

        if not self.merged:
            # LoRA path: x @ A^T @ B^T * scaling
            lora_out = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
            result = result + lora_out * self.scaling

        return result

    def merge(self):
        """Merge LoRA weights into the original linear layer for efficient inference."""
        if not self.merged:
            with torch.no_grad():
                # W' = W + scaling * B @ A
                delta = (self.lora_B @ self.lora_A) * self.scaling
                self.original_linear.weight.add_(delta.to(self.original_linear.weight.dtype))
            self.merged = True

    def unmerge(self):
        """Remove merged LoRA weights from the original linear layer."""
        if self.merged:
            with torch.no_grad():
                delta = (self.lora_B @ self.lora_A) * self.scaling
                self.original_linear.weight.sub_(delta.to(self.original_linear.weight.dtype))
            self.merged = False


def apply_lora_to_model(model, lora_config):
    """
    Walk the model and replace target nn.Linear modules with LoRALinear wrappers.

    Args:
        model: The full OptimusMoEModel
        lora_config: LoRAConfig with rank, alpha, dropout, target_modules

    Returns:
        int: Number of LoRA-adapted modules
    """
    target_names = set(lora_config.target_modules)
    lora_count = 0
    total_lora_params = 0

    # First, freeze ALL model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Walk all named modules and replace targets
    modules_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if the final segment of the module name matches a target
            leaf_name = name.split(".")[-1]
            if leaf_name in target_names:
                modules_to_replace.append((name, module))

    for name, original_module in modules_to_replace:
        # Navigate to the parent module
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        leaf = parts[-1]

        # Create LoRA wrapper
        lora_module = LoRALinear(
            original_module,
            rank=lora_config.rank,
            alpha=lora_config.alpha,
            dropout=lora_config.dropout,
        )

        # Replace the module
        setattr(parent, leaf, lora_module)
        lora_count += 1

        # Count LoRA params
        lora_params = lora_config.rank * original_module.in_features + \
                      original_module.out_features * lora_config.rank
        total_lora_params += lora_params

    print(f"   LoRA applied to {lora_count} modules")
    print(f"   LoRA parameters: {total_lora_params:,}")

    return lora_count


def get_lora_params(model):
    """
    Return only the LoRA adapter parameters (lora_A, lora_B) for the optimizer.

    Returns:
        list of nn.Parameter
    """
    lora_params = []
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
            lora_params.append(param)
    return lora_params


def get_lora_state_dict(model):
    """
    Extract only LoRA weights from the model state dict.

    Returns:
        dict: {name: tensor} for all LoRA parameters
    """
    lora_state = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_state[name] = param.data.clone()
    return lora_state


def save_lora(model, filepath, optimizer=None, scheduler=None,
              global_step=0, best_val_loss=float("inf"), config=None):
    """
    Save only the LoRA adapter weights and training state.

    The saved file is much smaller than a full model checkpoint since
    it only contains the low-rank matrices.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    state = {
        "lora_state_dict": get_lora_state_dict(model),
        "global_step": global_step,
        "best_val_loss": best_val_loss,
    }

    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if config is not None:
        state["lora_config"] = config.__dict__ if hasattr(config, "__dict__") else config

    # Atomic save
    tmp_path = filepath + ".tmp"
    torch.save(state, tmp_path)
    if os.path.exists(filepath):
        os.remove(filepath)
    os.replace(tmp_path, filepath)

    # Report size
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"   Saved LoRA adapter ({size_mb:.1f} MB) -> {filepath}")

    return filepath


def load_lora(model, filepath, optimizer=None, scheduler=None):
    """
    Load LoRA adapter weights into a model that already has LoRA modules applied.

    Args:
        model: Model with LoRALinear modules (from apply_lora_to_model)
        filepath: Path to saved LoRA checkpoint
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state

    Returns:
        dict: The full checkpoint dict (for reading global_step, etc.)
    """
    checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
    lora_state = checkpoint["lora_state_dict"]

    # Load LoRA parameters into the model
    model_state = model.state_dict()
    loaded = 0
    for name, param in lora_state.items():
        if name in model_state:
            model_state[name].copy_(param)
            loaded += 1
        else:
            print(f"   Warning: LoRA param '{name}' not found in model")

    print(f"   Loaded {loaded} LoRA parameters from {filepath}")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint


def merge_all_lora(model):
    """Merge all LoRA adapters into base weights for efficient inference."""
    count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
            count += 1
    print(f"   Merged {count} LoRA adapters into base weights")


def unmerge_all_lora(model):
    """Unmerge all LoRA adapters from base weights (for continued training)."""
    count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()
            count += 1
    print(f"   Unmerged {count} LoRA adapters from base weights")


def print_lora_summary(model):
    """Print a summary of LoRA-adapted modules and parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\n   Model Parameter Summary:")
    print(f"   Total:      {total_params:,}")
    print(f"   Trainable:  {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"   Frozen:     {frozen_params:,}")

    # List LoRA modules
    lora_modules = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            a_params = module.lora_A.numel()
            b_params = module.lora_B.numel()
            lora_modules.append((name, a_params + b_params))

    if lora_modules:
        print(f"\n   LoRA Modules ({len(lora_modules)}):")
        for name, params in lora_modules[:10]:  # Show first 10
            print(f"     {name}: {params:,} params")
        if len(lora_modules) > 10:
            print(f"     ... and {len(lora_modules) - 10} more")
