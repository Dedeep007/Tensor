"""Quick verification of OPTIMUS model architecture with MoE."""
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from config import OptimusConfig, TrainingConfig
from model import OptimusForCausalLM

print("=" * 60)
print("  OPTIMUS Verification (with MoE)")
print("=" * 60)

# Config check
c = OptimusConfig()
t = TrainingConfig()
print(f"\n[Config]")
print(f"  FineWeb:   {t.fineweb_name}")
print(f"  Vocab:     {c.vocab_size}")
print(f"  Hidden:    {c.hidden_size}")
print(f"  Layers:    {c.num_hidden_layers}")
print(f"  Q heads:   {c.num_attention_heads}")
print(f"  KV heads:  {c.num_key_value_heads}")

# MoE config
print(f"\n[MoE Config]")
print(f"  Enabled:        {c.use_moe}")
print(f"  Experts:        {c.num_experts} routed, {c.num_shared_experts} shared")
print(f"  Top-K:          {c.num_experts_per_tok}")
print(f"  Expert FFN dim: {c.expert_intermediate_size}")
print(f"  Dense FFN dim:  {c.intermediate_size}")
print(f"  Layer freq:     every {c.moe_layer_freq} layer(s)")

# Model check
model = OptimusForCausalLM(c)
total = sum(p.numel() for p in model.parameters())
active = model.get_num_params(count_active=True)
moe_layers = sum(1 for layer in model.layers if layer.is_moe)
dense_layers = c.num_hidden_layers - moe_layers

print(f"\n[Model Parameters]")
print(f"  Total params:      {total:,}")
print(f"  Active params/tok: {active:,}")
print(f"  MoE layers:        {moe_layers}")
print(f"  Dense layers:      {dense_layers}")
print(f"  Sparsity ratio:    {1 - active/total:.1%}")

# Forward pass
print(f"\n[Forward Pass Test]")
x = torch.randint(0, 1000, (2, 64))
labels = x.clone()
out = model(x, labels=labels)
print(f"  Logits shape:  {out['logits'].shape}")
print(f"  Total loss:    {out['loss'].item():.4f}")
print(f"  Aux loss:      {out['aux_loss'].item():.6f}")
print(f"  KV cache:      {len(out['past_key_values'])} layers")

# Generation step (single token with KV cache)
print(f"\n[Generation Test]")
out2 = model(x[:, :1], past_key_values=out['past_key_values'])
print(f"  Gen logits:    {out2['logits'].shape}")

print(f"\nAll checks passed!")
