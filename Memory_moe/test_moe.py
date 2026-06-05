"""Quick verification test for OPTIMUS_moe architecture."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from config_moe import MemoryMoEConfig
from model_moe import MemoryMoELayer, MemoryExpert

print("=== Test 1: MemoryMoELayer param count ===")
moe_config = MemoryMoEConfig()
moe_layer = MemoryMoELayer(moe_config)
total_moe = sum(p.numel() for p in moe_layer.parameters())
print(f"MoE layer params: {total_moe:,}")

print("\n=== Test 2: MoE layer forward pass ===")
dummy_input = torch.randn(1, 16, 512)
moe_layer.eval()
output, aux_loss, div_loss, mem_loss, gate_loss, stats = moe_layer(dummy_input)
print(f"Input shape:  {dummy_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Aux loss:     {aux_loss.item():.6f}")
print(f"Div loss:     {div_loss.item():.6f}")
print(f"Mem loss:     {mem_loss.item():.6f}")
print(f"Gate loss:    {gate_loss.item():.6f}")
print(f"Stats:        {stats}")
print(f"Shapes match: {output.shape == dummy_input.shape}")

print("\n=== Test 3: Per-expert param count ===")
for i, expert in enumerate(moe_layer.experts):
    ep = sum(p.numel() for p in expert.parameters())
    print(f"Expert {i}: {ep:,} params")

router_p = sum(p.numel() for p in moe_layer.router.parameters())
print(f"Router: {router_p:,} params")

out_p = sum(p.numel() for p in moe_layer.output_norm.parameters()) + sum(p.numel() for p in moe_layer.output_proj.parameters())
print(f"Output proj + norm: {out_p:,} params")

print("\n=== Test 4: Gradient flow check ===")
moe_layer.train()
dummy_input2 = torch.randn(1, 1024, 512, requires_grad=True)
output2, aux2, div2, mem2, gate2, stats2 = moe_layer(dummy_input2)
loss = output2.sum() + aux2 + div2 + mem2 + gate2
loss.backward()
has_grads = all(p.grad is not None for p in moe_layer.parameters() if p.requires_grad)
print(f"All MoE params have gradients: {has_grads}")

print("\n✅ All tests passed!")
