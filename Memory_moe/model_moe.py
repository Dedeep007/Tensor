"""
OPTIMUS_moe — Memory MoE Model Architecture
=============================================
8-way Memory Mixture of Experts built on top of the frozen OPTIMUS base model.

Architecture:
    Frozen OPTIMUS Base (204M)
        → Memory Router (Top-2)
        → 8× Memory Experts (each with learnable memory bank + cross-attention + SwiGLU FFN)
        → Weighted Merge + Residual
        → Frozen LM Head

Each Memory Expert contains:
    - Learnable Memory Bank: 256 key-value slots (retrievable knowledge store)
    - Memory Cross-Attention: 4-head attention to query the memory bank
    - SwiGLU FFN: 512 → 1536 → 512 (domain-specific transformation)

Total trainable parameters: ~51M
Total model parameters: ~255M (including frozen base)
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory to path so we can import base model
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import OptimusConfig
from model import OptimusForCausalLM, RMSNorm


# ===========================================================================
# Memory Bank — Learnable Key-Value Store
# ===========================================================================
class MemoryBank(nn.Module):
    """
    Learnable key-value memory store for a single expert.

    Added temperature parameter for sharper retrieval.
    """

    """
    Learnable key-value memory store for a single expert.

    Each memory bank consists of:
        - Keys: (num_slots, hidden_size) learnable parameters
        - Values: (num_slots, hidden_size) learnable parameters

    During forward pass, input hidden states query the memory bank via
    multi-head cross-attention, retrieving relevant stored knowledge.

    This allows each expert to develop a persistent "knowledge base" that
    captures domain-specific patterns beyond what the frozen base model knows.

    Args:
        hidden_size: Dimension of hidden states (must match base model)
        num_slots: Number of memory slots (default 256)
        num_heads: Number of attention heads for memory retrieval
    """

    def __init__(self, hidden_size, num_slots=8192, num_heads=4, key_dim=128, value_dim=512, memory_softmax_temp=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_slots = num_slots
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.memory_softmax_temp = memory_softmax_temp
        
        self.key_head_dim = key_dim // num_heads
        self.value_head_dim = value_dim // num_heads
        super().__init__()
        self.hidden_size = hidden_size
        self.num_slots = num_slots
        self.num_heads = num_heads
        
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        self.key_head_dim = key_dim // num_heads
        self.value_head_dim = value_dim // num_heads

        # Learnable memory keys and values
        self.memory_keys = nn.Parameter(
            torch.randn(num_slots, key_dim) * 0.02
        )
        self.memory_values = nn.Parameter(
            torch.randn(num_slots, value_dim) * 0.02
        )

        # Query projection for input hidden states (projects to key dimension)
        self.q_proj = nn.Linear(hidden_size, key_dim, bias=False)
        # Output projection after cross-attention
        self.out_proj = nn.Linear(value_dim, hidden_size, bias=False)

    def forward(self, hidden_states):
        """
        Query the memory bank with input hidden states.

        Args:
            hidden_states: (B*S, D) flattened token representations

        Returns:
            memory_output: (B*S, D) retrieved memory content
            mem_loss: Scalar memory usage entropy loss
        """
        num_tokens = hidden_states.shape[0]

        # Project queries
        queries = self.q_proj(hidden_states)  # (B*S, D)

        # Reshape for multi-head attention
        queries = queries.view(num_tokens, self.num_heads, self.key_head_dim)  # (B*S, H, K_head_dim)
        keys = self.memory_keys.view(self.num_slots, self.num_heads, self.key_head_dim)  # (M, H, K_head_dim)
        values = self.memory_values.view(self.num_slots, self.num_heads, self.value_head_dim)  # (M, H, V_head_dim)

        # Transpose for batched matmul: (H, B*S, K_head_dim) and (H, M, K_head_dim)
        queries = queries.permute(1, 0, 2)  # (H, B*S, K_head_dim)
        keys = keys.permute(1, 0, 2)        # (H, M, K_head_dim)
        values = values.permute(1, 0, 2)     # (H, M, V_head_dim)

        # Scaled dot-product attention: Q @ K^T / sqrt(d)
        scale = math.sqrt(self.key_head_dim)
        attn_weights = torch.bmm(queries, keys.transpose(1, 2)) / scale  # (H, B*S, M)
        attn_weights = F.softmax(attn_weights / self.memory_softmax_temp, dim=-1)

        # Calculate entropy of memory bank slot usage
        # attn_weights shape: (H, B*S, M)
        # Average probability of each slot across heads and tokens in this forward pass
        p = attn_weights.mean(dim=(0, 1))  # (M,)
        eps = 1e-8
        entropy = -torch.sum(p * torch.log(p + eps))
        mem_loss = -entropy

        # Calculate retrieval statistics
        top1_indices = attn_weights.argmax(dim=-1)  # (H, B*S)
        unique_slots = top1_indices.unique().numel()
        top1_prob = p.max()
        top10_mass = torch.topk(p, min(10, self.num_slots)).values.sum()

        mem_stats = {
            "entropy": entropy,
            "top1_prob": top1_prob,
            "top10_mass": top10_mass,
            "unique_slots": torch.tensor(float(unique_slots), device=entropy.device)
        }

        # Weighted sum of values
        attn_output = torch.bmm(attn_weights, values)  # (H, B*S, Vh_dim)

        # Reshape back: (B*S, H, Vh_dim) → (B*S, value_dim)
        attn_output = attn_output.permute(1, 0, 2).contiguous()  # (B*S, H, Vh_dim)
        attn_output = attn_output.view(num_tokens, self.value_dim)  # (B*S, value_dim)

        # Output projection (projects value_dim back to hidden_size)
        memory_output = self.out_proj(attn_output)  # (B*S, D)

        return memory_output, mem_loss, mem_stats


# ===========================================================================
# SwiGLU FFN (identical structure to base model, independent weights)
# ===========================================================================
class ExpertSwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward network for a single memory expert.

    SwiGLU(x) = (SiLU(xW_gate) ⊙ xW_up) W_down

    Each expert has its own independent set of weights, allowing
    specialization in different domains.
    """

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ===========================================================================
# Memory Expert — Memory Bank + Cross-Attention + SwiGLU FFN
# ===========================================================================
class MemoryExpert(nn.Module):
    """
    A single Memory Expert combining:
        1. Memory Bank retrieval (cross-attention to learnable KV store)
        2. SwiGLU FFN (domain-specific transformation)

    Architecture:
        x → RMSNorm → MemoryBank(cross-attn) → + residual
          → RMSNorm → SwiGLU FFN → + residual → output

    Args:
        config: MemoryMoEConfig instance
        expert_idx: Index of this expert (for identification)
    """

    def __init__(self, config, expert_idx=0):
        super().__init__()
        self.cfg = config
        self.expert_idx = expert_idx

        # Memory bank with cross-attention retrieval
        self.memory_norm = RMSNorm(config.expert_hidden_size, eps=config.rms_norm_eps)
        
        # Support separate key and value dimensions from config
        key_dim = getattr(config, "memory_key_dim", 128)
        value_dim = getattr(config, "memory_value_dim", 512)
        
        self.memory_bank = MemoryBank(
            hidden_size=config.expert_hidden_size,
            num_slots=config.memory_bank_size,
            num_heads=config.memory_heads,
            key_dim=key_dim,
            value_dim=value_dim,
            memory_softmax_temp=getattr(config, "memory_softmax_temp", 1.0)
        )

        # Influence Gate: σ(W[h; m])
        self.influence_gate = nn.Linear(config.expert_hidden_size * 2, config.expert_hidden_size)

        # SwiGLU FFN
        self.ffn_norm = RMSNorm(config.expert_hidden_size, eps=config.rms_norm_eps)
        self.ffn = ExpertSwiGLUFFN(
            hidden_size=config.expert_hidden_size,
            intermediate_size=config.expert_intermediate_size,
        )

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (N, D) tokens routed to this expert

        Returns:
            output: (N, D) expert-processed representations
            mem_loss: Scalar memory usage entropy loss
            gate_loss: Scalar L2 penalty of gate values
            mem_stats: dict of retrieval stats
        """
        # Memory retrieval with residual and gate
        residual = hidden_states
        hidden_states = self.memory_norm(hidden_states)
        memory_out, mem_loss, mem_stats = self.memory_bank(hidden_states)
        
        # Influence gate
        gate_input = torch.cat([residual, memory_out], dim=-1)
        alpha = self.cfg.gate_bias + torch.sigmoid(self.influence_gate(gate_input))
        hidden_states = residual + alpha * memory_out
        
        # L2 gate loss
        gate_loss = alpha.pow(2).mean()

        # SwiGLU FFN with residual
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        ffn_out = self.ffn(hidden_states)
        hidden_states = residual + ffn_out

        return hidden_states, mem_loss, gate_loss, mem_stats


# ===========================================================================
# Memory Router — Top-K Gating for Memory Experts
# ===========================================================================
class MemoryRouter(nn.Module):
    """
    Top-K router for the 8-way Memory MoE.

    Routes each token to the top-K memory experts based on a learned
    gating function. Includes load-balancing and z-loss for stability.

    Architecture identical to the base OPTIMUS MoERouter but configured
    for 8 experts with top-2 routing.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_memory_experts
        self.experts_per_token = config.experts_per_token
        self.aux_loss_coef = config.router_aux_loss_coef
        self.z_loss_coef = config.router_z_loss_coef

        # Router gate: projects hidden_size → num_experts
        self.gate = nn.Linear(config.expert_hidden_size, config.num_memory_experts, bias=False)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (B*S, D) flattened token representations

        Returns:
            router_weights: (B*S, top_k) normalized weights for selected experts
            selected_experts: (B*S, top_k) indices of selected experts
            aux_loss: Scalar auxiliary loss
        """
        router_logits = self.gate(hidden_states)  # (B*S, num_experts)

        aux_loss = torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32)

        if self.training:
            # Z-loss: penalize large logits
            log_z = torch.logsumexp(router_logits.float(), dim=-1)
            z_loss = (log_z ** 2).mean()
            aux_loss = aux_loss + self.z_loss_coef * z_loss

            # Load balancing loss
            routing_probs = F.softmax(router_logits.float(), dim=-1)

        # Top-K selection
        router_weights, selected_experts = torch.topk(
            router_logits, self.experts_per_token, dim=-1
        )
        router_weights = F.softmax(router_weights.float(), dim=-1).to(hidden_states.dtype)

        if self.training:
            num_tokens = hidden_states.shape[0]
            expert_mask = F.one_hot(
                selected_experts, num_classes=self.num_experts
            ).float()
            tokens_per_expert = expert_mask.sum(dim=1).mean(dim=0)
            prob_per_expert = routing_probs.mean(dim=0)
            balance_loss = self.num_experts * (tokens_per_expert * prob_per_expert).sum()
            aux_loss = aux_loss + self.aux_loss_coef * balance_loss

        return router_weights, selected_experts, aux_loss


# ===========================================================================
# Memory MoE Layer — Router + Experts + Merge
# ===========================================================================
class MemoryMoELayer(nn.Module):
    """
    Full Memory MoE layer that sits on top of the frozen base model.

    Architecture:
        hidden_states (from frozen base)
            → MemoryRouter (select top-2 experts)
            → Dispatch tokens to selected MemoryExperts
            → Weighted merge of expert outputs
            → Output projection + residual connection

    Args:
        config: MemoryMoEConfig instance
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_memory_experts
        self.experts_per_token = config.experts_per_token
        self.hidden_size = config.expert_hidden_size

        # Router
        self.router = MemoryRouter(config)

        # 8 Memory Experts
        self.experts = nn.ModuleList([
            MemoryExpert(config, expert_idx=i)
            for i in range(config.num_memory_experts)
        ])

        # Output projection with residual
        self.output_norm = RMSNorm(config.expert_hidden_size, eps=config.rms_norm_eps)
        self.output_proj = nn.Linear(config.expert_hidden_size, config.expert_hidden_size, bias=False)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (B, S, D) from frozen base model

        Returns:
            output: (B, S, D) MoE-processed representations
            aux_loss: Router auxiliary loss (load-balancing + z-loss)
            diversity_loss: Cosine similarity expert diversity loss
            total_memory_loss: Averaged memory usage entropy loss
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Save for residual connection
        residual = hidden_states

        # Flatten for token-level routing
        flat_hidden = hidden_states.view(-1, hidden_dim)  # (B*S, D)

        # Route tokens to experts
        router_weights, selected_experts, aux_loss = self.router(flat_hidden)

        # Compute expert outputs
        routed_output = torch.zeros_like(flat_hidden)
        total_memory_loss = torch.tensor(0.0, device=flat_hidden.device, dtype=torch.float32)
        total_gate_loss = torch.tensor(0.0, device=flat_hidden.device, dtype=torch.float32)
        
        sum_entropy = torch.tensor(0.0, device=flat_hidden.device, dtype=torch.float32)
        sum_top1_prob = torch.tensor(0.0, device=flat_hidden.device, dtype=torch.float32)
        sum_top10_mass = torch.tensor(0.0, device=flat_hidden.device, dtype=torch.float32)
        sum_unique_slots = torch.tensor(0.0, device=flat_hidden.device, dtype=torch.float32)
        
        active_experts = 0

        for expert_idx in range(self.num_experts):
            expert_mask = (selected_experts == expert_idx)

            if not expert_mask.any():
                continue

            token_indices, slot_indices = torch.where(expert_mask)

            expert_input = flat_hidden[token_indices]
            expert_output, expert_mem_loss, expert_gate_loss, expert_mem_stats = self.experts[expert_idx](expert_input)

            weights = router_weights[token_indices, slot_indices].unsqueeze(-1)
            routed_output.index_add_(0, token_indices, expert_output * weights)

            total_memory_loss = total_memory_loss + expert_mem_loss
            total_gate_loss = total_gate_loss + expert_gate_loss
            
            sum_entropy = sum_entropy + expert_mem_stats["entropy"]
            sum_top1_prob = sum_top1_prob + expert_mem_stats["top1_prob"]
            sum_top10_mass = sum_top10_mass + expert_mem_stats["top10_mass"]
            sum_unique_slots = sum_unique_slots + expert_mem_stats["unique_slots"]
            
            active_experts += 1

        avg_mem_stats = {}
        if active_experts > 0:
            total_memory_loss = total_memory_loss / active_experts
            total_gate_loss = total_gate_loss / active_experts
            avg_mem_stats = {
                "entropy": sum_entropy / active_experts,
                "top1_prob": sum_top1_prob / active_experts,
                "top10_mass": sum_top10_mass / active_experts,
                "unique_slots": sum_unique_slots / active_experts
            }
        else:
            avg_mem_stats = {
                "entropy": torch.tensor(0.0, device=flat_hidden.device),
                "top1_prob": torch.tensor(0.0, device=flat_hidden.device),
                "top10_mass": torch.tensor(0.0, device=flat_hidden.device),
                "unique_slots": torch.tensor(0.0, device=flat_hidden.device)
            }

        # Output projection + residual
        routed_output = self.output_norm(routed_output)
        routed_output = self.output_proj(routed_output)

        output = routed_output.view(batch_size, seq_len, hidden_dim)
        output = output + residual  # Skip connection from base

        # Expert diversity loss via FFN weights is disabled
        diversity_loss = torch.tensor(0.0, device=flat_hidden.device)

        return output, aux_loss, diversity_loss, total_memory_loss, total_gate_loss, avg_mem_stats


# ===========================================================================
# OPTIMUS_moe — Full Model (Frozen Base + Memory MoE + LM Head)
# ===========================================================================
class OptimusMoEModel(nn.Module):
    """
    OPTIMUS_moe: Memory Mixture of Experts on Frozen OPTIMUS Base.

    Architecture:
        Token Embedding (frozen)
        → [8× Transformer Blocks] (frozen)
        → RMSNorm (frozen)
        → MemoryMoELayer (TRAINABLE — router + 8 experts)
        → LM Head (frozen)

    The frozen base provides rich contextual representations. The Memory
    MoE layer adds domain-specific expertise through learnable memory
    banks and specialized FFNs.

    Args:
        moe_config: MemoryMoEConfig instance
        base_model: Pre-loaded OptimusForCausalLM (will be frozen)
    """

    def __init__(self, moe_config, base_model=None):
        super().__init__()
        self.moe_config = moe_config

        # Store the base model (will be frozen)
        if base_model is not None:
            self.base_model = base_model
        else:
            # Build a fresh base model (caller must load weights)
            base_config = OptimusConfig()
            self.base_model = OptimusForCausalLM(base_config)

        # Freeze all base model parameters
        self._freeze_base()

        # Memory MoE layer (TRAINABLE)
        self.memory_moe = MemoryMoELayer(moe_config)

        # Initialize MoE weights
        self.memory_moe.apply(self._init_moe_weights)

    def _freeze_base(self):
        """Freeze all base model parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

    def _init_moe_weights(self, module):
        """Initialize MoE weights with scaled normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def get_num_params(self):
        """Return parameter counts: total, trainable, frozen."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return total, trainable, frozen

    def get_trainable_params(self):
        """Return only trainable parameters (for optimizer)."""
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, input_ids, position_ids=None, attention_mask=None,
                past_key_values=None, labels=None):
        """
        Full forward pass: frozen base → Memory MoE → LM Head.

        Args:
            input_ids: (B, S) token indices
            position_ids: (B, S) position indices
            attention_mask: (B, S) padding mask
            past_key_values: KV-cache (for generation)
            labels: (B, S) target tokens for loss

        Returns:
            dict with: loss, logits, past_key_values, router_loss, diversity_loss, memory_loss
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # --- Frozen base forward pass (no gradients) ---
        with torch.no_grad():
            # Auto-generate position IDs
            if position_ids is None:
                if past_key_values is not None and past_key_values[0] is not None:
                    past_len = past_key_values[0][0].shape[1]
                    position_ids = torch.arange(
                        past_len, past_len + seq_len, dtype=torch.long, device=device
                    ).unsqueeze(0).expand(bsz, -1)
                else:
                    position_ids = torch.arange(
                        0, seq_len, dtype=torch.long, device=device
                    ).unsqueeze(0).expand(bsz, -1)

            # Process attention mask
            sdpa_mask = None
            if attention_mask is not None and attention_mask.dtype != torch.bool:
                if not attention_mask.all():
                    sdpa_mask = None

            # Embed tokens
            hidden_states = self.base_model.embed_tokens(input_ids)

            # Pass through frozen transformer blocks
            next_cache = []
            for i, layer in enumerate(self.base_model.layers):
                past_kv = past_key_values[i] if past_key_values is not None else None
                hidden_states, present_kv, _ = layer(
                    hidden_states,
                    position_ids=position_ids,
                    attention_mask=sdpa_mask,
                    past_key_value=past_kv,
                    rotary_emb=self.base_model.rotary_emb,
                )
                next_cache.append(present_kv)

            # Final norm (frozen)
            hidden_states = self.base_model.norm(hidden_states)

        # --- Memory MoE layer (TRAINABLE — gradients flow here) ---
        # Detach from base computation graph, re-enable gradients
        hidden_states = hidden_states.detach().requires_grad_(True)

        moe_output, router_loss, diversity_loss, memory_loss, gate_loss, mem_stats = self.memory_moe(hidden_states)

        # --- LM Head (frozen weights, but gradients flow through for MoE) ---
        logits = self.base_model.lm_head(moe_output)

        # --- Compute loss ---
        loss = None
        ce_loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, self.base_model.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            
            # Loss coefficients
            d_coef = getattr(self.moe_config, "diversity_loss_coef", 0.0)
            m_coef = getattr(self.moe_config, "memory_loss_coef", 0.005)
            g_coef = getattr(self.moe_config, "gate_loss_coef", 0.001)
            
            # Step-based decay of memory entropy and gate loss coefficients:
            # 0 - 2000 steps: constant at peak coefficients
            # 2000 - 10000 steps: linearly decay to 0.0
            # > 10000 steps: 0.0
            global_step = int(os.environ.get("OPTIMUS_GLOBAL_STEP", "0"))
            warmup = getattr(self.moe_config, "memory_entropy_warmup_steps", 2000)
            decay_steps = getattr(self.moe_config, "memory_entropy_decay_steps", 8000)
            if global_step < warmup:
                current_m_coef = m_coef
                current_g_coef = g_coef
            elif global_step < warmup + decay_steps:
                decay_factor = 1.0 - (global_step - warmup) / decay_steps
                current_m_coef = m_coef * decay_factor
                current_g_coef = g_coef * decay_factor
            else:
                current_m_coef = 0.0
                current_g_coef = 0.0
            
            # Calculate composite loss
            loss = ce_loss + router_loss + d_coef * diversity_loss + current_m_coef * memory_loss + current_g_coef * gate_loss

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": next_cache,
            "ce_loss": ce_loss,
            "router_loss": router_loss,
            "diversity_loss": diversity_loss,
            "memory_loss": memory_loss,
            "gate_loss": gate_loss,
            "mem_stats": mem_stats,
        }


# ===========================================================================
# Model Loading Utilities
# ===========================================================================

def load_base_model(checkpoint_path, device="cpu"):
    """
    Load the frozen OPTIMUS base model from a checkpoint.

    Args:
        checkpoint_path: Path to the base model .pt checkpoint
        device: Device to load onto

    Returns:
        model: OptimusForCausalLM with loaded weights
    """
    print(f"📂 Loading base OPTIMUS model: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config = OptimusConfig()
    if "model_config" in checkpoint:
        for k, v in checkpoint["model_config"].items():
            setattr(config, k, v)

    model = OptimusForCausalLM(config)

    # Handle torch.compile state_dict keys
    state_dict = checkpoint["model_state_dict"]
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("_orig_mod.", "")] = v

    model.load_state_dict(cleaned)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Base model params: {total_params:,}")

    if "global_step" in checkpoint:
        print(f"   Trained steps:    {checkpoint['global_step']}")
    if "best_val_loss" in checkpoint:
        print(f"   Best val loss:    {checkpoint['best_val_loss']:.4f}")

    return model


def build_optimus_moe(moe_config, device="cuda"):
    """
    Build the full OPTIMUS_moe model: load frozen base + attach Memory MoE.

    Args:
        moe_config: MemoryMoEConfig instance
        device: Device to place the model on

    Returns:
        model: OptimusMoEModel ready for training
    """
    # Load base model
    base_model = load_base_model(moe_config.base_checkpoint, device="cpu")

    # Build full model
    print("\n🏗️  Building OPTIMUS_moe (Memory MoE on frozen base)...")
    model = OptimusMoEModel(moe_config, base_model=base_model)
    model = model.to(device)

    total, trainable, frozen = model.get_num_params()
    print(f"   Total parameters:     {total:,}")
    print(f"   Trainable (MoE):      {trainable:,}")
    print(f"   Frozen (Base):        {frozen:,}")
    print(f"   Memory Experts:       {moe_config.num_memory_experts}")
    print(f"   Experts per token:    {moe_config.experts_per_token}")
    print(f"   Memory bank slots:    {moe_config.memory_bank_size}")
    print(f"   Expert FFN dim:       {moe_config.expert_intermediate_size}")

    return model


# ===========================================================================
# Modular Checkpoint Save/Load
# ===========================================================================

def save_moe_checkpoints(model, optimizer, scheduler, scaler, global_step,
                         best_val_loss, val_loss, moe_config, training_config,
                         checkpoint_dir, is_best=False):
    """
    Save MoE checkpoints modularly:
        - moe_full_latest.pt / moe_full_best.pt — Full state for resuming
        - router.pt — Router weights only
        - expert_0.pt through expert_7.pt — Each expert separately
    """
    import time as _time

    os.makedirs(checkpoint_dir, exist_ok=True)

    moe_layer = model.memory_moe

    # --- Full MoE state (for resuming training) ---
    full_state = {
        "moe_state_dict": moe_layer.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "val_loss": val_loss,
        "moe_config": moe_config.__dict__,
        "training_config": {k: v for k, v in training_config.__dict__.items()},
    }
    if scaler is not None and scaler.is_enabled():
        full_state["scaler_state_dict"] = scaler.state_dict()

    suffix = "best" if is_best else "latest"
    full_path = os.path.join(checkpoint_dir, f"moe_full_{suffix}.pt")
    _atomic_save(full_state, full_path)

    # --- Router weights ---
    router_path = os.path.join(checkpoint_dir, "router.pt")
    torch.save(moe_layer.router.state_dict(), router_path)

    # --- Individual expert weights ---
    for i, expert in enumerate(moe_layer.experts):
        expert_path = os.path.join(checkpoint_dir, f"expert_{i}.pt")
        torch.save(expert.state_dict(), expert_path)

    return full_path


def load_moe_checkpoint(filepath, model, optimizer=None, scheduler=None, scaler=None):
    """Load a full MoE checkpoint for resuming training."""
    checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)

    model.memory_moe.load_state_dict(checkpoint["moe_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint


def _atomic_save(state_dict, filepath):
    """Save a checkpoint atomically (write to tmp, then rename)."""
    import time as _time

    tmp_path = filepath + ".tmp"
    torch.save(state_dict, tmp_path)

    for attempt in range(60):
        try:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except PermissionError:
                    pass
            os.replace(tmp_path, filepath)
            break
        except PermissionError:
            _time.sleep(1)
            if attempt == 59:
                raise
