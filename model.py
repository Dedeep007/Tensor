"""
OPTIMUS Model Architecture
===========================
Full PyTorch implementation of the OPTIMUS causal language model with
Mixture of Experts (MoE).

Components:
    - RMSNorm (Pre-normalization)
    - Rotary Position Embeddings (RoPE)
    - Grouped Query Attention (GQA) with KV-cache
    - SwiGLU Feed-Forward Network (dense FFN)
    - MoE Router (Top-K gating with load-balancing & z-loss)
    - MoE Layer (routed experts + shared experts, DeepSeek-V2 style)
    - OptimusForCausalLM (end-to-end model with loss computation)

Design lineage: Qwen2-MoE, DeepSeek-V2, Mixtral, Baichuan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ===========================================================================
# RMS Normalization
# ===========================================================================
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Unlike standard LayerNorm, RMSNorm omits mean-centering and only
    normalizes by the root mean square of the input. This is computationally
    cheaper and has been shown to perform comparably or better.

    Forward pass computed in FP32 for numerical stability, then cast back.
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


# ===========================================================================
# Rotary Position Embeddings (RoPE)
# ===========================================================================
class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (Su et al., 2021).

    Encodes absolute position via rotation matrices while naturally capturing
    relative position dependencies. The rotation is applied to pairs of
    dimensions in the query and key vectors.

    Uses precomputed cos/sin caches for efficiency.
    """

    def __init__(self, dim, max_seq_len, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len

        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, position_ids):
        cos = self.cos_cached[position_ids].unsqueeze(2)
        sin = self.sin_cached[position_ids].unsqueeze(2)
        return cos, sin


def _rotate_half(x):
    """Rotate half the hidden dims of the input: [-x2, x1]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to query and key tensors."""
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


# ===========================================================================
# SwiGLU Feed-Forward Network
# ===========================================================================
class SwiGLUFFN(nn.Module):
    """
    SwiGLU activation feed-forward network.

    SwiGLU(x) = (SiLU(xW_gate) . xW_up) W_down

    Used as the dense FFN for non-MoE layers AND as individual expert FFNs
    inside the MoE layer. The `hidden_size` and `intermediate_size` can be
    configured independently to support smaller per-expert dimensions.
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
# MoE Router (Top-K Gating)
# ===========================================================================
class MoERouter(nn.Module):
    """
    Top-K router for Mixture of Experts.

    For each token, computes a probability distribution over all experts
    and selects the top-K experts to process that token. Includes two
    auxiliary losses for training stability:

    1. **Load Balancing Loss** (Switch Transformer):
       Encourages even distribution of tokens across experts to prevent
       expert collapse (all tokens routed to one expert).

       L_balance = N * sum_i(f_i * P_i)
       where f_i = fraction of tokens routed to expert i
             P_i = mean routing probability for expert i

    2. **Router Z-Loss** (ST-MoE):
       Penalizes large router logits to improve training stability.

       L_z = (1/B) * sum(log(sum(exp(z)))^2)
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.aux_loss_coef = config.router_aux_loss_coef
        self.z_loss_coef = config.router_z_loss_coef

        # Router gate: projects hidden_size -> num_experts
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (B*S, D) flattened token representations

        Returns:
            router_weights: (B*S, top_k) normalized weights for selected experts
            selected_experts: (B*S, top_k) indices of selected experts
            aux_loss: Scalar auxiliary loss (load balancing + z-loss)
        """
        # Compute router logits
        router_logits = self.gate(hidden_states)  # (B*S, num_experts)

        # --- Auxiliary Losses (computed before top-k for full distribution) ---
        aux_loss = torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32)

        if self.training:
            # Z-loss: penalize large logits for stability
            log_z = torch.logsumexp(router_logits.float(), dim=-1)  # (B*S,)
            z_loss = (log_z ** 2).mean()
            aux_loss = aux_loss + self.z_loss_coef * z_loss

            # Load balancing loss
            routing_probs = F.softmax(router_logits.float(), dim=-1)  # (B*S, E)

        # Top-K selection
        router_weights, selected_experts = torch.topk(
            router_logits, self.num_experts_per_tok, dim=-1
        )
        # Normalize weights across selected experts (softmax over top-k)
        router_weights = F.softmax(router_weights.float(), dim=-1).to(hidden_states.dtype)

        if self.training:
            # Load balancing: f_i (token fraction) * P_i (mean prob)
            # One-hot encode selected experts
            num_tokens = hidden_states.shape[0]
            expert_mask = F.one_hot(
                selected_experts, num_classes=self.num_experts
            ).float()  # (B*S, top_k, E)
            # f_i: fraction of tokens assigned to each expert
            tokens_per_expert = expert_mask.sum(dim=1).mean(dim=0)  # (E,)
            # P_i: mean routing probability for each expert
            prob_per_expert = routing_probs.mean(dim=0)  # (E,)
            # Balance loss
            balance_loss = self.num_experts * (tokens_per_expert * prob_per_expert).sum()
            aux_loss = aux_loss + self.aux_loss_coef * balance_loss

        return router_weights, selected_experts, aux_loss


# ===========================================================================
# Mixture of Experts Layer (DeepSeek-V2 Style)
# ===========================================================================
class OptimusMoELayer(nn.Module):
    """
    Mixture of Experts FFN layer with shared experts.

    Architecture (DeepSeek-V2 / Qwen2-MoE style):
        1. A router selects top-K experts per token
        2. Selected expert FFNs process the token, weighted by router scores
        3. A shared expert FFN processes ALL tokens (always active)
        4. Output = shared_output + weighted sum of routed expert outputs

    This design ensures:
        - Every token gets a baseline representation (shared expert)
        - Specialized knowledge is distributed across routed experts
        - Active compute per token stays bounded (top-K + shared)

    Args:
        config: OptimusConfig with MoE parameters
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.hidden_size = config.hidden_size

        # Router
        self.router = MoERouter(config)

        # Routed experts: each is a smaller SwiGLU FFN
        self.experts = nn.ModuleList([
            SwiGLUFFN(config.hidden_size, config.expert_intermediate_size)
            for _ in range(config.num_experts)
        ])

        # Shared expert(s): always active, full-sized or expert-sized
        self.shared_experts = nn.ModuleList([
            SwiGLUFFN(config.hidden_size, config.expert_intermediate_size)
            for _ in range(config.num_shared_experts)
        ]) if config.num_shared_experts > 0 else None

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (B, S, D) input tensor

        Returns:
            output: (B, S, D) combined expert outputs
            aux_loss: Scalar router auxiliary loss
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Flatten to (B*S, D) for token-level routing
        flat_hidden = hidden_states.view(-1, hidden_dim)
        num_tokens = flat_hidden.shape[0]

        # Route tokens to experts
        router_weights, selected_experts, aux_loss = self.router(flat_hidden)
        # router_weights: (B*S, top_k)
        # selected_experts: (B*S, top_k)

        # --- Compute routed expert outputs ---
        # Initialize output accumulator
        routed_output = torch.zeros_like(flat_hidden)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find which tokens selected this expert (across all top-k slots)
            # expert_mask: (B*S, top_k) boolean
            expert_mask = (selected_experts == expert_idx)

            if not expert_mask.any():
                continue

            # Get token indices and their corresponding top-k slot indices
            token_indices, slot_indices = torch.where(expert_mask)

            # Gather the tokens for this expert
            expert_input = flat_hidden[token_indices]  # (num_selected, D)

            # Run expert FFN
            expert_output = self.experts[expert_idx](expert_input)  # (num_selected, D)

            # Weight by router score and accumulate
            weights = router_weights[token_indices, slot_indices].unsqueeze(-1)  # (num_selected, 1)
            routed_output.index_add_(0, token_indices, expert_output * weights)

        # --- Compute shared expert output ---
        if self.shared_experts is not None:
            shared_output = torch.zeros_like(flat_hidden)
            for shared_expert in self.shared_experts:
                shared_output = shared_output + shared_expert(flat_hidden)
            # Combine: shared + routed
            final_output = shared_output + routed_output
        else:
            final_output = routed_output

        # Reshape back to (B, S, D)
        output = final_output.view(batch_size, seq_len, hidden_dim)
        return output, aux_loss


# ===========================================================================
# Grouped Query Attention (GQA) with KV-Cache
# ===========================================================================
class OptimusAttention(nn.Module):
    """
    Multi-head attention with Grouped Query Attention (GQA).

    GQA shares Key-Value heads across multiple Query heads, reducing the
    KV-cache memory footprint during inference. With num_attention_heads=16
    and num_key_value_heads=4, each KV head serves 4 query heads.

    Uses PyTorch's native scaled_dot_product_attention which auto-routes
    to FlashAttention-2 on supported CUDA hardware.
    """

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.hidden_size // self.num_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states, position_ids, attention_mask=None,
                past_key_value=None, rotary_emb=None):
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim)

        if rotary_emb is not None:
            cos, sin = rotary_emb(position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)
        present_key_value = (k, v)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=2)
            v = v.repeat_interleave(self.num_kv_groups, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        use_causal = past_key_value is None and q_len > 1

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask if not use_causal else None,
            is_causal=use_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_output), present_key_value


# ===========================================================================
# Transformer Block (Pre-RMSNorm with MoE or Dense FFN)
# ===========================================================================
class OptimusBlock(nn.Module):
    """
    Single transformer block with Pre-RMSNorm architecture.

    Supports two FFN modes:
        - **Dense**: Standard SwiGLU FFN (used for non-MoE layers)
        - **MoE**: Mixture of Experts with routing (used for MoE layers)

    Architecture:
        x -> RMSNorm -> Attention -> + residual
          -> RMSNorm -> [MoE / Dense FFN] -> + residual
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = OptimusAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Decide MoE vs Dense for this layer
        use_moe_this_layer = (
            config.use_moe
            and (layer_idx % config.moe_layer_freq == 0)
        )

        if use_moe_this_layer:
            self.mlp = OptimusMoELayer(config)
            self.is_moe = True
        else:
            self.mlp = SwiGLUFFN(config.hidden_size, config.intermediate_size)
            self.is_moe = False

    def forward(self, hidden_states, position_ids, attention_mask=None,
                past_key_value=None, rotary_emb=None):
        # Pre-Norm Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states, position_ids, attention_mask, past_key_value, rotary_emb
        )
        hidden_states = residual + hidden_states

        # Pre-Norm FFN (MoE or Dense)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        aux_loss = None
        if self.is_moe:
            hidden_states, aux_loss = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states, present_key_value, aux_loss


# ===========================================================================
# OPTIMUS Causal Language Model (with MoE)
# ===========================================================================
class OptimusForCausalLM(nn.Module):
    """
    OPTIMUS: End-to-end causal language model with Mixture of Experts.

    Architecture:
        Token Embedding -> [N x OptimusBlock (MoE/Dense)] -> RMSNorm -> LM Head

    Features:
        - Grouped Query Attention (GQA) with RoPE
        - Mixture of Experts with top-K routing + shared experts
        - SwiGLU FFN with Pre-RMSNorm
        - Load-balancing and z-loss for router stability
        - KV-cache support for efficient autoregressive generation
        - Cross-entropy loss with automatic label shifting

    Parameter accounting (MoE enabled, default config):
        - Total params:  ~1.2B (all expert weights counted)
        - Active params: ~500M per token (top-2 experts + shared + attention)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )

        # Rotary embeddings (shared across all layers)
        self.rotary_emb = RotaryEmbedding(
            dim=config.hidden_size // config.num_attention_heads,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_theta,
        )

        # Transformer blocks (each decides MoE vs Dense internally)
        self.layers = nn.ModuleList(
            [OptimusBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )

        # Final normalization
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Language model head (untied from embeddings)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with scaled normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def get_num_params(self, count_active=False):
        """
        Return parameter counts.

        Args:
            count_active: If True, estimate active params per token
                          (only top-K experts + shared, not all experts).
        """
        if not count_active:
            return sum(p.numel() for p in self.parameters())

        # Active params: everything except non-selected expert FFNs
        active = 0
        for name, param in self.named_parameters():
            if "experts." in name and "shared_experts" not in name:
                # Routed expert: only count top-K fraction
                # Each expert is counted as (num_experts_per_tok / num_experts) fraction
                active += param.numel() * self.config.num_experts_per_tok / self.config.num_experts
            else:
                active += param.numel()
        return int(active)

    def forward(self, input_ids, position_ids=None, attention_mask=None,
                past_key_values=None, labels=None):
        """
        Args:
            input_ids: (B, S) token indices
            position_ids: (B, S) position indices (auto-generated if None)
            attention_mask: (B, S) padding mask (1 = real token, 0 = padding)
            past_key_values: List of (K, V) tuples for KV-cache
            labels: (B, S) target token ids for loss computation

        Returns:
            dict with keys: "loss", "logits", "past_key_values", "aux_loss"
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

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
        hidden_states = self.embed_tokens(input_ids)

        # Pass through transformer blocks, accumulating MoE auxiliary losses
        next_cache = []
        total_aux_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        num_moe_layers = 0

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            hidden_states, present_kv, layer_aux_loss = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=sdpa_mask,
                past_key_value=past_kv,
                rotary_emb=self.rotary_emb,
            )
            next_cache.append(present_kv)

            if layer_aux_loss is not None:
                total_aux_loss = total_aux_loss + layer_aux_loss
                num_moe_layers += 1

        # Average auxiliary loss across MoE layers
        if num_moe_layers > 0:
            total_aux_loss = total_aux_loss / num_moe_layers

        # Final norm + LM head
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            # Total loss = cross-entropy + router auxiliary loss
            loss = ce_loss + total_aux_loss

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": next_cache,
            "aux_loss": total_aux_loss,
        }
