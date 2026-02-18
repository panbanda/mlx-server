# Qwen3-Coder-Next Architecture

## Overview

Hybrid SSM/attention transformer with Mixture of Experts (MoE).

- **48 layers** total
- Every 4th layer (3, 7, 11, ..., 47) uses full attention
- All other layers use GatedDeltaNet (SSM-like linear attention)
- ALL layers use MoE (decoder_sparse_step=1)

## Key Config Values (4-bit quantized)

```
hidden_size: 2048
head_dim: 256 (explicit, not computed from hidden_size/num_heads)
num_attention_heads: 16
num_key_value_heads: 2
partial_rotary_factor: 0.25 (64 dims of 256 get RoPE)
linear_num_key_heads: 16
linear_num_value_heads: 32
linear_key_head_dim: 128
linear_value_head_dim: 128
linear_conv_kernel_dim: 4
num_experts: 512
num_experts_per_tok: 10
moe_intermediate_size: 512
shared_expert_intermediate_size: 512
intermediate_size: 5120
full_attention_interval: 4
vocab_size: 151936
```

## Architecture Components

### Full Attention (Qwen3NextAttention)

- Q projected to `2 * num_heads * head_dim` (doubled for gating)
- Split into queries and gate
- K, V projected normally with GQA (2 KV heads for 16 Q heads)
- Q/K get per-head RmsNorm before RoPE
- Only 25% of head_dim gets rotary encoding (partial_rotary_factor=0.25)
- Output gated: `o_proj(attn_output * sigmoid(gate))`

### GatedDeltaNet (Linear Attention / SSM)

- `in_proj_qkvz`: projects to `key_dim*2 + value_dim*2`
- `in_proj_ba`: projects to `num_v_heads * 2`
- Splits into q, k, v (for content), z (for gated norm), b (beta/gate), a (decay)
- Conv1d (depthwise, kernel=4) applied to concatenated q,k,v with state padding
- RmsNorm on q,k (without learnable weight, just normalization)
- Gated delta recurrence:
  - `beta = sigmoid(b)`
  - `g = exp(-exp(A_log) * softplus(a + dt_bias))` (decay factor)
  - Sequential state update: `state = state * g + k * delta`
  - Output: `y = sum(state * q, dim=-1)`
- Output through gated RmsNorm (with z as SwiGLU gate) then projection

**Cache:** `ArraysCache(size=2)` storing `[conv_state, ssm_state]`
- conv_state: `[B, kernel-1, conv_dim]`
- ssm_state: `[B, Hv, Dv, Dk]`

### MoE (Qwen3NextSparseMoeBlock)

- Router: Linear(hidden_size, 512) -> softmax -> top-10 selection
- SwitchGLU: stacked expert weights [512, moe_intermediate_size, hidden_size]
  - Uses `gather_mm` (Python) / manual per-expert matmul (Rust)
- Shared expert: standard MLP with sigmoid gate
- Output: `sum(expert_outputs * scores) + sigmoid(shared_gate) * shared_output`

### Weight Sanitization (already done in saved model)

1. Individual expert weights stacked into `switch_mlp.{gate,up,down}_proj.weight`
2. Norm weights shifted by +1.0 (input_layernorm, post_attention_layernorm,
   model.norm, q_norm, k_norm)
3. Conv1d weights axis-reordered from [C, 1, K] to [C, K, 1]

### Per-Layer Quantization

- Most weights: 4-bit, group_size=64
- `mlp.gate` (router) and `mlp.shared_expert_gate`: 8-bit, group_size=64
- Conv1d weights, A_log, dt_bias, norm weights: NOT quantized (float16)

## Safetensors Key Structure

Linear attention layer (e.g., layer 0):
```
model.layers.0.linear_attn.A_log
model.layers.0.linear_attn.dt_bias
model.layers.0.linear_attn.conv1d.weight
model.layers.0.linear_attn.in_proj_qkvz.{weight,scales,biases}
model.layers.0.linear_attn.in_proj_ba.{weight,scales,biases}
model.layers.0.linear_attn.norm.weight
model.layers.0.linear_attn.out_proj.{weight,scales,biases}
```

Full attention layer (e.g., layer 3):
```
model.layers.3.self_attn.q_proj.{weight,scales,biases}
model.layers.3.self_attn.k_proj.{weight,scales,biases}
model.layers.3.self_attn.v_proj.{weight,scales,biases}
model.layers.3.self_attn.o_proj.{weight,scales,biases}
model.layers.3.self_attn.q_norm.weight
model.layers.3.self_attn.k_norm.weight
```

MoE (all layers):
```
model.layers.{i}.mlp.gate.{weight,scales,biases}
model.layers.{i}.mlp.switch_mlp.gate_proj.{weight,scales,biases}
model.layers.{i}.mlp.switch_mlp.up_proj.{weight,scales,biases}
model.layers.{i}.mlp.switch_mlp.down_proj.{weight,scales,biases}
model.layers.{i}.mlp.shared_expert.gate_proj.{weight,scales,biases}
model.layers.{i}.mlp.shared_expert.up_proj.{weight,scales,biases}
model.layers.{i}.mlp.shared_expert.down_proj.{weight,scales,biases}
model.layers.{i}.mlp.shared_expert_gate.{weight,scales,biases}
```

## Implementation Strategy (Rust)

Without `gather_mm`:
- Store expert weights as stacked Param<Array>
- Use `dequantize` + `matmul` for MoE forward (per-expert-batch)
- For decode (T=1): 10 dequantize+matmul ops per projection per layer

Without custom Metal kernel:
- Use ops-based sequential loop for gated_delta_update
- O(T) per layer for prefill, O(1) for decode

Cache:
- Create LayerCache enum: KVCache for attention, ArraysCache for SSM
- Engine manages Vec<Option<LayerCache>>
