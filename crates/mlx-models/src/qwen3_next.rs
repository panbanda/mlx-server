//! Qwen3-Coder-Next model implementation.
//!
//! Hybrid SSM/attention transformer with Mixture of Experts (MoE).
//! Every `full_attention_interval`-th layer uses full attention (Qwen3NextAttention),
//! all other layers use GatedDeltaNet (SSM-like linear attention).
//! All layers use Sparse MoE for the feed-forward block.

use std::path::Path;

use mlx_rs::{
    Array,
    builder::Builder,
    error::Exception,
    fast,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::{self, indexing::IndexOp},
};
use serde::Deserialize;

use crate::{
    cache::{ConcatKeyValueCache, KeyValueCache},
    error::ModelError,
    utils::scaled_dot_product_attention,
};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

fn default_full_attention_interval() -> i32 {
    4
}

fn default_rope_theta() -> f32 {
    10000.0
}

fn default_partial_rotary_factor() -> f32 {
    1.0
}

/// Quantization parameters from config.json (top-level defaults).
#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3NextModelArgs {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    pub max_position_embeddings: i32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,

    // Linear attention (GatedDeltaNet) params
    #[serde(default)]
    pub linear_num_value_heads: i32,
    #[serde(default)]
    pub linear_num_key_heads: i32,
    #[serde(default)]
    pub linear_key_head_dim: i32,
    #[serde(default)]
    pub linear_value_head_dim: i32,
    #[serde(default)]
    pub linear_conv_kernel_dim: i32,

    // MoE params
    #[serde(default)]
    pub num_experts: i32,
    #[serde(default)]
    pub num_experts_per_tok: i32,
    #[serde(default)]
    pub decoder_sparse_step: i32,
    #[serde(default)]
    pub shared_expert_intermediate_size: i32,
    #[serde(default)]
    pub moe_intermediate_size: i32,
    #[serde(default)]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub mlp_only_layers: Vec<i32>,
    #[serde(default = "default_full_attention_interval")]
    pub full_attention_interval: i32,

    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

// ---------------------------------------------------------------------------
// Quantized weight containers
// ---------------------------------------------------------------------------

/// Quantized linear layer stored as raw weight/scales/biases arrays.
/// Forward uses `quantized_matmul` directly.
#[derive(Debug, Clone, ModuleParameters)]
struct QLinear {
    #[param]
    weight: Param<Array>,
    #[param]
    scales: Param<Array>,
    #[param]
    biases: Param<Array>,
    group_size: i32,
    bits: i32,
}

impl QLinear {
    fn new(group_size: i32, bits: i32) -> Result<Self, Exception> {
        Ok(Self {
            weight: Param::new(Array::zeros::<f32>(&[1])?),
            scales: Param::new(Array::zeros::<f32>(&[1])?),
            biases: Param::new(Array::zeros::<f32>(&[1])?),
            group_size,
            bits,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        ops::quantized_matmul(
            x,
            &*self.weight,
            &*self.scales,
            &*self.biases,
            true,
            self.group_size,
            self.bits,
        )
    }
}

/// Quantized embedding stored as raw weight/scales/biases arrays.
#[derive(Debug, Clone, ModuleParameters)]
struct QEmbedding {
    #[param]
    weight: Param<Array>,
    #[param]
    scales: Param<Array>,
    #[param]
    biases: Param<Array>,
    group_size: i32,
    bits: i32,
}

impl QEmbedding {
    fn new(group_size: i32, bits: i32) -> Result<Self, Exception> {
        Ok(Self {
            weight: Param::new(Array::zeros::<f32>(&[1])?),
            scales: Param::new(Array::zeros::<f32>(&[1])?),
            biases: Param::new(Array::zeros::<f32>(&[1])?),
            group_size,
            bits,
        })
    }

    fn forward(&self, indices: &Array) -> Result<Array, Exception> {
        let full = ops::dequantize(
            &*self.weight,
            &*self.scales,
            &*self.biases,
            self.group_size,
            self.bits,
        )?;
        full.take_axis(indices, 0)
    }

    fn as_linear(&self, x: &Array) -> Result<Array, Exception> {
        ops::quantized_matmul(
            x,
            &*self.weight,
            &*self.scales,
            &*self.biases,
            true,
            self.group_size,
            self.bits,
        )
    }
}

// ---------------------------------------------------------------------------
// SwiGLU activation
// ---------------------------------------------------------------------------

/// `silu(gate) * x`
fn swiglu(gate: &Array, x: &Array) -> Result<Array, Exception> {
    nn::silu(gate)?.multiply(x)
}

// ---------------------------------------------------------------------------
// Qwen3NextAttention (full attention with gated Q and partial RoPE)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct Qwen3NextAttention {
    #[param]
    q_proj: QLinear,
    #[param]
    k_proj: QLinear,
    #[param]
    v_proj: QLinear,
    #[param]
    o_proj: QLinear,
    #[param]
    q_norm: nn::RmsNorm,
    #[param]
    k_norm: nn::RmsNorm,
    #[param]
    rope: nn::Rope,
    num_attention_heads: i32,
    num_key_value_heads: i32,
    scale: f32,
}

impl Qwen3NextAttention {
    fn new(args: &Qwen3NextModelArgs, ql: i32, qb: i32) -> Result<Self, Exception> {
        let head_dim = args.head_dim;
        let head_dim_f32 = f32::from(
            i16::try_from(head_dim).map_err(|_| Exception::custom("head_dim out of i16 range"))?,
        );
        let scale = head_dim_f32.sqrt().recip();
        let rope_dim_f32 = f32::from(
            i16::try_from(head_dim).map_err(|_| Exception::custom("head_dim out of i16 range"))?,
        );
        // partial_rotary_factor * head_dim is always a small positive integer (e.g. 64)
        #[allow(clippy::as_conversions)]
        let partial_dim = (rope_dim_f32 * args.partial_rotary_factor).round() as i32;

        Ok(Self {
            q_proj: QLinear::new(ql, qb)?,
            k_proj: QLinear::new(ql, qb)?,
            v_proj: QLinear::new(ql, qb)?,
            o_proj: QLinear::new(ql, qb)?,
            q_norm: nn::RmsNormBuilder::new(head_dim)
                .eps(args.rms_norm_eps)
                .build()?,
            k_norm: nn::RmsNormBuilder::new(head_dim)
                .eps(args.rms_norm_eps)
                .build()?,
            rope: nn::RopeBuilder::new(partial_dim)
                .traditional(false)
                .base(args.rope_theta)
                .scale(1.0)
                .build()
                .map_err(|e| Exception::custom(format!("Failed to build RoPE: {e}")))?,
            num_attention_heads: args.num_attention_heads,
            num_key_value_heads: args.num_key_value_heads,
            scale,
        })
    }

    #[allow(non_snake_case)]
    fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut ConcatKeyValueCache,
    ) -> Result<Array, Exception> {
        let shape = x.shape();
        let B = *shape
            .first()
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;
        let L = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;

        // Q is projected to 2 * num_heads * head_dim (doubled for gating)
        let q_proj_output = self.q_proj.forward(x)?;
        let q_reshaped = q_proj_output.reshape(&[B, L, self.num_attention_heads, -1])?;
        let q_halves = q_reshaped.split(2, Some(-1))?;
        let queries_pre = q_halves
            .first()
            .ok_or_else(|| Exception::custom("split produced empty result"))?;
        let gate = q_halves
            .get(1)
            .ok_or_else(|| Exception::custom("split produced empty result"))?
            .reshape(&[B, L, -1])?;

        let keys_raw = self.k_proj.forward(x)?;
        let values_raw = self.v_proj.forward(x)?;

        // Per-head RmsNorm then transpose to [B, H, L, D]
        let mut queries = self
            .q_norm
            .forward(queries_pre)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut keys = self
            .k_norm
            .forward(&keys_raw.reshape(&[B, L, self.num_key_value_heads, -1])?)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut values = values_raw
            .reshape(&[B, L, self.num_key_value_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // RoPE with cache offset
        let offset = cache.offset();
        let q_input = nn::RopeInputBuilder::new(&queries).offset(offset).build()?;
        queries = self.rope.forward(q_input)?;
        let k_input = nn::RopeInputBuilder::new(&keys).offset(offset).build()?;
        keys = self.rope.forward(k_input)?;

        let (cached_keys, cached_values) = cache.update_and_fetch(keys, values)?;
        keys = cached_keys;
        values = cached_values;

        let output = scaled_dot_product_attention(queries, keys, values, self.scale, mask)?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[B, L, -1])?;

        // Sigmoid gate on output
        let gated = output.multiply(nn::sigmoid(&gate)?)?;
        self.o_proj.forward(&gated)
    }
}

// ---------------------------------------------------------------------------
// Qwen3NextMLP (standard SwiGLU)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct Qwen3NextMLP {
    #[param]
    gate_proj: QLinear,
    #[param]
    down_proj: QLinear,
    #[param]
    up_proj: QLinear,
}

impl Qwen3NextMLP {
    fn new(ql: i32, qb: i32) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: QLinear::new(ql, qb)?,
            down_proj: QLinear::new(ql, qb)?,
            up_proj: QLinear::new(ql, qb)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let gate_out = self.gate_proj.forward(x)?;
        let up_out = self.up_proj.forward(x)?;
        let activated = swiglu(&gate_out, &up_out)?;
        self.down_proj.forward(&activated)
    }
}

// ---------------------------------------------------------------------------
// SwitchMLP weights (stacked expert weights for MoE)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct SwitchMlpWeights {
    #[param]
    gate_proj: QLinear,
    #[param]
    up_proj: QLinear,
    #[param]
    down_proj: QLinear,
}

impl SwitchMlpWeights {
    fn new(ql: i32, qb: i32) -> Result<Self, Exception> {
        Ok(Self {
            gate_proj: QLinear::new(ql, qb)?,
            up_proj: QLinear::new(ql, qb)?,
            down_proj: QLinear::new(ql, qb)?,
        })
    }

    /// Apply one expert's MLP to input x.
    /// expert_idx is a 0-d or 1-element array indexing into the stacked weights.
    fn forward_expert(&self, x: &Array, expert_idx: &Array) -> Result<Array, Exception> {
        let gate_w = self.gate_proj.weight.take_axis(expert_idx, 0)?;
        let gate_s = self.gate_proj.scales.take_axis(expert_idx, 0)?;
        let gate_b = self.gate_proj.biases.take_axis(expert_idx, 0)?;
        let gate_out = ops::quantized_matmul(
            x,
            &gate_w.squeeze_axes(&[0])?,
            &gate_s.squeeze_axes(&[0])?,
            &gate_b.squeeze_axes(&[0])?,
            true,
            self.gate_proj.group_size,
            self.gate_proj.bits,
        )?;

        let up_w = self.up_proj.weight.take_axis(expert_idx, 0)?;
        let up_s = self.up_proj.scales.take_axis(expert_idx, 0)?;
        let up_b = self.up_proj.biases.take_axis(expert_idx, 0)?;
        let up_out = ops::quantized_matmul(
            x,
            &up_w.squeeze_axes(&[0])?,
            &up_s.squeeze_axes(&[0])?,
            &up_b.squeeze_axes(&[0])?,
            true,
            self.up_proj.group_size,
            self.up_proj.bits,
        )?;

        let activated = swiglu(&gate_out, &up_out)?;

        let down_w = self.down_proj.weight.take_axis(expert_idx, 0)?;
        let down_s = self.down_proj.scales.take_axis(expert_idx, 0)?;
        let down_b = self.down_proj.biases.take_axis(expert_idx, 0)?;
        ops::quantized_matmul(
            &activated,
            &down_w.squeeze_axes(&[0])?,
            &down_s.squeeze_axes(&[0])?,
            &down_b.squeeze_axes(&[0])?,
            true,
            self.down_proj.group_size,
            self.down_proj.bits,
        )
    }
}

// ---------------------------------------------------------------------------
// SparseMoeBlock (router + SwitchGLU + shared expert)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct SparseMoeBlock {
    #[param]
    gate: QLinear,
    #[param]
    switch_mlp: SwitchMlpWeights,
    #[param]
    shared_expert: Qwen3NextMLP,
    #[param]
    shared_expert_gate: QLinear,
    top_k: i32,
    norm_topk_prob: bool,
}

impl SparseMoeBlock {
    fn new(args: &Qwen3NextModelArgs, ql: i32, qb: i32) -> Result<Self, Exception> {
        // Router gate and shared_expert_gate use 8-bit quantization
        Ok(Self {
            gate: QLinear::new(64, 8)?,
            switch_mlp: SwitchMlpWeights::new(ql, qb)?,
            shared_expert: Qwen3NextMLP::new(ql, qb)?,
            shared_expert_gate: QLinear::new(64, 8)?,
            top_k: args.num_experts_per_tok,
            norm_topk_prob: args.norm_topk_prob,
        })
    }

    #[allow(non_snake_case)]
    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let shape = x.shape();
        let B = *shape
            .first()
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;
        let L = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;
        let D = *shape
            .get(2)
            .ok_or_else(|| Exception::custom("Input must have >= 3 dims"))?;

        // Router: compute gate scores
        let gates = ops::softmax_axis(&self.gate.forward(x)?, -1, true)?;

        // Top-K selection via argpartition
        let neg_k = -self.top_k;
        let all_inds = ops::argpartition_axis(&gates, neg_k, -1)?;
        // Take last K indices (the top-K)
        let num_experts = *gates
            .shape()
            .last()
            .ok_or_else(|| Exception::custom("gates must have last dim"))?;
        let top_k_start = num_experts - self.top_k;
        let top_inds = all_inds.index((.., .., top_k_start..));
        let raw_scores = gates.take_along_axis(&top_inds, -1)?;

        let top_scores = if self.norm_topk_prob {
            let score_sum = raw_scores.sum_axes(&[-1], true)?;
            raw_scores.divide(score_sum)?
        } else {
            raw_scores
        };

        // Expert computation: loop over K selected experts per token
        let x_flat = x.reshape(&[-1, D])?;
        let n_tokens = B * L;
        let inds_flat = top_inds.reshape(&[-1, self.top_k])?;
        let scores_flat = top_scores.reshape(&[-1, self.top_k])?;

        // Accumulate weighted expert outputs per token
        let n_tokens_usize =
            usize::try_from(n_tokens).map_err(|_| Exception::custom("n_tokens overflow"))?;
        let mut token_results: Vec<Array> = (0..n_tokens_usize)
            .map(|_| Array::zeros::<f32>(&[1, D]))
            .collect::<Result<Vec<_>, _>>()?;

        for ki in 0..self.top_k {
            let token_expert_inds = inds_flat.index((.., ki..ki + 1));
            let token_scores = scores_flat.index((.., ki..ki + 1));

            for ti in 0..n_tokens {
                let ti_usize = usize::try_from(ti).map_err(|_| Exception::custom("ti overflow"))?;
                let token_x = x_flat.index((ti..ti + 1, ..));
                let expert_idx = token_expert_inds.index((ti, ..));
                let score = token_scores.index((ti, 0..1));

                let expert_out = self.switch_mlp.forward_expert(&token_x, &expert_idx)?;
                let weighted = expert_out.multiply(&score.reshape(&[1, 1])?)?;

                let current = token_results
                    .get(ti_usize)
                    .ok_or_else(|| Exception::custom("token index out of bounds"))?;
                let updated = current.add(&weighted)?;
                // Safe replacement via Vec indexing
                if let Some(slot) = token_results.get_mut(ti_usize) {
                    *slot = updated;
                }
            }
        }

        let result_refs: Vec<&Array> = token_results.iter().collect();
        let expert_sum = if n_tokens_usize > 1 {
            ops::concatenate_axis(&result_refs, 0)?
        } else if let Some(single) = token_results.into_iter().next() {
            single
        } else {
            Array::zeros::<f32>(&[n_tokens, D])?
        };

        let y = expert_sum.reshape(&[B, L, D])?;

        // Shared expert
        let shared_y = self.shared_expert.forward(x)?;
        let shared_gate_val = nn::sigmoid(&self.shared_expert_gate.forward(x)?)?;
        let shared_out = shared_y.multiply(&shared_gate_val)?;

        y.add(shared_out)
    }
}

// ---------------------------------------------------------------------------
// GatedDeltaNet (SSM-like linear attention)
// ---------------------------------------------------------------------------

/// Cache state for a GatedDeltaNet layer.
#[derive(Debug, Clone)]
pub struct ArraysCache {
    pub conv_state: Option<Array>,
    pub ssm_state: Option<Array>,
    pub offset: i32,
}

impl ArraysCache {
    pub fn new() -> Self {
        Self {
            conv_state: None,
            ssm_state: None,
            offset: 0,
        }
    }
}

impl Default for ArraysCache {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[derive(Debug, Clone, ModuleParameters)]
struct GatedDeltaNet {
    #[param]
    in_proj_qkvz: QLinear,
    #[param]
    in_proj_ba: QLinear,
    #[param]
    conv1d: nn::Conv1d,
    #[param]
    norm: nn::RmsNorm,
    #[param]
    out_proj: QLinear,
    #[param]
    A_log: Param<Array>,
    #[param]
    dt_bias: Param<Array>,
    num_k_heads: i32,
    num_v_heads: i32,
    head_k_dim: i32,
    head_v_dim: i32,
    key_dim: i32,
    conv_dim: i32,
    conv_kernel_size: i32,
}

impl GatedDeltaNet {
    fn new(args: &Qwen3NextModelArgs, ql: i32, qb: i32) -> Result<Self, Exception> {
        let num_k_heads = args.linear_num_key_heads;
        let num_v_heads = args.linear_num_value_heads;
        let head_k_dim = args.linear_key_head_dim;
        let head_v_dim = args.linear_value_head_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_dim = key_dim * 2 + value_dim;
        let conv_kernel_size = args.linear_conv_kernel_dim;

        Ok(Self {
            in_proj_qkvz: QLinear::new(ql, qb)?,
            in_proj_ba: QLinear::new(ql, qb)?,
            conv1d: nn::Conv1dBuilder::new(conv_dim, conv_dim, conv_kernel_size)
                .bias(false)
                .groups(conv_dim)
                .padding(0)
                .build()?,
            norm: nn::RmsNormBuilder::new(head_v_dim)
                .eps(args.rms_norm_eps)
                .build()?,
            out_proj: QLinear::new(ql, qb)?,
            A_log: Param::new(Array::zeros::<f32>(&[num_v_heads])?),
            dt_bias: Param::new(Array::zeros::<f32>(&[num_v_heads])?),
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            conv_dim,
            conv_kernel_size,
        })
    }

    #[allow(non_snake_case)]
    fn forward(
        &mut self,
        inputs: &Array,
        _mask: Option<&Array>,
        cache: &mut ArraysCache,
    ) -> Result<Array, Exception> {
        let shape = inputs.shape();
        let B = *shape
            .first()
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;
        let S = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Input must have >= 2 dims"))?;

        // Project inputs
        let mixed_qkvz = self.in_proj_qkvz.forward(inputs)?;
        let mixed_ba = self.in_proj_ba.forward(inputs)?;

        // Split into q, k, v, z, b, a
        let (q, k, v, z, b, a) = self.fix_query_key_value_ordering(&mixed_qkvz, &mixed_ba, B, S)?;

        // Conv1d with state management
        let conv_state = cache.conv_state.take().unwrap_or_else(|| {
            Array::zeros::<f32>(&[B, self.conv_kernel_size - 1, self.conv_dim])
                .unwrap_or_else(|_| Array::from_slice(&[0.0_f32], &[1]))
        });

        // Concatenate q, k, v for conv input
        let q_flat = q.reshape(&[B, S, -1])?;
        let k_flat = k.reshape(&[B, S, -1])?;
        let v_flat = v.reshape(&[B, S, -1])?;
        let mixed_qkv = ops::concatenate_axis(&[&q_flat, &k_flat, &v_flat], -1)?;

        // Prepend conv state
        let conv_input = ops::concatenate_axis(&[&conv_state, &mixed_qkv], 1)?;

        // Update conv state cache (keep last kernel-1 timesteps)
        let n_keep = self.conv_kernel_size - 1;
        let conv_input_len = *conv_input
            .shape()
            .get(1)
            .ok_or_else(|| Exception::custom("conv_input missing seq dim"))?;
        let keep_start = conv_input_len - n_keep;
        cache.conv_state = Some(conv_input.index((.., keep_start.., ..)));

        // Apply conv1d + silu
        let conv_out = nn::silu(self.conv1d.forward(&conv_input)?)?;

        // Split conv output back to q, k, v
        let split_indices = &[self.key_dim, self.key_dim * 2];
        let conv_parts = conv_out.split_axis(split_indices, Some(-1))?;
        let conv_q = conv_parts
            .first()
            .ok_or_else(|| Exception::custom("conv split failed"))?
            .reshape(&[B, S, self.num_k_heads, self.head_k_dim])?;
        let conv_k = conv_parts
            .get(1)
            .ok_or_else(|| Exception::custom("conv split failed"))?
            .reshape(&[B, S, self.num_k_heads, self.head_k_dim])?;
        let conv_v = conv_parts
            .get(2)
            .ok_or_else(|| Exception::custom("conv split failed"))?
            .reshape(&[B, S, self.num_v_heads, self.head_v_dim])?;

        // RMS normalize q and k (no learnable weight)
        let inv_scale_f32 = f32::from(
            i16::try_from(self.head_k_dim).map_err(|_| Exception::custom("head_k_dim overflow"))?,
        )
        .sqrt()
        .recip();
        let inv_scale_sq = inv_scale_f32 * inv_scale_f32;
        let norm_q = fast::rms_norm(&conv_q, &Array::ones::<f32>(&[self.head_k_dim])?, 1e-6)?
            .multiply(Array::from_f32(inv_scale_sq))?;
        let norm_k = fast::rms_norm(&conv_k, &Array::ones::<f32>(&[self.head_k_dim])?, 1e-6)?
            .multiply(Array::from_f32(inv_scale_f32))?;

        // Compute gating: g = exp(-exp(A_log) * softplus(a + dt_bias))
        let a_plus_bias = a.add(&*self.dt_bias)?;
        let sp = nn::softplus(&a_plus_bias)?;
        let neg_decay = self.A_log.exp()?.negative()?.multiply(sp)?;
        let g = neg_decay.exp()?;

        // beta = sigmoid(b)
        let beta = nn::sigmoid(&b)?;

        // Get or initialize SSM state: [B, Hv, Dv, Dk]
        let mut state = cache.ssm_state.take().unwrap_or_else(|| {
            Array::zeros::<f32>(&[B, self.num_v_heads, self.head_v_dim, self.head_k_dim])
                .unwrap_or_else(|_| Array::from_slice(&[0.0_f32], &[1]))
        });

        // Repeat q, k for value head groups if needed
        let repeat_factor = self.num_v_heads / self.num_k_heads;
        let final_q = if repeat_factor > 1 {
            ops::repeat_axis::<f32>(norm_q, repeat_factor, -2)?
        } else {
            norm_q
        };
        let final_k = if repeat_factor > 1 {
            ops::repeat_axis::<f32>(norm_k, repeat_factor, -2)?
        } else {
            norm_k
        };

        // Gated delta update: sequential loop over timesteps
        let mut ys =
            Vec::with_capacity(usize::try_from(S).map_err(|_| Exception::custom("S overflow"))?);
        for t in 0..S {
            let qt = final_q.index((.., t, .., ..));
            let kt = final_k.index((.., t, .., ..));
            let vt = conv_v.index((.., t, .., ..));
            let gt = g.index((.., t, ..));
            let bt = beta.index((.., t, ..));

            let (y_t, new_state) = gated_delta_step(&qt, &kt, &vt, &gt, &bt, &state)?;
            state = new_state;
            ys.push(y_t);
        }

        cache.ssm_state = Some(state);
        cache.offset += S;

        let y_refs: Vec<&Array> = ys.iter().collect();
        let y = ops::stack_axis(&y_refs, 1)?;

        // Gated RMSNorm: norm(out) * silu(z) (swiglu with z as gate)
        let normed = self.norm.forward(&y)?;
        let gated_out = swiglu(&z, &normed)?;

        // Output projection
        let out_flat = gated_out.reshape(&[B, S, -1])?;
        self.out_proj.forward(&out_flat)
    }

    /// Reorder the projected qkvz and ba tensors into separate heads.
    #[allow(non_snake_case, clippy::type_complexity)]
    fn fix_query_key_value_ordering(
        &self,
        mixed_qkvz: &Array,
        mixed_ba: &Array,
        B: i32,
        S: i32,
    ) -> Result<(Array, Array, Array, Array, Array, Array), Exception> {
        let nk = self.num_k_heads;
        let dn = self.head_k_dim;
        let nv = self.num_v_heads;
        let dv = self.head_v_dim;
        let v_per_k = nv / nk;

        // Reshape to [B, S, nk, -1]
        let qkvz = mixed_qkvz.reshape(&[B, S, nk, -1])?;
        let ba = mixed_ba.reshape(&[B, S, nk, -1])?;

        // Split qkvz at [dn, 2*dn, 2*dn + v_per_k*dv]
        let split_at = &[dn, 2 * dn, 2 * dn + v_per_k * dv];
        let qkvz_parts = qkvz.split_axis(split_at, Some(-1))?;
        let q = qkvz_parts
            .first()
            .ok_or_else(|| Exception::custom("qkvz split failed"))?
            .clone();
        let k = qkvz_parts
            .get(1)
            .ok_or_else(|| Exception::custom("qkvz split failed"))?
            .clone();
        let v_raw = qkvz_parts
            .get(2)
            .ok_or_else(|| Exception::custom("qkvz split failed"))?;
        let z_raw = qkvz_parts
            .get(3)
            .ok_or_else(|| Exception::custom("qkvz split failed"))?;

        let v = v_raw.reshape(&[B, S, nv, dv])?;
        let z = z_raw.reshape(&[B, S, nv, dv])?;

        // Split ba at [v_per_k]
        let ba_parts = ba.split_axis(&[v_per_k], Some(-1))?;
        let b_raw = ba_parts
            .first()
            .ok_or_else(|| Exception::custom("ba split failed"))?;
        let a_raw = ba_parts
            .get(1)
            .ok_or_else(|| Exception::custom("ba split failed"))?;

        let b = b_raw.reshape(&[B, S, nv])?;
        let a = a_raw.reshape(&[B, S, nv])?;

        Ok((q, k, v, z, b, a))
    }
}

/// Single gated delta recurrence step.
///
/// q, k: [B, Hv, Dk]  (already repeated if Hv > Hk)
/// v: [B, Hv, Dv]
/// g: [B, Hv]  (scalar gating)
/// beta: [B, Hv]
/// state: [B, Hv, Dv, Dk]
fn gated_delta_step(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
) -> Result<(Array, Array), Exception> {
    // Decay: g is [B, Hv], expand to [B, Hv, 1, 1]
    let decay = g.expand_dims(-1)?.expand_dims(-1)?;
    let decayed_state = state.multiply(&decay)?;

    // kv_mem = sum(state * k[..., None, :], axis=-1)  -> [B, Hv, Dv]
    let k_expanded = k.expand_dims(-2)?; // [B, Hv, 1, Dk]
    let kv_mem = decayed_state
        .multiply(&k_expanded)?
        .sum_axes(&[-1], false)?;

    // delta = (v - kv_mem) * beta[..., None]
    let beta_expanded = beta.expand_dims(-1)?; // [B, Hv, 1]
    let delta = v.subtract(&kv_mem)?.multiply(&beta_expanded)?;

    // state = decayed_state + k[..., None, :] * delta[..., None]
    let delta_expanded = delta.expand_dims(-1)?; // [B, Hv, Dv, 1]
    let new_state = decayed_state.add(k_expanded.multiply(&delta_expanded)?)?;

    // y = sum(state * q[..., None, :], axis=-1)  -> [B, Hv, Dv]
    let q_expanded = q.expand_dims(-2)?; // [B, Hv, 1, Dk]
    let y = new_state.multiply(&q_expanded)?.sum_axes(&[-1], false)?;

    Ok((y, new_state))
}

// ---------------------------------------------------------------------------
// DecoderLayer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct DecoderLayer {
    #[param]
    linear_attn: Option<GatedDeltaNet>,
    #[param]
    self_attn: Option<Qwen3NextAttention>,
    #[param]
    input_layernorm: nn::RmsNorm,
    #[param]
    post_attention_layernorm: nn::RmsNorm,
    #[param]
    mlp: SparseMoeBlock,
    is_linear: bool,
}

impl DecoderLayer {
    fn new(args: &Qwen3NextModelArgs, layer_idx: i32, ql: i32, qb: i32) -> Result<Self, Exception> {
        let is_linear = (layer_idx + 1) % args.full_attention_interval != 0;

        let linear_attn = if is_linear {
            Some(GatedDeltaNet::new(args, ql, qb)?)
        } else {
            None
        };
        let self_attn = if !is_linear {
            Some(Qwen3NextAttention::new(args, ql, qb)?)
        } else {
            None
        };

        Ok(Self {
            linear_attn,
            self_attn,
            input_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            mlp: SparseMoeBlock::new(args, ql, qb)?,
            is_linear,
        })
    }

    fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut LayerCache,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(x)?;
        let r = if self.is_linear {
            let attn = self
                .linear_attn
                .as_mut()
                .ok_or_else(|| Exception::custom("linear_attn missing on linear layer"))?;
            let ssm_cache = match cache {
                LayerCache::Arrays(c) => c,
                _ => return Err(Exception::custom("Expected ArraysCache for linear layer")),
            };
            attn.forward(&normed, mask, ssm_cache)?
        } else {
            let attn = self
                .self_attn
                .as_mut()
                .ok_or_else(|| Exception::custom("self_attn missing on attention layer"))?;
            let kv_cache = match cache {
                LayerCache::KV(c) => c,
                _ => return Err(Exception::custom("Expected KVCache for attention layer")),
            };
            attn.forward(&normed, mask, kv_cache)?
        };

        let h = x.add(r)?;
        let normed_post = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed_post)?;
        h.add(mlp_out)
    }
}

// ---------------------------------------------------------------------------
// LayerCache enum
// ---------------------------------------------------------------------------

/// Per-layer cache: either KV cache (full attention) or arrays (SSM).
#[derive(Debug, Clone)]
pub enum LayerCache {
    KV(ConcatKeyValueCache),
    Arrays(ArraysCache),
}

// ---------------------------------------------------------------------------
// Qwen3NextInner (embed + layers + norm)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
struct Qwen3NextInner {
    #[param]
    embed_tokens: QEmbedding,
    #[param]
    layers: Vec<DecoderLayer>,
    #[param]
    norm: nn::RmsNorm,
    full_attention_interval: i32,
}

impl Qwen3NextInner {
    fn new(args: &Qwen3NextModelArgs, ql: i32, qb: i32) -> Result<Self, Exception> {
        let layers = (0..args.num_hidden_layers)
            .map(|i| DecoderLayer::new(args, i, ql, qb))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            embed_tokens: QEmbedding::new(ql, qb)?,
            layers,
            norm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            full_attention_interval: args.full_attention_interval,
        })
    }
}

// ---------------------------------------------------------------------------
// Qwen3NextCausalLM (the public model type)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct Qwen3NextCausalLM {
    pub args: Qwen3NextModelArgs,
    #[param]
    model: Qwen3NextInner,
    #[param]
    lm_head: Option<QLinear>,
}

impl Qwen3NextCausalLM {
    pub fn new(args: Qwen3NextModelArgs) -> Result<Self, Exception> {
        let ql = args.quantization.as_ref().map_or(64, |q| q.group_size);
        let qb = args.quantization.as_ref().map_or(4, |q| q.bits);

        let model = Qwen3NextInner::new(&args, ql, qb)?;
        let lm_head = if !args.tie_word_embeddings {
            Some(QLinear::new(ql, qb)?)
        } else {
            None
        };

        Ok(Self {
            args,
            model,
            lm_head,
        })
    }

    /// Create the per-layer cache vector.
    pub fn make_cache(&self) -> Vec<Option<LayerCache>> {
        self.model
            .layers
            .iter()
            .map(|layer| {
                if layer.is_linear {
                    Some(LayerCache::Arrays(ArraysCache::new()))
                } else {
                    Some(LayerCache::KV(ConcatKeyValueCache::new()))
                }
            })
            .collect()
    }

    /// Forward pass producing logits.
    #[allow(non_snake_case)]
    pub fn forward(
        &mut self,
        inputs: &Array,
        _mask: Option<&Array>,
        kv_cache: &mut Vec<Option<LayerCache>>,
    ) -> Result<Array, Exception> {
        let mut h = self.model.embed_tokens.forward(inputs)?;

        if kv_cache.is_empty() {
            *kv_cache = self.make_cache();
        }

        if kv_cache.len() != self.model.layers.len() {
            return Err(Exception::custom(format!(
                "cache length ({}) must match num layers ({})",
                kv_cache.len(),
                self.model.layers.len()
            )));
        }

        // Create attention mask for full-attention layers
        let shape = h.shape();
        let T = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Hidden state must have >= 2 dims"))?;

        let fa_mask = if T > 1 {
            // Find the first full-attention layer's cache to get offset
            let fa_idx = self.model.full_attention_interval - 1;
            let fa_idx_usize =
                usize::try_from(fa_idx).map_err(|_| Exception::custom("fa_idx overflow"))?;
            let offset = kv_cache
                .get(fa_idx_usize)
                .and_then(|c| c.as_ref())
                .map(|c| match c {
                    LayerCache::KV(kv) => kv.offset(),
                    LayerCache::Arrays(a) => a.offset,
                })
                .unwrap_or(0);
            Some(crate::utils::create_causal_mask(T, Some(offset))?)
        } else {
            None
        };

        for (layer, layer_cache) in self.model.layers.iter_mut().zip(kv_cache.iter_mut()) {
            let cache = layer_cache
                .as_mut()
                .ok_or_else(|| Exception::custom("Layer cache is None"))?;
            let mask = if layer.is_linear {
                None
            } else {
                fa_mask.as_ref()
            };
            h = layer.forward(&h, mask, cache)?;
        }

        h = self.model.norm.forward(&h)?;

        // LM head
        match self.lm_head.as_ref() {
            Some(head) => head.forward(&h),
            None => self.model.embed_tokens.as_linear(&h),
        }
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Load model args from config.json.
pub fn load_model_args(model_dir: impl AsRef<Path>) -> Result<Qwen3NextModelArgs, ModelError> {
    let config_path = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(config_path)?;
    Ok(serde_json::from_reader(file)?)
}

/// Load a Qwen3Next model from a directory containing safetensors + config.json.
pub fn load_qwen3_next_model(model_dir: impl AsRef<Path>) -> Result<Qwen3NextCausalLM, ModelError> {
    let model_path = model_dir.as_ref();
    let args = load_model_args(model_path)?;

    tracing::info!(
        model_type = %args.model_type,
        hidden_size = args.hidden_size,
        num_layers = args.num_hidden_layers,
        num_heads = args.num_attention_heads,
        num_kv_heads = args.num_key_value_heads,
        num_experts = args.num_experts,
        vocab_size = args.vocab_size,
        "Loading qwen3_next model"
    );

    let mut model = Qwen3NextCausalLM::new(args)?;

    // Load weights directly from safetensors (no key remapping needed
    // since our param names match the safetensors keys exactly)
    crate::load_safetensors_weights(&mut model, model_path)?;

    tracing::info!("Qwen3Next model loaded successfully");
    Ok(model)
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_config_deserialization() {
        let json = r#"{
            "model_type": "qwen3_next",
            "hidden_size": 2048,
            "num_hidden_layers": 48,
            "intermediate_size": 5120,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "rms_norm_eps": 1e-06,
            "vocab_size": 151936,
            "rope_theta": 5000000,
            "partial_rotary_factor": 0.25,
            "max_position_embeddings": 262144,
            "linear_num_value_heads": 32,
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": 512,
            "num_experts_per_tok": 10,
            "decoder_sparse_step": 1,
            "shared_expert_intermediate_size": 512,
            "moe_intermediate_size": 512,
            "norm_topk_prob": true,
            "full_attention_interval": 4,
            "tie_word_embeddings": false,
            "quantization": { "group_size": 64, "bits": 4 }
        }"#;

        let args: Qwen3NextModelArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.model_type, "qwen3_next");
        assert_eq!(args.hidden_size, 2048);
        assert_eq!(args.num_hidden_layers, 48);
        assert_eq!(args.head_dim, 256);
        assert_eq!(args.num_experts, 512);
        assert_eq!(args.num_experts_per_tok, 10);
        assert_eq!(args.full_attention_interval, 4);
        assert_eq!(args.linear_conv_kernel_dim, 4);
        assert!(!args.tie_word_embeddings);
        assert!(args.norm_topk_prob);
        let qc = args.quantization.unwrap();
        assert_eq!(qc.group_size, 64);
        assert_eq!(qc.bits, 4);
    }

    #[test]
    fn test_swiglu() {
        let gate = Array::from_slice(&[1.0_f32, -1.0, 0.5], &[1, 3]);
        let x = Array::from_slice(&[2.0_f32, 3.0, 4.0], &[1, 3]);
        let result = swiglu(&gate, &x).unwrap();
        assert_eq!(result.shape(), &[1, 3]);
        // silu(1.0) * 2.0 = 0.7311 * 2.0 ~= 1.462
        let first: f32 = result.index((.., 0..1)).item();
        assert!(first > 1.0);
    }

    #[test]
    fn test_gated_delta_step_shapes() {
        // B=1, Hv=2, Dk=4, Dv=3
        let q = Array::ones::<f32>(&[1, 2, 4]).unwrap();
        let k = Array::ones::<f32>(&[1, 2, 4]).unwrap();
        let v = Array::ones::<f32>(&[1, 2, 3]).unwrap();
        let g = Array::ones::<f32>(&[1, 2]).unwrap();
        let beta = ops::broadcast_to(&Array::from_f32(0.5), &[1, 2]).unwrap();
        let state = Array::zeros::<f32>(&[1, 2, 3, 4]).unwrap();

        let (y, new_state) = gated_delta_step(&q, &k, &v, &g, &beta, &state).unwrap();
        assert_eq!(y.shape(), &[1, 2, 3]);
        assert_eq!(new_state.shape(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_layer_cache_variants() {
        let kv = LayerCache::KV(ConcatKeyValueCache::new());
        let arrays = LayerCache::Arrays(ArraysCache::new());
        match &kv {
            LayerCache::KV(c) => assert_eq!(c.offset(), 0),
            _ => panic!("Expected KV variant"),
        }
        match &arrays {
            LayerCache::Arrays(c) => assert_eq!(c.offset, 0),
            _ => panic!("Expected Arrays variant"),
        }
    }
}
