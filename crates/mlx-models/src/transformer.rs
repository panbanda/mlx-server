//! Unified transformer model implementation.
//!
//! Supports Qwen2, Llama, and Mistral architectures. Architecture-specific
//! behavior (e.g., Q/K/V bias) is parameterized through `ModelArgs`.

use std::path::Path;

use mlx_rs::{
    Array,
    builder::Builder,
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::Module,
    nn,
    quantization::MaybeQuantized,
};
use serde::Deserialize;

use crate::{
    cache::KeyValueCache,
    error::ModelError,
    utils::{AttentionMask, create_attention_mask, scaled_dot_product_attention},
};

fn default_rope_theta() -> f32 {
    10000.0
}

/// Quantization parameters from config.json.
#[derive(Debug, Clone, Deserialize)]
pub struct QuantizationConfig {
    pub group_size: i32,
    pub bits: i32,
}

/// Unified model configuration, deserialized from config.json.
///
/// Architecture-specific fields use serde defaults so that configs from
/// Qwen2, Llama, and Mistral all deserialize into the same struct.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    pub num_key_value_heads: i32,
    pub max_position_embeddings: i32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,

    // Architecture-specific optional fields
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default)]
    pub sliding_window: Option<i32>,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,

    // Quantization (present in pre-quantized MLX models)
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

impl ModelArgs {
    /// Whether Q/K/V projections should have bias.
    ///
    /// Qwen2/Qwen3 use bias; Llama and Mistral do not.
    pub fn qkv_bias(&self) -> bool {
        matches!(self.model_type.as_str(), "qwen2" | "qwen3")
    }

    /// Head dimension, computed from hidden_size / num_attention_heads.
    ///
    /// Panics in debug builds if not evenly divisible.
    pub fn head_dim(&self) -> i32 {
        debug_assert!(
            self.num_attention_heads != 0 && self.hidden_size % self.num_attention_heads == 0,
            "hidden_size ({}) must be divisible by num_attention_heads ({})",
            self.hidden_size,
            self.num_attention_heads
        );
        self.hidden_size / self.num_attention_heads
    }

    /// Validated head dimension that returns an error if not evenly divisible.
    pub fn checked_head_dim(&self) -> Result<i32, ModelError> {
        if self.num_attention_heads == 0 {
            return Err(ModelError::ShapeMismatch(
                "num_attention_heads must be positive".to_owned(),
            ));
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(ModelError::ShapeMismatch(format!(
                "hidden_size ({}) must be divisible by num_attention_heads ({})",
                self.hidden_size, self.num_attention_heads
            )));
        }
        Ok(self.hidden_size / self.num_attention_heads)
    }
}

/// Multi-head attention module.
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Attention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub scale: f32,

    #[quantizable]
    #[param]
    pub q_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub k_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub v_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub o_proj: MaybeQuantized<nn::Linear>,
    #[param]
    pub rope: nn::Rope,
}

impl Attention {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        let dim = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;
        let head_dim = args
            .checked_head_dim()
            .map_err(|e| Exception::custom(e.to_string()))?;
        let head_dim_f32 = f32::from(
            i16::try_from(head_dim).map_err(|_| Exception::custom("head_dim out of i16 range"))?,
        );
        let scale = head_dim_f32.sqrt().recip();

        let qkv_bias = args.qkv_bias();
        let q_proj = nn::LinearBuilder::new(dim, n_heads * head_dim)
            .bias(qkv_bias)
            .build()?;
        let k_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(qkv_bias)
            .build()?;
        let v_proj = nn::LinearBuilder::new(dim, n_kv_heads * head_dim)
            .bias(qkv_bias)
            .build()?;
        let o_proj = nn::LinearBuilder::new(n_heads * head_dim, dim)
            .bias(false)
            .build()?;

        let rope = nn::RopeBuilder::new(head_dim)
            .traditional(false)
            .base(args.rope_theta)
            .scale(1.0)
            .build()
            .map_err(|e| Exception::custom(format!("Failed to build RoPE: {e}")))?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            scale,
            q_proj: MaybeQuantized::Original(q_proj),
            k_proj: MaybeQuantized::Original(k_proj),
            v_proj: MaybeQuantized::Original(v_proj),
            o_proj: MaybeQuantized::Original(o_proj),
            rope,
        })
    }
}

/// Input to the attention module.
pub struct AttentionInput<'a, C> {
    pub x: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: Option<&'a mut C>,
}

impl<C> Module<AttentionInput<'_, C>> for Attention
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(&mut self, input: AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, mut cache } = input;

        let shape = x.shape();
        let B = *shape
            .first()
            .ok_or_else(|| Exception::custom("Input must have at least 2 dimensions"))?;
        let L = *shape
            .get(1)
            .ok_or_else(|| Exception::custom("Input must have at least 2 dimensions"))?;

        let q_raw = self.q_proj.forward(x)?;
        let k_raw = self.k_proj.forward(x)?;
        let v_raw = self.v_proj.forward(x)?;

        let mut queries = q_raw
            .reshape(&[B, L, self.n_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut keys = k_raw
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let mut values = v_raw
            .reshape(&[B, L, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        if let Some(ref mut kv_cache) = cache {
            let q_input = nn::RopeInputBuilder::new(&queries)
                .offset(kv_cache.offset())
                .build()?;
            queries = self.rope.forward(q_input)?;
            let k_input = nn::RopeInputBuilder::new(&keys)
                .offset(kv_cache.offset())
                .build()?;
            keys = self.rope.forward(k_input)?;

            let (cached_keys, cached_values) = kv_cache.update_and_fetch(keys, values)?;
            keys = cached_keys;
            values = cached_values;
        } else {
            queries = self.rope.forward(nn::RopeInput::new(&queries))?;
            keys = self.rope.forward(nn::RopeInput::new(&keys))?;
        }

        let output = scaled_dot_product_attention(queries, keys, values, self.scale, mask)?
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[B, L, -1])?;

        self.o_proj.forward(&output)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
        <nn::Rope as Module<nn::RopeInput>>::training_mode(&mut self.rope, mode);
    }
}

/// SiLU-gated MLP.
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Mlp {
    #[quantizable]
    #[param]
    pub gate_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub down_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    pub up_proj: MaybeQuantized<nn::Linear>,
}

impl Mlp {
    pub fn new(dim: i32, hidden_dim: i32) -> Result<Self, Exception> {
        let gate_proj = nn::LinearBuilder::new(dim, hidden_dim)
            .bias(false)
            .build()?;
        let down_proj = nn::LinearBuilder::new(hidden_dim, dim)
            .bias(false)
            .build()?;
        let up_proj = nn::LinearBuilder::new(dim, hidden_dim)
            .bias(false)
            .build()?;

        Ok(Self {
            gate_proj: MaybeQuantized::Original(gate_proj),
            down_proj: MaybeQuantized::Original(down_proj),
            up_proj: MaybeQuantized::Original(up_proj),
        })
    }
}

impl Module<&Array> for Mlp {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array) -> Result<Self::Output, Self::Error> {
        let gated =
            nn::silu(self.gate_proj.forward(input)?)?.multiply(self.up_proj.forward(input)?)?;
        self.down_proj.forward(&gated)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate_proj.training_mode(mode);
        self.down_proj.training_mode(mode);
        self.up_proj.training_mode(mode);
    }
}

/// A single transformer block (attention + MLP with residual connections).
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct TransformerBlock {
    pub num_attention_heads: i32,
    pub hidden_size: i32,

    #[quantizable]
    #[param]
    pub self_attn: Attention,
    #[quantizable]
    #[param]
    pub mlp: Mlp,
    #[param]
    pub input_layernorm: nn::RmsNorm,
    #[param]
    pub post_attention_layernorm: nn::RmsNorm,
}

impl TransformerBlock {
    pub fn new(args: &ModelArgs) -> Result<Self, Exception> {
        Ok(Self {
            num_attention_heads: args.num_attention_heads,
            hidden_size: args.hidden_size,
            self_attn: Attention::new(args)?,
            mlp: Mlp::new(args.hidden_size, args.intermediate_size)?,
            input_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
            post_attention_layernorm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
        })
    }
}

impl<C> Module<AttentionInput<'_, C>> for TransformerBlock
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: AttentionInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, cache } = input;

        let normed = self.input_layernorm.forward(x)?;
        let residual = self.self_attn.forward(AttentionInput {
            x: &normed,
            mask,
            cache,
        })?;
        let h = x.add(residual)?;

        let normed_post = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed_post)?;
        h.add(mlp_out)
    }

    fn training_mode(&mut self, mode: bool) {
        <Attention as Module<AttentionInput<'_, C>>>::training_mode(&mut self.self_attn, mode);
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
    }
}

/// Transformer model (embedding + layers + norm, without LM head).
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
struct TransformerModel {
    pub vocab_size: i32,
    pub num_hidden_layers: i32,

    #[quantizable]
    #[param]
    pub embed_tokens: MaybeQuantized<nn::Embedding>,
    #[quantizable]
    #[param]
    pub layers: Vec<TransformerBlock>,
    #[param]
    pub norm: nn::RmsNorm,
}

impl TransformerModel {
    fn new(args: &ModelArgs) -> Result<Self, Exception> {
        if !args.vocab_size.is_positive() {
            return Err(Exception::custom("vocab_size must be positive"));
        }

        Ok(Self {
            vocab_size: args.vocab_size,
            num_hidden_layers: args.num_hidden_layers,
            embed_tokens: MaybeQuantized::Original(nn::Embedding::new(
                args.vocab_size,
                args.hidden_size,
            )?),
            layers: (0..args.num_hidden_layers)
                .map(|_| TransformerBlock::new(args))
                .collect::<Result<Vec<_>, _>>()?,
            norm: nn::RmsNormBuilder::new(args.hidden_size)
                .eps(args.rms_norm_eps)
                .build()?,
        })
    }
}

/// Input to the transformer model.
struct ModelInput<'a, C> {
    pub inputs: &'a Array,
    pub mask: Option<&'a Array>,
    pub cache: &'a mut Vec<Option<C>>,
}

impl<C> Module<ModelInput<'_, C>> for TransformerModel
where
    C: KeyValueCache,
{
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: ModelInput<'_, C>) -> Result<Self::Output, Self::Error> {
        let ModelInput {
            inputs,
            mask,
            cache,
        } = input;

        let mut h = self.embed_tokens.forward(inputs)?;

        let computed_mask = match mask {
            Some(m) => Some(m.clone()),
            None => match create_attention_mask(&h, cache, Some(true))? {
                Some(AttentionMask::Array(a)) => Some(a),
                Some(AttentionMask::Causal) => {
                    return Err(Exception::custom("Only Array mask is supported"));
                }
                None => None,
            },
        };

        if cache.is_empty() {
            *cache = (0..self.layers.len()).map(|_| None).collect();
        } else if cache.len() != self.layers.len() {
            return Err(Exception::custom(format!(
                "kv_cache length ({}) must match num layers ({})",
                cache.len(),
                self.layers.len()
            )));
        }

        for (layer, layer_cache) in self.layers.iter_mut().zip(cache.iter_mut()) {
            h = layer.forward(AttentionInput {
                x: &h,
                mask: computed_mask.as_ref(),
                cache: layer_cache.as_mut(),
            })?;
        }

        self.norm.forward(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.embed_tokens.training_mode(mode);
        for layer in &mut self.layers {
            <TransformerBlock as Module<AttentionInput<'_, C>>>::training_mode(layer, mode);
        }
        self.norm.training_mode(mode);
    }
}

/// Full causal language model with LM head.
#[derive(Debug, Clone, ModuleParameters, Quantizable)]
pub struct Model {
    pub args: ModelArgs,

    #[quantizable]
    #[param]
    model: TransformerModel,

    #[quantizable]
    #[param]
    lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    pub fn new(args: ModelArgs) -> Result<Self, Exception> {
        let model = TransformerModel::new(&args)?;
        let lm_head = if !args.tie_word_embeddings {
            Some(MaybeQuantized::Original(
                nn::LinearBuilder::new(args.hidden_size, args.vocab_size)
                    .bias(false)
                    .build()?,
            ))
        } else {
            None
        };

        Ok(Self {
            args,
            model,
            lm_head,
        })
    }

    pub fn model_type(&self) -> &str {
        &self.args.model_type
    }

    /// Run a forward pass producing logits.
    pub fn forward<C: KeyValueCache>(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        kv_cache: &mut Vec<Option<C>>,
    ) -> Result<Array, Exception> {
        let out = self.model.forward(ModelInput {
            inputs,
            mask,
            cache: kv_cache,
        })?;

        match self.lm_head.as_mut() {
            Some(head) => head.forward(&out),
            None => match &mut self.model.embed_tokens {
                MaybeQuantized::Original(embed) => embed.as_linear(&out),
                MaybeQuantized::Quantized(q_embed) => q_embed.as_linear(&out),
            },
        }
    }
}

// --- Loading ---

/// Load model args from config.json.
pub fn load_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, ModelError> {
    let config_path = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(config_path)?;
    Ok(serde_json::from_reader(file)?)
}

/// Load a model from a directory containing safetensors + config.json.
pub fn load_model(model_dir: impl AsRef<Path>) -> Result<Model, ModelError> {
    let model_path = model_dir.as_ref();

    let args = load_model_args(model_path)?;
    tracing::info!(
        model_type = %args.model_type,
        hidden_size = args.hidden_size,
        num_layers = args.num_hidden_layers,
        num_heads = args.num_attention_heads,
        num_kv_heads = args.num_key_value_heads,
        vocab_size = args.vocab_size,
        qkv_bias = args.qkv_bias(),
        "Loading model"
    );

    let quantization = args.quantization.clone();
    let raw_model = Model::new(args)?;

    // Pre-quantized models need the MaybeQuantized fields converted to
    // Quantized variants before loading weights, so that the parameter
    // names (inner.weight, scales, biases) match the safetensors keys.
    let mut model = if let Some(ref qc) = quantization {
        tracing::info!(
            group_size = qc.group_size,
            bits = qc.bits,
            "Applying quantization structure"
        );
        mlx_rs::nn::quantize(raw_model, qc.group_size, qc.bits).map_err(|e| {
            ModelError::ShapeMismatch(format!("Failed to quantize model structure: {e}"))
        })?
    } else {
        raw_model
    };

    crate::load_quantized_safetensors_weights(&mut model, model_path, quantization.is_some())?;

    tracing::info!("Model loaded successfully");
    Ok(model)
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen2_config_deserialization() {
        let json = r#"{
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "hidden_size": 1536,
            "num_hidden_layers": 28,
            "intermediate_size": 8960,
            "num_attention_heads": 12,
            "rms_norm_eps": 1e-06,
            "vocab_size": 151936,
            "num_key_value_heads": 2,
            "max_position_embeddings": 32768,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": true,
            "use_sliding_window": false,
            "sliding_window": 32768
        }"#;

        let args: ModelArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.model_type, "qwen2");
        assert_eq!(args.hidden_size, 1536);
        assert_eq!(args.num_hidden_layers, 28);
        assert_eq!(args.num_attention_heads, 12);
        assert_eq!(args.num_key_value_heads, 2);
        assert_eq!(args.head_dim(), 128);
        assert!(args.tie_word_embeddings);
        assert!(args.qkv_bias());
    }

    #[test]
    fn test_llama_config_deserialization() {
        let json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "rms_norm_eps": 1e-06,
            "vocab_size": 32000,
            "num_key_value_heads": 32,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "tie_word_embeddings": false
        }"#;

        let args: ModelArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.model_type, "llama");
        assert_eq!(args.hidden_size, 4096);
        assert_eq!(args.head_dim(), 128);
        assert!(!args.tie_word_embeddings);
        assert!(!args.qkv_bias());
    }

    #[test]
    fn test_llama_config_defaults() {
        let json = r#"{
            "model_type": "llama",
            "hidden_size": 2048,
            "num_hidden_layers": 22,
            "intermediate_size": 5632,
            "num_attention_heads": 32,
            "rms_norm_eps": 1e-05,
            "vocab_size": 32000,
            "num_key_value_heads": 4,
            "max_position_embeddings": 2048
        }"#;

        let args: ModelArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.rope_theta, 10000.0);
        assert!(!args.tie_word_embeddings);
    }

    #[test]
    fn test_mistral_config_deserialization() {
        let json = r#"{
            "architectures": ["MistralForCausalLM"],
            "model_type": "mistral",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "rms_norm_eps": 1e-05,
            "vocab_size": 32000,
            "num_key_value_heads": 8,
            "max_position_embeddings": 32768,
            "rope_theta": 10000.0,
            "sliding_window": 4096
        }"#;

        let args: ModelArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.model_type, "mistral");
        assert_eq!(args.hidden_size, 4096);
        assert_eq!(args.head_dim(), 128);
        assert_eq!(args.sliding_window, Some(4096));
        assert!(!args.qkv_bias());
    }

    #[test]
    fn test_mistral_config_no_sliding_window() {
        let json = r#"{
            "model_type": "mistral",
            "hidden_size": 2048,
            "num_hidden_layers": 22,
            "intermediate_size": 5632,
            "num_attention_heads": 32,
            "rms_norm_eps": 1e-05,
            "vocab_size": 32000,
            "num_key_value_heads": 4,
            "max_position_embeddings": 2048
        }"#;

        let args: ModelArgs = serde_json::from_str(json).unwrap();
        assert!(args.sliding_window.is_none());
        assert_eq!(args.rope_theta, 10000.0);
    }

    #[test]
    fn test_head_dim_computation() {
        let json = r#"{
            "model_type": "qwen2",
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "intermediate_size": 2048,
            "num_attention_heads": 12,
            "rms_norm_eps": 1e-06,
            "vocab_size": 32000,
            "num_key_value_heads": 4,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0
        }"#;

        let args: ModelArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.head_dim(), 64);
    }

    #[test]
    fn test_checked_head_dim_zero_heads() {
        let json = r#"{
            "model_type": "llama",
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "intermediate_size": 2048,
            "num_attention_heads": 0,
            "rms_norm_eps": 1e-06,
            "vocab_size": 32000,
            "num_key_value_heads": 4,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0
        }"#;

        let args: ModelArgs = serde_json::from_str(json).unwrap();
        assert!(args.checked_head_dim().is_err());
    }

    #[test]
    fn test_checked_head_dim_not_divisible() {
        let json = r#"{
            "model_type": "llama",
            "hidden_size": 100,
            "num_hidden_layers": 12,
            "intermediate_size": 2048,
            "num_attention_heads": 3,
            "rms_norm_eps": 1e-06,
            "vocab_size": 32000,
            "num_key_value_heads": 1,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0
        }"#;

        let args: ModelArgs = serde_json::from_str(json).unwrap();
        assert!(args.checked_head_dim().is_err());
    }

    #[test]
    fn test_qkv_bias_for_qwen3() {
        let json = r#"{
            "model_type": "qwen3",
            "hidden_size": 2048,
            "num_hidden_layers": 28,
            "intermediate_size": 8960,
            "num_attention_heads": 16,
            "rms_norm_eps": 1e-06,
            "vocab_size": 151936,
            "num_key_value_heads": 2,
            "max_position_embeddings": 32768
        }"#;
        let args: ModelArgs = serde_json::from_str(json).unwrap();
        assert!(args.qkv_bias());
    }

    #[test]
    fn test_load_model_args_missing_file_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let result = load_model_args(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_load_model_args_invalid_json_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), "not json").unwrap();
        let result = load_model_args(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_quantization_config_deserialization() {
        let json = r#"{
            "model_type": "qwen2",
            "hidden_size": 1536,
            "num_hidden_layers": 28,
            "intermediate_size": 8960,
            "num_attention_heads": 12,
            "rms_norm_eps": 1e-06,
            "vocab_size": 151936,
            "num_key_value_heads": 2,
            "max_position_embeddings": 32768,
            "quantization": {
                "group_size": 64,
                "bits": 4
            }
        }"#;
        let args: ModelArgs = serde_json::from_str(json).unwrap();
        let qc = args.quantization.unwrap();
        assert_eq!(qc.group_size, 64);
        assert_eq!(qc.bits, 4);
    }

    #[test]
    fn test_no_quantization_config() {
        let json = r#"{
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "rms_norm_eps": 1e-06,
            "vocab_size": 32000,
            "num_key_value_heads": 32,
            "max_position_embeddings": 4096
        }"#;
        let args: ModelArgs = serde_json::from_str(json).unwrap();
        assert!(args.quantization.is_none());
    }
}
