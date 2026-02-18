pub mod cache;
pub mod error;
pub mod qwen3_next;
pub mod registry;
pub mod transformer;
pub mod utils;

use std::collections::{HashMap, HashSet};
use std::path::Path;

use mlx_rs::module::ModuleParametersExt;
use mlx_rs::{Array, argmax_axis, array, categorical, error::Exception};
use serde::Deserialize;
use serde_json::Value;

use crate::error::ModelError;

pub use qwen3_next::{LayerCache, Qwen3NextCausalLM};
pub use transformer::{Model, ModelArgs};

// ---------------------------------------------------------------------------
// AnyModel / AnyCache -- unified dispatch across model architectures
// ---------------------------------------------------------------------------

/// Cache type that works for all supported model architectures.
#[derive(Debug, Clone)]
pub enum AnyCache {
    /// Standard KV cache for transformer models (Qwen2/Llama/Mistral).
    KV(Vec<Option<cache::ConcatKeyValueCache>>),
    /// Hybrid KV+SSM cache for qwen3_next.
    Hybrid(Vec<Option<LayerCache>>),
}

/// Unified model wrapper dispatching to the correct architecture.
pub enum AnyModel {
    Transformer(Model),
    Qwen3Next(Qwen3NextCausalLM),
}

impl AnyModel {
    pub fn forward(
        &mut self,
        inputs: &Array,
        mask: Option<&Array>,
        cache: &mut AnyCache,
    ) -> Result<Array, Exception> {
        match (self, cache) {
            (AnyModel::Transformer(m), AnyCache::KV(c)) => m.forward(inputs, mask, c),
            (AnyModel::Qwen3Next(m), AnyCache::Hybrid(c)) => m.forward(inputs, mask, c),
            _ => Err(Exception::custom("Model/cache type mismatch")),
        }
    }

    pub fn make_cache(&self) -> AnyCache {
        match self {
            AnyModel::Transformer(_) => AnyCache::KV(Vec::new()),
            AnyModel::Qwen3Next(m) => AnyCache::Hybrid(m.make_cache()),
        }
    }
}

/// Sample a token from logits with temperature and top-p (nucleus) sampling.
///
/// Shared across all model architectures.
pub fn sample(logits: &Array, temp: f32, top_p: f32) -> Result<Array, Exception> {
    if temp == 0.0 {
        argmax_axis!(logits, -1)
    } else {
        let scaled = logits.multiply(array!(1.0 / temp))?;
        if top_p < 1.0 {
            sample_top_p(&scaled, top_p)
        } else {
            categorical!(scaled)
        }
    }
}

/// Nucleus (top-p) sampling: sample from the smallest set of tokens whose
/// cumulative probability exceeds `top_p`.
fn sample_top_p(logits: &Array, top_p: f32) -> Result<Array, Exception> {
    use mlx_rs::ops::{argsort_axis, concatenate_axis, maximum, softmax_axis};

    let probs = softmax_axis(logits, -1, None)?;

    // Sort descending: negate then ascending argsort
    let neg_probs = probs.negative()?;
    let sorted_indices = argsort_axis(&neg_probs, -1)?;
    let sorted_probs = probs.take_along_axis(&sorted_indices, -1)?;

    let cumsum = sorted_probs.cumsum(-1, None, None)?;

    // Mask tokens where cumulative sum exceeds top_p
    let cumsum_mask = cumsum.le(array!(top_p))?;

    // Always keep at least the top token
    let n_vocab = *probs
        .shape()
        .last()
        .ok_or_else(|| Exception::custom("logits must have at least 1 dimension"))?;
    let ones = Array::ones::<f32>(&[1])?;
    let zeros_count = n_vocab - 1;
    let zeros = Array::zeros::<f32>(&[zeros_count])?;
    let first_token_mask = if probs.ndim() > 1 {
        let full = concatenate_axis(&[&ones, &zeros], 0)?;
        full.reshape(&[1, -1])?
    } else {
        concatenate_axis(&[&ones, &zeros], 0)?
    };
    let final_mask = maximum(&cumsum_mask, &first_token_mask)?;

    let filtered_probs = sorted_probs.multiply(final_mask)?;
    // Re-normalize
    let sum = filtered_probs.sum_axes(&[-1], true)?;
    let normalized = filtered_probs.divide(sum)?;

    let sampled = categorical!(normalized)?;
    // Map back to original indices
    sorted_indices
        .take_along_axis(&sampled.reshape(&[-1, 1])?, -1)?
        .squeeze_axes(&[-1])
}

/// Weight map index from model.safetensors.index.json.
#[derive(Debug, Clone, Deserialize)]
pub struct WeightMapIndex {
    pub metadata: HashMap<String, Value>,
    pub weight_map: HashMap<String, String>,
}

/// Load a tokenizer from a model directory.
pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<tokenizers::Tokenizer, ModelError> {
    let file = model_dir.as_ref().join("tokenizer.json");
    tokenizers::Tokenizer::from_file(file)
        .map_err(|e| ModelError::Io(std::io::Error::other(e.to_string())))
}

/// Load safetensors weights into any model that implements `ModuleParametersExt`.
pub fn load_safetensors_weights<M: ModuleParametersExt>(
    model: &mut M,
    model_path: &Path,
) -> Result<(), ModelError> {
    load_quantized_safetensors_weights(model, model_path, false)
}

/// Load safetensors weights with optional name remapping for quantized models.
///
/// Pre-quantized MLX models store weights with flat names (e.g., `q_proj.weight`)
/// but the Rust quantized types use a nested structure (`q_proj.inner.weight`).
/// When `quantized` is true, keys ending in `.weight` or `.bias` (but not
/// `.biases`) that don't match directly are retried with `.inner.` inserted.
pub fn load_quantized_safetensors_weights<M: ModuleParametersExt>(
    model: &mut M,
    model_path: &Path,
    quantized: bool,
) -> Result<(), ModelError> {
    let safetensors_files = collect_safetensors_files(model_path)?;

    let mut params = model.parameters_mut().flatten();

    for file_path in &safetensors_files {
        tracing::debug!(file = %file_path.display(), "Loading weights");
        let loaded = Array::load_safetensors(file_path)
            .map_err(|e| ModelError::Io(std::io::Error::other(e.to_string())))?;

        for (key, value) in loaded {
            if let Some(param) = params.get_mut(&*key) {
                **param = value;
            } else if quantized {
                // Remap: "a.b.weight" -> "a.b.inner.weight", "a.b.bias" -> "a.b.inner.bias"
                if let Some(remapped) = remap_quantized_key(&key) {
                    if let Some(param) = params.get_mut(&*remapped) {
                        **param = value;
                    }
                }
            }
        }
    }

    model
        .eval()
        .map_err(|e| ModelError::Io(std::io::Error::other(e.to_string())))?;

    Ok(())
}

/// Collect safetensors file paths from a model directory.
fn collect_safetensors_files(model_path: &Path) -> Result<Vec<std::path::PathBuf>, ModelError> {
    let index_path = model_path.join("model.safetensors.index.json");
    if index_path.exists() {
        let json = std::fs::read_to_string(&index_path)?;
        let index: WeightMapIndex = serde_json::from_str(&json)?;
        let weight_files: HashSet<&String> = index.weight_map.values().collect();
        Ok(weight_files
            .into_iter()
            .map(|f| model_path.join(f))
            .collect())
    } else {
        let single_path = model_path.join("model.safetensors");
        if single_path.exists() {
            Ok(vec![single_path])
        } else {
            Err(ModelError::MissingWeight(
                "No safetensors files found".to_owned(),
            ))
        }
    }
}

/// Remap a safetensors key for quantized model parameter names.
///
/// MLX Python saves quantized weights as `layer.weight` but Rust mlx-rs
/// QuantizedLinear/QuantizedEmbedding nest them as `layer.inner.weight`.
fn remap_quantized_key(key: &str) -> Option<String> {
    if let Some(prefix) = key.strip_suffix(".weight") {
        Some(format!("{prefix}.inner.weight"))
    } else if key.ends_with(".bias") && !key.ends_with(".biases") {
        let prefix = key.strip_suffix(".bias")?;
        Some(format!("{prefix}.inner.bias"))
    } else {
        None
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_greedy() {
        let logits = Array::from_slice(&[0.1_f32, 0.9, 0.0], &[1, 3]);
        let token = sample(&logits, 0.0, 1.0).unwrap();
        let val: u32 = token.item();
        assert_eq!(val, 1);
    }

    #[test]
    fn sample_2d_input_returns_1d_shape() {
        // 2D [1, V] input (correct decode loop usage after slicing)
        let logits = Array::from_slice(&[0.1_f32, 0.9, 0.0], &[1, 3]);
        let token = sample(&logits, 0.0, 1.0).unwrap();
        assert_eq!(token.shape(), &[1], "sample on [1, V] must return [1]");
    }

    #[test]
    fn sample_3d_input_returns_2d_shape() {
        // 3D [1, 1, V] input (raw decode logits -- the bug)
        // argmax on axis -1 of [1, 1, V] produces [1, 1], not [1].
        // Callers must slice to 2D before calling sample to avoid
        // dimension accumulation in the decode loop.
        let logits = Array::from_slice(&[0.1_f32, 0.9, 0.0], &[1, 1, 3]);
        let token = sample(&logits, 0.0, 1.0).unwrap();
        assert_eq!(
            token.shape(),
            &[1, 1],
            "sample on [1, 1, V] returns [1, 1] -- callers must slice first"
        );
    }

    #[test]
    fn remap_quantized_key_weight() {
        assert_eq!(
            remap_quantized_key("model.layers.0.self_attn.q_proj.weight"),
            Some("model.layers.0.self_attn.q_proj.inner.weight".to_owned())
        );
    }

    #[test]
    fn remap_quantized_key_bias() {
        assert_eq!(
            remap_quantized_key("model.layers.0.self_attn.q_proj.bias"),
            Some("model.layers.0.self_attn.q_proj.inner.bias".to_owned())
        );
    }

    #[test]
    fn remap_quantized_key_biases_not_remapped() {
        // ".biases" is a quantization parameter, not a layer bias -- don't remap
        assert_eq!(
            remap_quantized_key("model.layers.0.self_attn.q_proj.biases"),
            None
        );
    }

    #[test]
    fn remap_quantized_key_scales_not_remapped() {
        assert_eq!(
            remap_quantized_key("model.layers.0.self_attn.q_proj.scales"),
            None
        );
    }

    #[test]
    fn remap_quantized_key_embed_tokens_weight() {
        assert_eq!(
            remap_quantized_key("model.embed_tokens.weight"),
            Some("model.embed_tokens.inner.weight".to_owned())
        );
    }

    #[test]
    fn remap_quantized_key_norm_weight() {
        // RmsNorm weight should still remap -- it will just not match any param
        // and be silently skipped by the loader
        assert_eq!(
            remap_quantized_key("model.norm.weight"),
            Some("model.norm.inner.weight".to_owned())
        );
    }
}
