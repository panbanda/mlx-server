pub mod cache;
pub mod error;
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

pub use transformer::{Model, ModelArgs};

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
    let index_path = model_path.join("model.safetensors.index.json");
    if index_path.exists() {
        let json = std::fs::read_to_string(&index_path)?;
        let index: WeightMapIndex = serde_json::from_str(&json)?;
        let weight_files: HashSet<&String> = index.weight_map.values().collect();

        for weight_file in weight_files {
            let weights_path = model_path.join(weight_file);
            tracing::debug!(file = %weight_file, "Loading weights");
            model.load_safetensors(&weights_path)?;
        }
    } else {
        let single_path = model_path.join("model.safetensors");
        if single_path.exists() {
            model.load_safetensors(&single_path)?;
        } else {
            return Err(ModelError::MissingWeight(
                "No safetensors files found".to_owned(),
            ));
        }
    }

    Ok(())
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
}
