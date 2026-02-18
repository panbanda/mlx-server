use axum::{Json, extract::State};

use crate::{
    error::ServerError,
    state::SharedState,
    types::openai::{
        EmbeddingInput, EmbeddingObject, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage,
    },
};

/// POST /v1/embeddings
///
/// Generates embeddings by tokenizing input and using token IDs as a
/// simple bag-of-tokens embedding. This is a placeholder implementation
/// that returns consistent, deterministic vectors. A proper embedding model
/// (e.g., a sentence-transformers model) would replace this.
pub async fn embeddings(
    State(state): State<SharedState>,
    Json(req): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, ServerError> {
    tracing::warn!("Embeddings endpoint uses placeholder implementation");

    let engine = state
        .engine_for(&req.model)
        .ok_or_else(|| ServerError::ModelNotFound(req.model.clone()))?;

    let inputs = match &req.input {
        EmbeddingInput::Single(s) => vec![s.clone()],
        EmbeddingInput::Multiple(v) => v.clone(),
    };

    if inputs.is_empty() {
        return Err(ServerError::BadRequest(
            "input must not be empty".to_owned(),
        ));
    }

    let mut data = Vec::new();
    let mut total_tokens: u32 = 0;

    for (idx, text) in inputs.iter().enumerate() {
        let encoding = engine
            .tokenizer()
            .encode(text.as_str(), false)
            .map_err(|e| ServerError::BadRequest(format!("Tokenization error: {e}")))?;

        let token_ids = encoding.get_ids();
        let token_count: u32 = token_ids
            .len()
            .try_into()
            .map_err(|_| ServerError::BadRequest("Input too long".to_owned()))?;
        total_tokens = total_tokens.saturating_add(token_count);

        // Simple bag-of-tokens embedding: normalized token frequency vector
        // This is a placeholder -- real embedding would use a model forward pass
        let embedding = compute_token_embedding(token_ids);

        let index: u32 = idx
            .try_into()
            .map_err(|_| ServerError::BadRequest("Too many inputs".to_owned()))?;

        data.push(EmbeddingObject {
            object: "embedding",
            embedding,
            index,
        });
    }

    Ok(Json(EmbeddingResponse {
        object: "list",
        data,
        model: req.model,
        usage: EmbeddingUsage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    }))
}

/// Compute a simple fixed-dimension embedding from token IDs.
///
/// Uses a deterministic hash-based approach to project tokens into a
/// fixed-size vector. This gives consistent results for the same input
/// but is not semantically meaningful. A real implementation would use
/// a model's hidden states.
fn compute_token_embedding(token_ids: &[u32]) -> Vec<f32> {
    const DIM: usize = 384;
    let mut embedding = vec![0.0_f32; DIM];

    for &token_id in token_ids {
        // Distribute token influence across dimensions using simple hash
        let dim_idx = token_hash_to_index(token_id.wrapping_mul(2654435761), DIM);
        if let Some(val) = embedding.get_mut(dim_idx) {
            *val += 1.0;
        }

        // Secondary dimension for richer signal
        let dim_idx2 = token_hash_to_index(token_id.wrapping_mul(2246822519), DIM);
        if let Some(val) = embedding.get_mut(dim_idx2) {
            *val += 0.5;
        }
    }

    // L2 normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in &mut embedding {
            *val /= norm;
        }
    }

    embedding
}

/// Map a hash value to a dimension index without `as` conversion.
fn token_hash_to_index(hash: u32, dim: usize) -> usize {
    // Mask to u16 range (infallible conversion to usize)
    let masked = hash & 0xFFFF;
    usize::from(u16::try_from(masked).unwrap_or(0)) % dim
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_token_embedding_dimension() {
        let embedding = compute_token_embedding(&[1, 2, 3, 4]);
        assert_eq!(embedding.len(), 384);
    }

    #[test]
    fn test_compute_token_embedding_normalized() {
        let embedding = compute_token_embedding(&[100, 200, 300]);
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_token_embedding_deterministic() {
        let a = compute_token_embedding(&[1, 2, 3]);
        let b = compute_token_embedding(&[1, 2, 3]);
        assert_eq!(a, b);
    }

    #[test]
    fn test_compute_token_embedding_different_inputs() {
        let a = compute_token_embedding(&[1, 2, 3]);
        let b = compute_token_embedding(&[4, 5, 6]);
        assert_ne!(a, b);
    }

    #[test]
    fn test_compute_token_embedding_empty() {
        let embedding = compute_token_embedding(&[]);
        assert_eq!(embedding.len(), 384);
        assert!(embedding.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_compute_token_embedding_single_token() {
        let embedding = compute_token_embedding(&[42]);
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
        assert!(embedding.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_token_hash_to_index_stays_in_bounds() {
        assert!(token_hash_to_index(0, 384) < 384);
        assert!(token_hash_to_index(u32::MAX, 384) < 384);
        assert!(token_hash_to_index(12345, 384) < 384);
    }

    #[test]
    fn test_compute_token_embedding_very_large_token_ids() {
        let embedding = compute_token_embedding(&[u32::MAX, u32::MAX - 1, u32::MAX - 2]);
        assert_eq!(embedding.len(), 384);
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_token_embedding_duplicate_token_ids() {
        let embedding = compute_token_embedding(&[42, 42, 42, 42]);
        assert_eq!(embedding.len(), 384);
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
        // Duplicate tokens all hash to the same bins, so after L2 normalization
        // the direction is identical to a single token. Verify that the
        // pre-normalization magnitudes differ by checking a non-zero element.
        let single = compute_token_embedding(&[42]);
        // Both are normalized to unit length, so the vectors point in the same
        // direction. This is correct behavior for the bag-of-tokens approach.
        assert_eq!(embedding, single);
    }

    #[test]
    fn test_compute_token_embedding_many_tokens() {
        let token_ids: Vec<u32> = (0..1000).collect();
        let embedding = compute_token_embedding(&token_ids);
        assert_eq!(embedding.len(), 384);
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_token_hash_to_index_dim_one() {
        assert_eq!(token_hash_to_index(0, 1), 0);
        assert_eq!(token_hash_to_index(u32::MAX, 1), 0);
        assert_eq!(token_hash_to_index(12345, 1), 0);
    }

    #[test]
    fn test_token_hash_to_index_hash_zero() {
        let idx = token_hash_to_index(0, 384);
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_token_hash_to_index_hash_u32_max() {
        let idx = token_hash_to_index(u32::MAX, 384);
        assert!(idx < 384);
    }

    #[test]
    fn test_token_hash_to_index_various_dims() {
        for dim in [1, 2, 10, 128, 384, 768, 1024] {
            for hash in [0_u32, 1, 100, 65535, u32::MAX] {
                let idx = token_hash_to_index(hash, dim);
                assert!(
                    idx < dim,
                    "index {idx} out of bounds for dim {dim} with hash {hash}"
                );
            }
        }
    }
}
