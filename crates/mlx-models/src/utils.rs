use mlx_rs::{
    Array, arange,
    error::Exception,
    fast::ScaledDotProductAttentionMask,
    ops::indexing::{IndexOp, NewAxis},
};

use crate::cache::KeyValueCache;

/// Attention mask variant.
#[derive(Debug, Clone)]
pub(crate) enum AttentionMask {
    Array(Array),
    Causal,
}

impl<'a> From<&'a AttentionMask> for ScaledDotProductAttentionMask<'a> {
    fn from(mask: &'a AttentionMask) -> Self {
        match mask {
            AttentionMask::Array(array) => ScaledDotProductAttentionMask::Array(array),
            AttentionMask::Causal => ScaledDotProductAttentionMask::Causal,
        }
    }
}

/// Non-quantized scaled dot product attention using MLX fast path.
pub(crate) fn scaled_dot_product_attention(
    queries: Array,
    keys: Array,
    values: Array,
    scale: f32,
    mask: Option<&Array>,
) -> Result<Array, Exception> {
    mlx_rs::fast::scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale,
        mask.map(ScaledDotProductAttentionMask::Array),
    )
}

/// Create a causal attention mask.
#[allow(non_snake_case)]
pub(crate) fn create_causal_mask(N: i32, raw_offset: Option<i32>) -> Result<Array, Exception> {
    let offset = raw_offset.unwrap_or(0);

    let row_indices = arange!(stop = offset + N)?;
    let col_indices = arange!(start = offset, stop = offset + N)?;
    let col_expanded = col_indices.index((.., NewAxis));
    let row_expanded = row_indices.index(NewAxis);

    col_expanded.ge(&row_expanded)
}

/// Create an attention mask from the hidden state and cache.
#[allow(non_snake_case)]
pub(crate) fn create_attention_mask<C>(
    h: &Array,
    cache: &[Option<C>],
    as_array: Option<bool>,
) -> Result<Option<AttentionMask>, Exception>
where
    C: KeyValueCache,
{
    let use_array = as_array.unwrap_or(false);
    let shape = h.shape();
    let T = *shape
        .get(1)
        .ok_or_else(|| Exception::custom("Hidden state must have at least 2 dimensions"))?;

    if T > 1 {
        let mut offset = 0;
        if let Some(c) = cache.first().and_then(|c| c.as_ref()) {
            offset = c.offset();
        }

        if use_array {
            create_causal_mask(T, Some(offset))
                .map(AttentionMask::Array)
                .map(Some)
        } else {
            Ok(Some(AttentionMask::Causal))
        }
    } else {
        Ok(None)
    }
}
