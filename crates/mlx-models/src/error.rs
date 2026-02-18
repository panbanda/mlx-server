use mlx_rs::error::{Exception, IoError};

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("MLX error: {0}")]
    Mlx(#[from] Exception),

    #[error("MLX IO error: {0}")]
    MlxIo(#[from] IoError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Unsupported model type: {0}")]
    UnsupportedModel(String),

    #[error("Missing weight: {0}")]
    MissingWeight(String),

    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
}
