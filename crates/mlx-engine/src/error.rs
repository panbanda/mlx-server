use mlx_models::error::ModelError;

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("Model error: {0}")]
    Model(#[from] ModelError),

    #[error("MLX error: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),

    #[error("Tokenization error: {0}")]
    Tokenization(String),

    #[error("Template error: {0}")]
    Template(String),

    #[error("Generation error: {0}")]
    Generation(String),
}
