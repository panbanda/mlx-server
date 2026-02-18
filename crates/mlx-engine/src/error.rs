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

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_error_display_tokenization() {
        let err = EngineError::Tokenization("bad token".to_owned());
        assert!(err.to_string().contains("bad token"));
    }

    #[test]
    fn test_engine_error_display_template() {
        let err = EngineError::Template("syntax error".to_owned());
        assert!(err.to_string().contains("syntax error"));
    }

    #[test]
    fn test_engine_error_display_generation() {
        let err = EngineError::Generation("out of memory".to_owned());
        assert!(err.to_string().contains("out of memory"));
    }
}
