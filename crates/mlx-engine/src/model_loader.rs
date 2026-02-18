use std::path::{Path, PathBuf};

use mlx_models::{AnyModel, load_tokenizer as shared_load_tokenizer, registry, transformer};

use crate::error::EngineError;

/// Configuration for loading a model from a directory.
pub struct ModelConfig {
    pub model_dir: PathBuf,
    pub model_type: String,
}

impl ModelConfig {
    /// Detect model type and create a config from a model directory.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, EngineError> {
        let model_dir = dir.as_ref().to_path_buf();
        let model_type = registry::detect_model_type(&model_dir)?;

        if !registry::is_supported(&model_type) {
            return Err(EngineError::Model(
                mlx_models::error::ModelError::UnsupportedModel(model_type),
            ));
        }

        Ok(Self {
            model_dir,
            model_type,
        })
    }
}

/// Load a model from a directory, auto-detecting the architecture.
pub fn load_model(model_dir: impl AsRef<Path>) -> Result<AnyModel, EngineError> {
    let config = ModelConfig::from_dir(&model_dir)?;

    match config.model_type.as_str() {
        "qwen2" | "qwen3" | "llama" | "mistral" => {
            let model = transformer::load_model(&config.model_dir).map_err(EngineError::Model)?;
            Ok(AnyModel::Transformer(model))
        }
        "qwen3_next" => {
            let model = mlx_models::qwen3_next::load_qwen3_next_model(&config.model_dir)
                .map_err(EngineError::Model)?;
            Ok(AnyModel::Qwen3Next(model))
        }
        other => Err(EngineError::Model(
            mlx_models::error::ModelError::UnsupportedModel(other.to_owned()),
        )),
    }
}

/// Load a tokenizer from a model directory.
pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<tokenizers::Tokenizer, EngineError> {
    shared_load_tokenizer(model_dir).map_err(|e| EngineError::Tokenization(e.to_string()))
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn model_config_from_dir_qwen2() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), r#"{"model_type": "qwen2"}"#).unwrap();
        let config = ModelConfig::from_dir(dir.path()).unwrap();
        assert_eq!(config.model_type, "qwen2");
        assert_eq!(config.model_dir, dir.path());
    }

    #[test]
    fn model_config_from_dir_qwen3() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), r#"{"model_type": "qwen3"}"#).unwrap();
        let config = ModelConfig::from_dir(dir.path()).unwrap();
        assert_eq!(config.model_type, "qwen3");
    }

    #[test]
    fn model_config_from_dir_llama() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), r#"{"model_type": "llama"}"#).unwrap();
        let config = ModelConfig::from_dir(dir.path()).unwrap();
        assert_eq!(config.model_type, "llama");
    }

    #[test]
    fn model_config_from_dir_mistral() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "mistral"}"#,
        )
        .unwrap();
        let config = ModelConfig::from_dir(dir.path()).unwrap();
        assert_eq!(config.model_type, "mistral");
    }

    #[test]
    fn model_config_from_dir_qwen3_next() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "qwen3_next"}"#,
        )
        .unwrap();
        let config = ModelConfig::from_dir(dir.path()).unwrap();
        assert_eq!(config.model_type, "qwen3_next");
    }

    #[test]
    fn model_config_from_dir_unsupported_model_type() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), r#"{"model_type": "gpt2"}"#).unwrap();
        let result = ModelConfig::from_dir(dir.path());
        match result {
            Err(e) => assert!(e.to_string().contains("gpt2")),
            Ok(_) => panic!("Expected error for unsupported model type"),
        }
    }

    #[test]
    fn model_config_from_dir_missing_config_json() {
        let dir = tempfile::tempdir().unwrap();
        let result = ModelConfig::from_dir(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn model_config_from_dir_invalid_json() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), "not valid json {{{").unwrap();
        let result = ModelConfig::from_dir(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn model_config_from_dir_missing_model_type_field() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"vocab_size": 32000, "hidden_size": 4096}"#,
        )
        .unwrap();
        let result = ModelConfig::from_dir(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn load_tokenizer_missing_tokenizer_json() {
        let dir = tempfile::tempdir().unwrap();
        let result = load_tokenizer(dir.path());
        match result {
            Err(e) => assert!(e.to_string().contains("Tokenization error")),
            Ok(_) => panic!("Expected error for missing tokenizer.json"),
        }
    }
}
