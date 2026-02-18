use std::path::Path;

use crate::error::ModelError;

/// Detect the model architecture from config.json's `model_type` field.
pub fn detect_model_type(model_dir: impl AsRef<Path>) -> Result<String, ModelError> {
    let config_path = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(config_path)?;
    let config: serde_json::Value = serde_json::from_reader(file)?;

    config
        .get("model_type")
        .and_then(|v| v.as_str())
        .map(|s| s.to_owned())
        .ok_or_else(|| ModelError::UnsupportedModel("missing model_type in config.json".into()))
}

/// Supported model architectures.
pub fn is_supported(model_type: &str) -> bool {
    matches!(
        model_type,
        "qwen2" | "qwen3" | "llama" | "mistral" | "qwen3_next"
    )
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_supported_models() {
        assert!(is_supported("qwen2"));
        assert!(is_supported("qwen3"));
        assert!(is_supported("llama"));
        assert!(is_supported("mistral"));
        assert!(!is_supported("gpt2"));
        assert!(!is_supported("unknown"));
    }

    #[test]
    fn test_detect_model_type_missing_config_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let result = detect_model_type(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_detect_model_type_missing_model_type_field() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), r#"{"vocab_size": 32000}"#).unwrap();
        let result = detect_model_type(dir.path());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Unsupported model"));
    }

    #[test]
    fn test_detect_model_type_valid_config() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), r#"{"model_type": "qwen2"}"#).unwrap();
        let result = detect_model_type(dir.path()).unwrap();
        assert_eq!(result, "qwen2");
    }

    #[test]
    fn test_is_supported_empty_string() {
        assert!(!is_supported(""));
    }

    #[test]
    fn test_is_supported_case_sensitive() {
        assert!(!is_supported("Qwen2"));
        assert!(!is_supported("LLAMA"));
    }
}
