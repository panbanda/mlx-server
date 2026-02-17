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
    matches!(model_type, "qwen2" | "qwen3" | "llama" | "mistral")
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
}
