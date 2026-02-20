use std::path::Path;

use crate::error::ModelError;

/// Detect the model architecture from config.json's `model_type` field.
pub fn detect_model_type<P: AsRef<Path>>(model_dir: P) -> Result<String, ModelError> {
    let config_path = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(config_path)?;
    let config: serde_json::Value = serde_json::from_reader(file)?;

    config
        .get("model_type")
        .and_then(|v| v.as_str())
        .map(ToOwned::to_owned)
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

    /// Write a config.json with the given raw JSON content and return the tempdir.
    fn write_config(json: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), json).unwrap();
        dir
    }

    /// Write a config.json containing only `{"model_type": "<value>"}`.
    fn write_model_type_config(model_type: &str) -> tempfile::TempDir {
        write_config(&format!(r#"{{"model_type": "{model_type}"}}"#))
    }

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
        let dir = write_config(r#"{"vocab_size": 32000}"#);
        let err = detect_model_type(dir.path()).unwrap_err();
        assert!(err.to_string().contains("Unsupported model"));
    }

    #[test]
    fn test_detect_model_type_valid_config() {
        let dir = write_model_type_config("qwen2");
        assert_eq!(detect_model_type(dir.path()).unwrap(), "qwen2");
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

    #[test]
    fn test_is_supported_qwen3_next() {
        assert!(is_supported("qwen3_next"));
    }

    #[test]
    fn test_detect_model_type_qwen3() {
        let dir = write_model_type_config("qwen3");
        assert_eq!(detect_model_type(dir.path()).unwrap(), "qwen3");
    }

    #[test]
    fn test_detect_model_type_llama() {
        let dir = write_model_type_config("llama");
        assert_eq!(detect_model_type(dir.path()).unwrap(), "llama");
    }

    #[test]
    fn test_detect_model_type_mistral() {
        let dir = write_model_type_config("mistral");
        assert_eq!(detect_model_type(dir.path()).unwrap(), "mistral");
    }

    #[test]
    fn test_detect_model_type_qwen3_next() {
        let dir = write_model_type_config("qwen3_next");
        assert_eq!(detect_model_type(dir.path()).unwrap(), "qwen3_next");
    }

    #[test]
    fn test_detect_model_type_null_value() {
        let dir = write_config(r#"{"model_type": null}"#);
        let err = detect_model_type(dir.path()).unwrap_err();
        assert!(err.to_string().contains("Unsupported model"));
    }

    #[test]
    fn test_detect_model_type_number_value() {
        let dir = write_config(r#"{"model_type": 42}"#);
        let err = detect_model_type(dir.path()).unwrap_err();
        assert!(err.to_string().contains("Unsupported model"));
    }

    #[test]
    fn test_detect_model_type_empty_string_value() {
        let dir = write_model_type_config("");
        assert_eq!(detect_model_type(dir.path()).unwrap(), "");
    }

    #[test]
    fn test_detect_model_type_empty_json_object() {
        let dir = write_config(r"{}");
        let err = detect_model_type(dir.path()).unwrap_err();
        assert!(err.to_string().contains("Unsupported model"));
    }

    #[test]
    fn test_detect_model_type_invalid_json() {
        let dir = write_config("not json at all");
        assert!(detect_model_type(dir.path()).is_err());
    }
}
