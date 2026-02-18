use clap::Parser;
use figment::{
    Figment,
    providers::{Env, Serialized},
};
use serde::{Deserialize, Serialize};

/// CLI arguments (used as the highest-priority figment layer).
#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "MLX Server - OpenAI-compatible inference server for Apple Silicon"
)]
struct CliArgs {
    /// Path to the model directory or HuggingFace model ID.
    #[arg(long)]
    model: Option<String>,

    /// Host to bind the server to.
    #[arg(long)]
    host: Option<String>,

    /// Port to bind the server to.
    #[arg(long)]
    port: Option<u16>,

    /// Default maximum tokens for generation.
    #[arg(long)]
    max_tokens: Option<u32>,

    /// API key for authentication (if unset, no auth required).
    #[arg(long)]
    api_key: Option<String>,

    /// Rate limit (requests per minute per client, 0 = disabled).
    #[arg(long)]
    rate_limit: Option<u32>,

    /// Default request timeout in seconds.
    #[arg(long)]
    timeout: Option<f64>,
}

/// Resolved server configuration.
///
/// Layered resolution order (later wins):
/// 1. Built-in defaults
/// 2. `MLX_SERVER_*` environment variables
/// 3. CLI arguments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub model: String,
    pub host: String,
    pub port: u16,
    pub max_tokens: u32,
    pub api_key: Option<String>,
    pub rate_limit: u32,
    pub timeout: f64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            model: default_model_path(),
            host: "0.0.0.0".to_owned(),
            port: 8000,
            max_tokens: 32768,
            api_key: None,
            rate_limit: 0,
            timeout: 300.0,
        }
    }
}

/// Default model directory: `~/.cache/huggingface/hub/`
///
/// This matches the standard HuggingFace cache location shared with
/// mlx-lm, transformers, and other HF ecosystem tools.
fn default_model_path() -> String {
    directories::BaseDirs::new()
        .map(|dirs| {
            dirs.cache_dir()
                .join("huggingface")
                .join("hub")
                .to_string_lossy()
                .into_owned()
        })
        .unwrap_or_else(|| "~/.cache/huggingface/hub".to_owned())
}

impl ServerConfig {
    /// Load configuration from defaults, environment, and CLI args.
    pub fn load() -> Result<Self, Box<figment::Error>> {
        let cli = CliArgs::parse();

        let mut figment = Figment::new()
            .merge(Serialized::defaults(ServerConfig::default()))
            .merge(Env::prefixed("MLX_SERVER_"));

        // Overlay CLI args (only non-None values)
        if let Some(ref model) = cli.model {
            figment = figment.merge(Serialized::default("model", model));
        }
        if let Some(ref host) = cli.host {
            figment = figment.merge(Serialized::default("host", host));
        }
        if let Some(port) = cli.port {
            figment = figment.merge(Serialized::default("port", port));
        }
        if let Some(max_tokens) = cli.max_tokens {
            figment = figment.merge(Serialized::default("max_tokens", max_tokens));
        }
        if let Some(ref api_key) = cli.api_key {
            figment = figment.merge(Serialized::default("api_key", api_key));
        }
        if let Some(rate_limit) = cli.rate_limit {
            figment = figment.merge(Serialized::default("rate_limit", rate_limit));
        }
        if let Some(timeout) = cli.timeout {
            figment = figment.merge(Serialized::default("timeout", timeout));
        }

        figment.extract().map_err(Box::new)
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_has_reasonable_values() {
        let config = ServerConfig::default();
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 8000);
        assert_eq!(config.max_tokens, 32768);
        assert!(config.api_key.is_none());
        assert_eq!(config.rate_limit, 0);
        assert!((config.timeout - 300.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_default_model_path_contains_huggingface() {
        let path = default_model_path();
        assert!(path.contains("huggingface"));
        assert!(path.contains("hub"));
    }

    #[test]
    fn test_config_layered_override() {
        // Verifies that later figment layers override earlier ones,
        // which is the same mechanism used by Env::prefixed("MLX_SERVER_").
        let figment = Figment::new()
            .merge(Serialized::defaults(ServerConfig::default()))
            .merge(Serialized::default("port", 9000_u16))
            .merge(Serialized::default("host", "127.0.0.1"));

        let config: ServerConfig = figment.extract().unwrap();
        assert_eq!(config.port, 9000);
        assert_eq!(config.host, "127.0.0.1");
    }

    #[test]
    fn test_cli_overrides_defaults() {
        let figment = Figment::new()
            .merge(Serialized::defaults(ServerConfig::default()))
            .merge(Serialized::default("max_tokens", 1024_u32))
            .merge(Serialized::default("timeout", 60.0_f64));

        let config: ServerConfig = figment.extract().unwrap();
        assert_eq!(config.max_tokens, 1024);
        assert!((config.timeout - 60.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_api_key_can_be_set() {
        let figment = Figment::new()
            .merge(Serialized::defaults(ServerConfig::default()))
            .merge(Serialized::default("api_key", "sk-test-123"));

        let config: ServerConfig = figment.extract().unwrap();
        assert_eq!(config.api_key, Some("sk-test-123".to_owned()));
    }
}
