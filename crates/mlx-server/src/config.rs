use clap::Parser;

/// MLX Server - OpenAI-compatible inference server for Apple Silicon.
#[derive(Parser, Debug)]
#[command(author, version, about)]
pub struct ServerConfig {
    /// Path to the model directory (local path or HuggingFace model name).
    #[arg(long, default_value = "~/dev/models/Arch-Router-1.5B-4bit")]
    pub model: String,

    /// Host to bind the server to.
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,

    /// Port to bind the server to.
    #[arg(long, default_value_t = 8000)]
    pub port: u16,

    /// Default maximum tokens for generation.
    #[arg(long, default_value_t = 32768)]
    pub max_tokens: u32,

    /// API key for authentication (if unset, no auth required).
    #[arg(long)]
    pub api_key: Option<String>,

    /// Rate limit (requests per minute per client, 0 = disabled).
    #[arg(long, default_value_t = 0)]
    pub rate_limit: u32,

    /// Default request timeout in seconds.
    #[arg(long, default_value_t = 300.0)]
    pub timeout: f64,
}
