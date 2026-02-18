use std::sync::Arc;

use mlx_engine::simple::SimpleEngine;

use crate::config::ServerConfig;

/// Shared application state available to all route handlers.
pub struct AppState {
    /// The inference engine used to run model generation.
    pub engine: SimpleEngine,
    /// Server configuration (host, port, model path, etc.).
    pub config: ServerConfig,
}

/// Type alias for the shared state used by Axum handlers.
pub type SharedState = Arc<AppState>;
