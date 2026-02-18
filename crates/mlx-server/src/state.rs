use std::sync::Arc;

use mlx_engine::simple::SimpleEngine;

use crate::config::ServerConfig;

/// Shared application state available to all route handlers.
pub struct AppState {
    pub engine: SimpleEngine,
    pub config: ServerConfig,
}

/// Type alias for the shared state used by Axum handlers.
pub type SharedState = Arc<AppState>;
