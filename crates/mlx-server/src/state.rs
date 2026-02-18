use std::collections::HashMap;
use std::sync::Arc;

use mlx_engine::simple::SimpleEngine;

use crate::config::ServerConfig;

/// Shared application state available to all route handlers.
pub struct AppState {
    /// Inference engines keyed by model name, one per loaded model.
    pub engines: HashMap<String, Arc<SimpleEngine>>,
    /// Server configuration (host, port, etc.).
    pub config: ServerConfig,
}

impl AppState {
    /// Look up an engine by the model name from the request.
    pub fn engine_for(&self, model: &str) -> Option<Arc<SimpleEngine>> {
        self.engines.get(model).cloned()
    }
}

/// Type alias for the shared state used by Axum handlers.
pub type SharedState = Arc<AppState>;
