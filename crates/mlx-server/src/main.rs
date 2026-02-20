use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use mlx_engine::simple::SimpleEngine;

use mlx_server::{build_router, config::ServerConfig, model_resolver, state::AppState};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config = ServerConfig::load()?;

    let mut engines = HashMap::new();
    for model_path in &config.models {
        tracing::info!(model = %model_path, "Resolving model path");
        let resolved = model_resolver::resolve(model_path)?;
        tracing::info!(model = %model_path, resolved = %resolved.display(), "Loading model");
        let engine = SimpleEngine::load(&resolved)?;
        let name = engine.model_name().to_owned();
        tracing::info!(model_name = %name, "Model loaded");
        if engines.insert(name.clone(), Arc::new(engine)).is_some() {
            return Err(format!(
                "model name collision: two model paths resolve to the same name '{name}'"
            )
            .into());
        }
    }

    let timeout_secs = config.timeout;
    if !timeout_secs.is_finite() || timeout_secs <= 0.0 {
        return Err("timeout must be a positive, finite number".into());
    }
    let api_key = config.api_key.clone();
    let rate_limit = config.rate_limit;
    let bind_addr = format!("{}:{}", config.host, config.port);

    let shared_state = Arc::new(AppState { engines, config });

    let app = build_router(shared_state, timeout_secs, api_key, rate_limit);

    tracing::info!(addr = %bind_addr, "Starting server");
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await?;

    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("Shutdown signal received, draining connections");
}
