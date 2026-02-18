use std::net::SocketAddr;
use std::sync::Arc;

use mlx_engine::simple::SimpleEngine;

use mlx_server::{build_router, config::ServerConfig, state::AppState};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config = ServerConfig::load()?;

    tracing::info!(model = %config.model, "Loading model");
    let engine = SimpleEngine::load(&config.model)?;
    tracing::info!(model_name = %engine.model_name(), "Model loaded");

    let timeout_secs = config.timeout;
    if !timeout_secs.is_finite() || timeout_secs <= 0.0 {
        return Err("timeout must be a positive, finite number".into());
    }
    let api_key = config.api_key.clone();
    let rate_limit = config.rate_limit;
    let bind_addr = format!("{}:{}", config.host, config.port);

    let shared_state = Arc::new(AppState { engine, config });

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
