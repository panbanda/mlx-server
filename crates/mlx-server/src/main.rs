mod anthropic_adapter;
mod config;
mod error;
mod routes;
mod state;
mod types;

use std::net::SocketAddr;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;

use axum::{
    Router,
    extract::{ConnectInfo, Request},
    middleware::{self, Next},
    response::Response,
    routing::{get, post},
};
use clap::Parser;
use governor::{Quota, RateLimiter, clock::DefaultClock, state::keyed::DefaultKeyedStateStore};
use mlx_engine::simple::SimpleEngine;
use tower_http::{
    cors::CorsLayer, timeout::TimeoutLayer, trace::TraceLayer,
    validate_request::ValidateRequestHeaderLayer,
};

use axum::http::StatusCode;

use crate::{config::ServerConfig, state::AppState};

type SharedRateLimiter = Arc<RateLimiter<String, DefaultKeyedStateStore<String>, DefaultClock>>;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing (silently falls back to "info" if RUST_LOG is unset or invalid)
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config = ServerConfig::parse();

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

fn build_router(
    state: Arc<AppState>,
    timeout_secs: f64,
    api_key: Option<String>,
    rate_limit: u32,
) -> Router {
    let timeout_duration = Duration::from_secs_f64(timeout_secs);

    // Build API routes (v1/*)
    let mut api_routes = Router::new()
        .route("/v1/models", get(routes::models::list_models))
        .route("/v1/chat/completions", post(routes::chat::chat_completions))
        .route("/v1/completions", post(routes::completions::completions))
        .route("/v1/embeddings", post(routes::embeddings::embeddings))
        .route("/v1/messages", post(routes::anthropic::create_message))
        .route(
            "/v1/messages/count_tokens",
            post(routes::anthropic::count_tokens),
        );

    // Apply rate limiting to API routes if configured
    if let Some(rpm) = NonZeroU32::new(rate_limit) {
        let limiter: SharedRateLimiter = Arc::new(RateLimiter::keyed(Quota::per_minute(rpm)));
        api_routes = api_routes.layer(middleware::from_fn(move |req, next| {
            let limiter_clone = Arc::clone(&limiter);
            rate_limit_middleware(limiter_clone, req, next)
        }));
        tracing::info!(requests_per_minute = rate_limit, "Rate limiting enabled");
    }

    // Apply bearer auth to API routes if configured
    if let Some(ref key) = api_key {
        #[allow(deprecated)]
        // tower-http deprecated this as "too basic", but it's fine for a local inference server
        let auth_layer = ValidateRequestHeaderLayer::bearer(key);
        api_routes = api_routes.layer(auth_layer);
        tracing::info!("API key authentication enabled");
    }

    Router::new()
        .route("/health", get(routes::health::health))
        .merge(api_routes)
        .layer(TraceLayer::new_for_http())
        .layer(TimeoutLayer::with_status_code(
            StatusCode::GATEWAY_TIMEOUT,
            timeout_duration,
        ))
        // CorsLayer::permissive() is fine for local dev; tighten for production
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn rate_limit_middleware(
    limiter: SharedRateLimiter,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let key = req
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ci| ci.0.ip().to_string())
        .unwrap_or_else(|| "unknown".to_owned());

    match limiter.check_key(&key) {
        Ok(_) => Ok(next.run(req).await),
        Err(_) => Err(StatusCode::TOO_MANY_REQUESTS),
    }
}
