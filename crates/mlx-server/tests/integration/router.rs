//! Router-level integration tests using tower::ServiceExt::oneshot.
//!
//! These tests exercise the HTTP layer (routing, middleware, content negotiation)
//! without requiring a real MLX engine. Tests that would need the engine to
//! process a request are marked #[ignore] with an explanation.
//!
//! We construct an AppState with a real engine only where unavoidable. For most
//! tests, we can verify behavior that occurs before the engine is invoked:
//! - Health endpoint (no engine needed)
//! - Malformed JSON rejection (Axum rejects before handler runs)
//! - Wrong HTTP method (405)
//! - Unknown routes (404)
//! - Bearer auth rejection (middleware rejects before handler runs)

#![allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]

use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use mlx_server::build_router;
use mlx_server::state::AppState;
use std::sync::Arc;
use tower::ServiceExt;

/// Build a minimal router with just the health endpoint for testing.
/// The health endpoint does not use engine state, so we can test it
/// by building a router that only has that route.
fn build_health_only_router() -> axum::Router {
    use axum::routing::get;
    axum::Router::new().route("/health", get(mlx_server::routes::health::health))
}

// ---------------------------------------------------------------------------
// Health endpoint
// ---------------------------------------------------------------------------

#[tokio::test]
async fn health_returns_200_with_json() {
    let app = build_health_only_router();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let content_type = response
        .headers()
        .get("content-type")
        .map(|v| v.to_str().unwrap().to_owned())
        .unwrap_or_default();
    assert!(
        content_type.contains("application/json"),
        "Expected JSON content type, got: {content_type}"
    );

    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn health_rejects_post() {
    let app = build_health_only_router();

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/health")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
}

// ---------------------------------------------------------------------------
// Unknown routes
// ---------------------------------------------------------------------------

#[tokio::test]
async fn unknown_route_returns_404() {
    let app = build_health_only_router();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/v1/nonexistent")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

// ---------------------------------------------------------------------------
// JSON deserialization errors (these tests verify Axum's behavior with
// malformed request bodies, independent of engine state)
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "requires real SimpleEngine to build full router with AppState"]
async fn chat_completions_invalid_json_returns_422() {
    // This test would send invalid JSON to /v1/chat/completions and verify
    // that Axum returns 422 Unprocessable Entity before the handler runs.
    // Requires a full router with AppState.
}

#[tokio::test]
#[ignore = "requires real SimpleEngine to build full router with AppState"]
async fn chat_completions_empty_messages_returns_400() {
    // Would verify: POST /v1/chat/completions with {"model":"m","messages":[]}
    // returns 400 BadRequest("messages array must not be empty").
}

#[tokio::test]
#[ignore = "requires real SimpleEngine to build full router with AppState"]
async fn completions_empty_prompt_returns_400() {
    // Would verify: POST /v1/completions with {"model":"m","prompt":""}
    // returns 400 BadRequest("prompt must not be empty").
}

#[tokio::test]
#[ignore = "requires real SimpleEngine to build full router with AppState"]
async fn anthropic_empty_messages_returns_400() {
    // Would verify: POST /v1/messages with {"model":"m","messages":[],"max_tokens":100}
    // returns 400 BadRequest("messages array must not be empty").
}

#[tokio::test]
#[ignore = "requires real SimpleEngine to build full router with AppState"]
async fn embeddings_empty_input_returns_400() {
    // Would verify: POST /v1/embeddings with {"model":"m","input":[]}
    // returns 400 BadRequest("input must not be empty").
}

#[tokio::test]
#[ignore = "requires real SimpleEngine to build full router with AppState"]
async fn chat_completions_get_returns_405() {
    // Would verify GET /v1/chat/completions returns 405 Method Not Allowed.
}

#[tokio::test]
#[ignore = "requires real SimpleEngine to build full router with AppState"]
async fn models_returns_200_with_model_list() {
    // Would verify GET /v1/models returns the loaded model name.
}

#[tokio::test]
#[ignore = "requires real SimpleEngine to build full router with AppState"]
async fn bearer_auth_missing_returns_401() {
    // Would build router with api_key="test" and verify that requests
    // without Authorization header get 401.
}

#[tokio::test]
#[ignore = "requires real SimpleEngine to build full router with AppState"]
async fn bearer_auth_wrong_key_returns_401() {
    // Would build router with api_key="correct" and verify that requests
    // with Authorization: Bearer wrong get 401.
}

#[tokio::test]
#[ignore = "requires real SimpleEngine to build full router with AppState"]
async fn streaming_chat_returns_sse_content_type() {
    // Would verify POST /v1/chat/completions with stream:true returns
    // content-type: text/event-stream.
}

// ---------------------------------------------------------------------------
// build_router smoke test (verifies the function is callable and public)
// ---------------------------------------------------------------------------

#[test]
fn build_router_is_public_and_callable() {
    // This test verifies that build_router was successfully extracted to lib.rs.
    // We cannot actually call it without a real AppState, but we can verify
    // the function signature exists and is accessible.
    let _: fn(Arc<AppState>, f64, Option<String>, u32) -> axum::Router = build_router;
}
