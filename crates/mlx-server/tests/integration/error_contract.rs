//! Tests that ServerError variants produce the correct HTTP status codes,
//! response shapes, and content types per the OpenAI error contract.

#![allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]

use axum::http::StatusCode;
use axum::response::IntoResponse;
use http_body_util::BodyExt;
use mlx_server::error::ServerError;

async fn extract_response(error: ServerError) -> (StatusCode, serde_json::Value, String) {
    let response = error.into_response();
    let status = response.status();
    let content_type = response
        .headers()
        .get("content-type")
        .map(|v| v.to_str().unwrap().to_owned())
        .unwrap_or_default();
    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
    (status, body, content_type)
}

#[tokio::test]
async fn bad_request_returns_400() {
    let error = ServerError::BadRequest("field 'model' is required".to_owned());
    let (status, body, content_type) = extract_response(error).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(content_type.contains("application/json"));
    assert_eq!(body["error"]["message"], "field 'model' is required");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(body["error"]["code"].is_null());
}

#[tokio::test]
async fn internal_error_returns_500_with_masked_message() {
    let error = ServerError::InternalError("database unreachable".to_owned());
    let (status, body, _) = extract_response(error).await;

    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    let message = body["error"]["message"].as_str().unwrap();
    assert_eq!(message, "Internal server error");
    assert!(
        !message.contains("database unreachable"),
        "Internal error details leaked to client"
    );
    assert_eq!(body["error"]["type"], "server_error");
}

#[tokio::test]
async fn engine_error_returns_500_with_masked_message() {
    let engine_err =
        mlx_engine::error::EngineError::Generation("detailed internal stack trace".to_owned());
    let error = ServerError::Engine(engine_err);
    let (status, body, _) = extract_response(error).await;

    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    // Engine errors must NOT leak internal details to the client
    let message = body["error"]["message"].as_str().unwrap();
    assert_eq!(message, "Internal server error");
    assert!(
        !message.contains("detailed internal stack trace"),
        "Engine error details leaked to client"
    );
    assert_eq!(body["error"]["type"], "server_error");
}

#[tokio::test]
async fn engine_tokenization_error_masked() {
    let engine_err =
        mlx_engine::error::EngineError::Tokenization("invalid byte 0xFF at position 42".to_owned());
    let error = ServerError::Engine(engine_err);
    let (status, body, _) = extract_response(error).await;

    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    let message = body["error"]["message"].as_str().unwrap();
    assert!(!message.contains("0xFF"));
}

#[tokio::test]
async fn engine_template_error_masked() {
    let engine_err =
        mlx_engine::error::EngineError::Template("missing variable 'content'".to_owned());
    let error = ServerError::Engine(engine_err);
    let (status, body, _) = extract_response(error).await;

    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    let message = body["error"]["message"].as_str().unwrap();
    assert_eq!(message, "Internal server error");
}

#[tokio::test]
async fn error_response_has_openai_shape() {
    // All error responses must have the shape: { "error": { "message": "...", "type": "...", "code": ... } }
    let error = ServerError::BadRequest("test".to_owned());
    let (_, body, _) = extract_response(error).await;

    assert!(body.get("error").is_some(), "missing top-level 'error' key");
    let error_obj = &body["error"];
    assert!(
        error_obj.get("message").is_some(),
        "missing 'error.message'"
    );
    assert!(error_obj.get("type").is_some(), "missing 'error.type'");
    assert!(error_obj.get("code").is_some(), "missing 'error.code'");
}

#[tokio::test]
async fn error_response_content_type_is_json() {
    let error = ServerError::BadRequest("test".to_owned());
    let (_, _, content_type) = extract_response(error).await;

    assert!(
        content_type.contains("application/json"),
        "Expected application/json, got: {content_type}"
    );
}
