use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Serialize;

/// JSON error response body (OpenAI-compatible).
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
    pub code: Option<String>,
}

/// Server error types.
#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error("Engine error: {0}")]
    Engine(#[from] mlx_engine::error::EngineError),

    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Internal error: {0}")]
    InternalError(String),
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let (status, error_type, message) = match &self {
            ServerError::Engine(e) => {
                tracing::error!(error = %e, "Engine error");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "server_error",
                    "Internal server error".to_owned(),
                )
            }
            ServerError::BadRequest(msg) => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                msg.clone(),
            ),
            ServerError::InternalError(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                msg.clone(),
            ),
        };

        let body = Json(ErrorResponse {
            error: ErrorDetail {
                message,
                r#type: error_type.to_owned(),
                code: None,
            },
        });

        (status, body).into_response()
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;
    use http_body_util::BodyExt;

    async fn response_status_and_body(resp: Response) -> (StatusCode, serde_json::Value) {
        let status = resp.status();
        let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        (status, body)
    }

    #[tokio::test]
    async fn test_engine_error_returns_500_with_masked_message() {
        let engine_err =
            mlx_engine::error::EngineError::Generation("sensitive internal details".to_owned());
        let error = ServerError::Engine(engine_err);
        let resp = error.into_response();
        let (status, body) = response_status_and_body(resp).await;

        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        let message = body["error"]["message"].as_str().unwrap();
        assert_eq!(message, "Internal server error");
        // The actual engine error details must NOT leak
        assert!(!message.contains("sensitive internal details"));
        assert_eq!(body["error"]["type"].as_str().unwrap(), "server_error");
    }

    #[tokio::test]
    async fn test_bad_request_returns_400_with_actual_message() {
        let error = ServerError::BadRequest("missing field: model".to_owned());
        let resp = error.into_response();
        let (status, body) = response_status_and_body(resp).await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(
            body["error"]["message"].as_str().unwrap(),
            "missing field: model"
        );
        assert_eq!(
            body["error"]["type"].as_str().unwrap(),
            "invalid_request_error"
        );
    }

    #[tokio::test]
    async fn test_internal_error_returns_500_with_actual_message() {
        let error = ServerError::InternalError("disk full".to_owned());
        let resp = error.into_response();
        let (status, body) = response_status_and_body(resp).await;

        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(body["error"]["message"].as_str().unwrap(), "disk full");
        assert_eq!(body["error"]["type"].as_str().unwrap(), "server_error");
    }

    #[tokio::test]
    async fn test_error_code_field_is_null() {
        let error = ServerError::BadRequest("test".to_owned());
        let resp = error.into_response();
        let (_, body) = response_status_and_body(resp).await;

        assert!(body["error"]["code"].is_null());
    }

    #[tokio::test]
    async fn test_engine_tokenization_error_masked() {
        let engine_err = mlx_engine::error::EngineError::Tokenization(
            "tokenizer failed on byte 0xFF".to_owned(),
        );
        let error = ServerError::Engine(engine_err);
        let resp = error.into_response();
        let (status, body) = response_status_and_body(resp).await;

        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        let message = body["error"]["message"].as_str().unwrap();
        assert_eq!(message, "Internal server error");
        assert!(!message.contains("0xFF"));
    }

    #[tokio::test]
    async fn test_engine_template_error_masked() {
        let engine_err =
            mlx_engine::error::EngineError::Template("template parse failed".to_owned());
        let error = ServerError::Engine(engine_err);
        let resp = error.into_response();
        let (status, body) = response_status_and_body(resp).await;

        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        let message = body["error"]["message"].as_str().unwrap();
        assert_eq!(message, "Internal server error");
        assert!(!message.contains("template parse failed"));
    }
}
