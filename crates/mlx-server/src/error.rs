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
