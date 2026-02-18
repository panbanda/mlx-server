use std::convert::Infallible;

use axum::{
    Json,
    extract::State,
    response::{
        IntoResponse, Sse,
        sse::{Event, KeepAlive},
    },
};
use tokio_stream::Stream;

use crate::{
    error::ServerError,
    state::SharedState,
    types::openai::{
        ChatCompletionChoice, ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionDelta,
        ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse, CompletionUsage,
        StopSequence, ToolCall, ToolCallFunction,
    },
};

pub async fn chat_completions(
    State(state): State<SharedState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, ServerError> {
    if req.messages.is_empty() {
        return Err(ServerError::BadRequest(
            "messages array must not be empty".to_owned(),
        ));
    }

    if req.response_format.is_some() {
        tracing::warn!("response_format is not yet enforced");
    }

    if req.stream == Some(true) {
        let stream = chat_completions_stream(state, req)?;
        let sse = Sse::new(stream).keep_alive(KeepAlive::default());
        Ok(sse.into_response())
    } else {
        let response = chat_completions_non_streaming(state, req).await?;
        Ok(Json(response).into_response())
    }
}

async fn chat_completions_non_streaming(
    state: SharedState,
    req: ChatCompletionRequest,
) -> Result<ChatCompletionResponse, ServerError> {
    let max_tokens = req.max_tokens.unwrap_or(state.config.max_tokens);
    let temperature = req.temperature.unwrap_or(1.0);
    let top_p = req.top_p.unwrap_or(1.0);
    let stop_sequences = StopSequence::extract(req.stop);

    let messages = convert_messages(&req.messages);
    let tools = req.tools.as_deref();

    let prompt_tokens = state
        .engine
        .prepare_chat_prompt(&messages, tools)
        .map_err(ServerError::Engine)?;

    let output = tokio::task::spawn_blocking(move || {
        state.engine.generate(
            &prompt_tokens,
            max_tokens,
            temperature,
            top_p,
            &stop_sequences,
        )
    })
    .await
    .map_err(|e| ServerError::InternalError(format!("Task join error: {e}")))?
    .map_err(ServerError::Engine)?;

    let request_id = generate_request_id();
    let has_tools = req.tools.is_some();

    let (content, tool_calls, finish_reason) = if has_tools {
        let parsed = mlx_engine::tool_parser::parse_tool_calls(&output.text);
        if parsed.tool_calls.is_empty() {
            (Some(output.text), None, output.finish_reason)
        } else {
            let calls: Vec<ToolCall> = parsed
                .tool_calls
                .iter()
                .enumerate()
                .map(|(i, tc)| ToolCall {
                    id: format!("call_{i}_{}", uuid::Uuid::new_v4()),
                    r#type: "function".to_owned(),
                    function: ToolCallFunction {
                        name: tc.name.clone(),
                        arguments: tc.arguments.to_string(),
                    },
                })
                .collect();
            let text = if parsed.text.is_empty() {
                None
            } else {
                Some(parsed.text)
            };
            (text, Some(calls), "tool_calls".to_owned())
        }
    } else {
        (Some(output.text), None, output.finish_reason)
    };

    Ok(ChatCompletionResponse {
        id: request_id,
        object: "chat.completion",
        created: current_unix_timestamp(),
        model: req.model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: "assistant".to_owned(),
                content,
                tool_calls,
                tool_call_id: None,
            },
            finish_reason,
        }],
        usage: CompletionUsage {
            prompt_tokens: output.prompt_tokens,
            completion_tokens: output.completion_tokens,
            total_tokens: output.prompt_tokens + output.completion_tokens,
        },
    })
}

fn chat_completions_stream(
    state: SharedState,
    req: ChatCompletionRequest,
) -> Result<impl Stream<Item = Result<Event, Infallible>>, ServerError> {
    if req.tools.is_some() {
        return Err(ServerError::BadRequest(
            "Streaming with tool_calls is not yet supported".to_owned(),
        ));
    }
    let max_tokens = req.max_tokens.unwrap_or(state.config.max_tokens);
    let temperature = req.temperature.unwrap_or(1.0);
    let top_p = req.top_p.unwrap_or(1.0);
    let stop_sequences = StopSequence::extract(req.stop);

    let messages = convert_messages(&req.messages);
    let tools = req.tools.as_deref();

    let prompt_tokens = state
        .engine
        .prepare_chat_prompt(&messages, tools)
        .map_err(ServerError::Engine)?;

    let request_id = generate_request_id();
    let created = current_unix_timestamp();
    let model = req.model;

    let (tx, mut rx) = tokio::sync::mpsc::channel(32);

    tokio::task::spawn_blocking(move || {
        let result = state.engine.generate_streaming(
            &prompt_tokens,
            max_tokens,
            temperature,
            top_p,
            &stop_sequences,
            tx,
        );
        if let Err(e) = result {
            tracing::error!(error = %e, "Generation error during streaming");
        }
    });

    let stream = async_stream::stream! {
        // Send initial role chunk
        let role_chunk = ChatCompletionChunk {
            id: request_id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model.clone(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: ChatCompletionDelta {
                    role: Some("assistant".to_owned()),
                    content: None,
                    tool_calls: None,
                },
                finish_reason: None,
            }],
        };
        match serde_json::to_string(&role_chunk) {
            Ok(json) => yield Ok(Event::default().data(json)),
            Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
        }

        while let Some(output) = rx.recv().await {
            let chunk = ChatCompletionChunk {
                id: request_id.clone(),
                object: "chat.completion.chunk",
                created,
                model: model.clone(),
                choices: vec![ChatCompletionChunkChoice {
                    index: 0,
                    delta: ChatCompletionDelta {
                        role: None,
                        content: Some(output.new_text),
                        tool_calls: None,
                    },
                    finish_reason: output.finish_reason,
                }],
            };
            match serde_json::to_string(&chunk) {
                Ok(json) => yield Ok(Event::default().data(json)),
                Err(e) => tracing::error!(error = %e, "Failed to serialize SSE chunk"),
            }
        }

        // Send [DONE] sentinel
        yield Ok(Event::default().data("[DONE]"));
    };

    Ok(stream)
}

fn convert_messages(
    messages: &[ChatCompletionMessage],
) -> Vec<mlx_engine::chat_template::ChatMessage> {
    messages
        .iter()
        .map(|m| {
            let tool_calls_json = m.tool_calls.as_ref().map(|calls| {
                calls
                    .iter()
                    .filter_map(|tc| serde_json::to_value(tc).ok())
                    .collect()
            });
            mlx_engine::chat_template::ChatMessage {
                role: m.role.clone(),
                content: m.content.clone().unwrap_or_default(),
                tool_calls: tool_calls_json,
            }
        })
        .collect()
}

fn generate_request_id() -> String {
    format!("chatcmpl-{}", uuid::Uuid::new_v4())
}

fn current_unix_timestamp() -> i64 {
    chrono::Utc::now().timestamp()
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    fn simple_message(role: &str, content: Option<&str>) -> ChatCompletionMessage {
        ChatCompletionMessage {
            role: role.to_owned(),
            content: content.map(str::to_owned),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    fn tool_call(id: &str, name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: id.to_owned(),
            r#type: "function".to_owned(),
            function: ToolCallFunction {
                name: name.to_owned(),
                arguments: arguments.to_owned(),
            },
        }
    }

    fn tool_message(role: &str, calls: Vec<ToolCall>) -> ChatCompletionMessage {
        ChatCompletionMessage {
            role: role.to_owned(),
            content: None,
            tool_calls: Some(calls),
            tool_call_id: None,
        }
    }

    #[test]
    fn test_convert_messages() {
        let msgs = vec![
            simple_message("user", Some("Hello")),
            simple_message("assistant", None),
        ];
        let converted = convert_messages(&msgs);
        assert_eq!(converted.len(), 2);
        assert_eq!(converted.first().map(|m| m.role.as_str()), Some("user"));
        assert_eq!(converted.first().map(|m| m.content.as_str()), Some("Hello"));
        assert_eq!(converted.get(1).map(|m| m.content.as_str()), Some(""));
    }

    #[test]
    fn test_generate_request_id_format() {
        let id = generate_request_id();
        assert!(id.starts_with("chatcmpl-"));
    }

    #[test]
    fn test_convert_messages_with_tool_calls() {
        let msgs = vec![tool_message(
            "assistant",
            vec![tool_call("call_1", "get_weather", r#"{"city":"NYC"}"#)],
        )];
        let converted = convert_messages(&msgs);
        assert_eq!(converted.len(), 1);
        let calls = converted
            .first()
            .and_then(|m| m.tool_calls.as_ref())
            .unwrap();
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn test_convert_messages_empty_list() {
        let result = convert_messages(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_convert_messages_with_null_content() {
        let msgs = vec![simple_message("assistant", None)];
        let converted = convert_messages(&msgs);
        assert_eq!(converted.len(), 1);
        assert_eq!(converted.first().map(|m| m.content.as_str()), Some(""));
    }

    #[test]
    fn test_convert_messages_with_tool_calls_complex_arguments() {
        let msgs = vec![tool_message(
            "assistant",
            vec![
                tool_call(
                    "call_1",
                    "search",
                    r#"{"query":"rust programming","filters":{"language":"en","year":2024}}"#,
                ),
                tool_call("call_2", "calculate", r#"{"expression":"2+2"}"#),
            ],
        )];
        let converted = convert_messages(&msgs);
        assert_eq!(converted.len(), 1);
        let calls = converted
            .first()
            .and_then(|m| m.tool_calls.as_ref())
            .unwrap();
        assert_eq!(calls.len(), 2);
    }

    #[test]
    fn test_generate_request_id_uniqueness() {
        let mut ids = std::collections::HashSet::new();
        for _ in 0..100 {
            let id = generate_request_id();
            assert!(ids.insert(id), "duplicate request ID generated");
        }
        assert_eq!(ids.len(), 100);
    }

    #[test]
    fn test_generate_request_id_prefix() {
        let id = generate_request_id();
        assert!(id.starts_with("chatcmpl-"));
        assert!(id.len() > "chatcmpl-".len());
    }

    #[test]
    fn test_current_unix_timestamp_reasonable_value() {
        let ts = current_unix_timestamp();
        assert!(ts > 1_700_000_000, "timestamp too old: {ts}");
        assert!(ts < 2_000_000_000, "timestamp too far in future: {ts}");
    }
}
