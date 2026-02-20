use serde::{Deserialize, Serialize};

/// POST /v1/chat/completions request body.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatCompletionMessage>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<StopSequence>,
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
}

/// Response format specification.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResponseFormat {
    pub r#type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<serde_json::Value>,
}

/// A message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// A tool call in the `OpenAI` format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: ToolCallFunction,
}

/// The function details of a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

/// Stop sequence: single string or array of strings.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StopSequence {
    Single(String),
    Multiple(Vec<String>),
}

impl StopSequence {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::Single(s) => vec![s],
            Self::Multiple(v) => v,
        }
    }

    /// Extract stop sequences from an optional value, returning empty vec if None.
    pub fn extract(stop: Option<Self>) -> Vec<String> {
        stop.map_or_else(Vec::new, Self::into_vec)
    }
}

/// POST /v1/chat/completions response (non-streaming).
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: CompletionUsage,
}

/// A choice in a chat completion response.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatCompletionMessage,
    pub finish_reason: String,
}

/// Streaming chunk for /v1/chat/completions.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatCompletionChunkChoice>,
}

/// A choice in a streaming chunk.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunkChoice {
    pub index: u32,
    pub delta: ChatCompletionDelta,
    pub finish_reason: Option<String>,
}

/// Delta content in a streaming chunk.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

/// A tool call delta for streaming.
#[derive(Debug, Clone, Serialize)]
pub struct ToolCallDelta {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<ToolCallFunctionDelta>,
}

/// Function delta in a streaming tool call.
#[derive(Debug, Clone, Serialize)]
pub struct ToolCallFunctionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// POST /v1/completions request body.
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<StopSequence>,
}

/// POST /v1/completions response (non-streaming).
#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: CompletionUsage,
}

/// A choice in a completions response.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: String,
}

/// Streaming chunk for /v1/completions.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChunkChoice>,
}

/// A choice in a completions streaming chunk.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionChunkChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: Option<String>,
}

/// Token usage statistics.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// GET /v1/models response.
#[derive(Debug, Clone, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

/// A model in the models list.
#[derive(Debug, Clone, Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub owned_by: String,
}

/// POST /v1/embeddings request body.
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
    #[serde(default)]
    #[allow(dead_code)] // Required for API deserialization compatibility
    pub encoding_format: Option<String>,
}

/// Embedding input: single string or array of strings.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

/// POST /v1/embeddings response.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

/// A single embedding result.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingObject {
    pub object: &'static str,
    pub embedding: Vec<f32>,
    pub index: u32,
}

/// Usage for embeddings.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    /// Deserialize a chat completion request from JSON with a single user message
    /// and one extra field merged in (e.g., `"max_tokens": 0`).
    fn chat_request_with(extra_field: &str) -> ChatCompletionRequest {
        let json = format!(
            r#"{{"model": "m", "messages": [{{"role": "user", "content": "hi"}}], {extra_field}}}"#,
        );
        serde_json::from_str(&json).unwrap()
    }

    fn make_chat_chunk(
        id: &str,
        delta: ChatCompletionDelta,
        finish_reason: Option<String>,
    ) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: id.to_owned(),
            object: "chat.completion.chunk",
            created: 1_700_000_000,
            model: "test".to_owned(),
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta,
                finish_reason,
            }],
        }
    }

    fn make_empty_delta_chunk(id: &str, finish_reason: Option<String>) -> ChatCompletionChunk {
        make_chat_chunk(
            id,
            ChatCompletionDelta {
                role: None,
                content: None,
                tool_calls: None,
            },
            finish_reason,
        )
    }

    fn make_usage(prompt: u32, completion: u32) -> CompletionUsage {
        CompletionUsage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
        }
    }

    #[test]
    fn test_chat_request_minimal_deserialization() {
        let json = r#"{"model": "test", "messages": [{"role": "user", "content": "hi"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "test");
        assert_eq!(req.messages.len(), 1);
        assert!(req.stream.is_none());
        assert!(req.max_tokens.is_none());
    }

    #[test]
    fn test_chat_request_full_deserialization() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": true,
            "stop": ["\n", "END"]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.max_tokens, Some(100));
        assert!(req.stream == Some(true));
    }

    #[test]
    fn test_stop_sequence_single() {
        let json = r#"{"model": "m", "messages": [], "stop": "END"}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req.stop, Some(StopSequence::Single(_))));
    }

    #[test]
    fn test_chat_response_serialization() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-123".to_owned(),
            object: "chat.completion",
            created: 1_234_567_890,
            model: "test".to_owned(),
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatCompletionMessage {
                    role: "assistant".to_owned(),
                    content: Some("Hello!".to_owned()),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: "stop".to_owned(),
            }],
            usage: make_usage(5, 1),
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("chat.completion"));
        assert!(json.contains("Hello!"));
    }

    #[test]
    fn test_completion_request_deserialization() {
        let json = r#"{"model": "test", "prompt": "Once upon a time"}"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.prompt, "Once upon a time");
    }

    #[test]
    fn test_model_list_serialization() {
        let list = ModelList {
            object: "list",
            data: vec![ModelObject {
                id: "test-model".to_owned(),
                object: "model",
                created: 1_234_567_890,
                owned_by: "local".to_owned(),
            }],
        };
        let json = serde_json::to_string(&list).unwrap();
        assert!(json.contains("test-model"));
    }

    #[test]
    fn test_response_format_json_mode() {
        let req = chat_request_with(r#""response_format": {"type": "json_object"}"#);
        let fmt = req.response_format.unwrap();
        assert_eq!(fmt.r#type, "json_object");
        assert!(fmt.json_schema.is_none());
    }

    #[test]
    fn test_response_format_json_schema() {
        let req = chat_request_with(
            r#""response_format": {"type": "json_schema", "json_schema": {"type": "object", "properties": {"name": {"type": "string"}}}}"#,
        );
        let fmt = req.response_format.unwrap();
        assert_eq!(fmt.r#type, "json_schema");
        assert!(fmt.json_schema.is_some());
    }

    #[test]
    fn test_tool_call_serialization() {
        let msg = ChatCompletionMessage {
            role: "assistant".to_owned(),
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_123".to_owned(),
                r#type: "function".to_owned(),
                function: ToolCallFunction {
                    name: "get_weather".to_owned(),
                    arguments: r#"{"city":"London"}"#.to_owned(),
                },
            }]),
            tool_call_id: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("call_123"));
        assert!(json.contains("get_weather"));
        assert!(!json.contains("\"content\""));
    }

    #[test]
    fn test_tool_call_message_deserialization() {
        let json = r#"{
            "role": "tool",
            "content": "72 degrees",
            "tool_call_id": "call_123"
        }"#;
        let msg: ChatCompletionMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "tool");
        assert_eq!(msg.tool_call_id, Some("call_123".to_owned()));
    }

    #[test]
    fn test_embedding_request_single() {
        let json = r#"{"model": "test", "input": "Hello world"}"#;
        let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req.input, EmbeddingInput::Single(_)));
    }

    #[test]
    fn test_embedding_request_multiple() {
        let json = r#"{"model": "test", "input": ["Hello", "World"]}"#;
        let req: EmbeddingRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req.input, EmbeddingInput::Multiple(_)));
    }

    #[test]
    fn test_embedding_response_serialization() {
        let resp = EmbeddingResponse {
            object: "list",
            data: vec![EmbeddingObject {
                object: "embedding",
                embedding: vec![0.1, 0.2, 0.3],
                index: 0,
            }],
            model: "test".to_owned(),
            usage: EmbeddingUsage {
                prompt_tokens: 3,
                total_tokens: 3,
            },
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("embedding"));
        assert!(json.contains("0.1"));
    }

    #[test]
    fn test_stop_sequence_into_vec() {
        let single = StopSequence::Single("END".to_owned());
        assert_eq!(single.into_vec(), vec!["END"]);

        let multiple = StopSequence::Multiple(vec!["a".to_owned(), "b".to_owned(), "c".to_owned()]);
        assert_eq!(multiple.into_vec(), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_stop_sequence_extract() {
        assert!(StopSequence::extract(None).is_empty());

        let single = StopSequence::extract(Some(StopSequence::Single("x".to_owned())));
        assert_eq!(single, vec!["x"]);
    }

    #[test]
    fn test_stop_sequence_multiple_deserialization() {
        let json = r#"{"model": "m", "messages": [], "stop": ["a", "b"]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(matches!(req.stop, Some(StopSequence::Multiple(_))));
    }

    #[test]
    fn test_chat_completion_chunk_serialization() {
        let chunk = make_chat_chunk(
            "chatcmpl-123",
            ChatCompletionDelta {
                role: Some("assistant".to_owned()),
                content: Some("Hi".to_owned()),
                tool_calls: None,
            },
            None,
        );
        let json = serde_json::to_string(&chunk).unwrap();
        assert!(json.contains("chat.completion.chunk"));
    }

    #[test]
    fn test_completion_chunk_serialization() {
        let chunk = CompletionChunk {
            id: "cmpl-123".to_owned(),
            object: "text_completion",
            created: 1_234_567_890,
            model: "test".to_owned(),
            choices: vec![CompletionChunkChoice {
                index: 0,
                text: "hello".to_owned(),
                finish_reason: None,
            }],
        };
        let json = serde_json::to_string(&chunk).unwrap();
        assert!(json.contains("text_completion"));
    }

    #[test]
    fn test_chat_completion_delta_skips_none_fields() {
        let delta = ChatCompletionDelta {
            role: None,
            content: None,
            tool_calls: None,
        };
        let json = serde_json::to_string(&delta).unwrap();
        assert_eq!(json, "{}");
    }

    #[test]
    fn test_chat_request_with_max_tokens_zero() {
        let req = chat_request_with(r#""max_tokens": 0"#);
        assert_eq!(req.max_tokens, Some(0));
    }

    #[test]
    fn test_chat_request_with_temperature_zero() {
        let req = chat_request_with(r#""temperature": 0.0"#);
        assert!((req.temperature.unwrap()).abs() < f32::EPSILON);
    }

    #[test]
    fn test_chat_request_with_top_p_zero() {
        let req = chat_request_with(r#""top_p": 0.0"#);
        assert!((req.top_p.unwrap()).abs() < f32::EPSILON);
    }

    #[test]
    fn test_chat_request_with_top_p_one() {
        let req = chat_request_with(r#""top_p": 1.0"#);
        assert!((req.top_p.unwrap() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_response_format_with_json_schema_type() {
        let json = r#"{
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {"type": "object", "properties": {"answer": {"type": "string"}}}
                }
            }
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        let fmt = req.response_format.unwrap();
        assert_eq!(fmt.r#type, "json_schema");
        assert!(fmt.json_schema.is_some());
    }

    #[test]
    fn test_embedding_input_single_vs_array() {
        let single_json = r#"{"model": "m", "input": "hello"}"#;
        let single: EmbeddingRequest = serde_json::from_str(single_json).unwrap();
        assert!(matches!(single.input, EmbeddingInput::Single(_)));

        let array_json = r#"{"model": "m", "input": ["hello", "world"]}"#;
        let array: EmbeddingRequest = serde_json::from_str(array_json).unwrap();
        assert!(matches!(array.input, EmbeddingInput::Multiple(_)));
    }

    #[test]
    fn test_completion_request_all_optional_fields_missing() {
        let json = r#"{"model": "m", "prompt": "test"}"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.max_tokens.is_none());
        assert!(req.temperature.is_none());
        assert!(req.top_p.is_none());
        assert!(req.stream.is_none());
        assert!(req.stop.is_none());
    }

    #[test]
    fn test_completion_request_all_optional_fields_present() {
        let json = r#"{
            "model": "m",
            "prompt": "test",
            "max_tokens": 512,
            "temperature": 0.5,
            "top_p": 0.8,
            "stream": false,
            "stop": ["END"]
        }"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.max_tokens, Some(512));
        assert!((req.temperature.unwrap() - 0.5).abs() < f32::EPSILON);
        assert!((req.top_p.unwrap() - 0.8).abs() < f32::EPSILON);
        assert_eq!(req.stream, Some(false));
        assert!(req.stop.is_some());
    }

    #[test]
    fn test_extra_unknown_fields_silently_ignored() {
        let json = r#"{
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "unknown_field_xyz": 42,
            "another_unknown": "hello"
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "m");
    }

    #[test]
    fn test_chat_message_with_null_content() {
        let json = r#"{"role": "assistant", "content": null}"#;
        let msg: ChatCompletionMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "assistant");
        assert!(msg.content.is_none());
    }

    #[test]
    fn test_chat_message_without_content_field() {
        let json = r#"{"role": "assistant"}"#;
        let msg: ChatCompletionMessage = serde_json::from_str(json).unwrap();
        assert!(msg.content.is_none());
    }

    #[test]
    fn test_completion_request_with_stop_sequences_as_array() {
        let json = r#"{"model": "m", "prompt": "test", "stop": ["END", "\n", "DONE"]}"#;
        let req: CompletionRequest = serde_json::from_str(json).unwrap();
        match req.stop.unwrap() {
            StopSequence::Multiple(v) => assert_eq!(v.len(), 3),
            StopSequence::Single(_) => panic!("expected Multiple variant"),
        }
    }

    #[test]
    fn test_chat_response_with_tool_calls_serialization() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-tools".to_owned(),
            object: "chat.completion",
            created: 1_700_000_000,
            model: "test".to_owned(),
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatCompletionMessage {
                    role: "assistant".to_owned(),
                    content: None,
                    tool_calls: Some(vec![ToolCall {
                        id: "call_1".to_owned(),
                        r#type: "function".to_owned(),
                        function: ToolCallFunction {
                            name: "search".to_owned(),
                            arguments: r#"{"query":"rust"}"#.to_owned(),
                        },
                    }]),
                    tool_call_id: None,
                },
                finish_reason: "tool_calls".to_owned(),
            }],
            usage: make_usage(10, 5),
        };
        let json_val: serde_json::Value = serde_json::to_value(&resp).unwrap();
        assert_eq!(json_val["choices"][0]["finish_reason"], "tool_calls");
        assert!(json_val["choices"][0]["message"].get("content").is_none());
        assert!(json_val["choices"][0]["message"]["tool_calls"].is_array());
    }

    #[test]
    fn test_streaming_chunk_with_finish_reason() {
        let chunk = make_empty_delta_chunk("chatcmpl-fin", Some("stop".to_owned()));
        let json_val: serde_json::Value = serde_json::to_value(&chunk).unwrap();
        assert_eq!(json_val["choices"][0]["finish_reason"], "stop");
    }
}
