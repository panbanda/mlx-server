use serde::{Deserialize, Serialize};

/// POST /v1/messages request body (Anthropic Messages API).
#[derive(Debug, Clone, Deserialize)]
pub struct CreateMessageRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: u32,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
}

/// A message in the Anthropic format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicContent,
}

/// Content can be a plain string or an array of content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// A content block in the Anthropic format.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

/// POST /v1/messages response (non-streaming).
#[derive(Debug, Clone, Serialize)]
pub struct CreateMessageResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub message_type: &'static str,
    pub role: &'static str,
    pub content: Vec<ContentBlockResponse>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub usage: AnthropicUsage,
}

/// A content block in the response.
#[derive(Debug, Clone, Serialize)]
pub struct ContentBlockResponse {
    #[serde(rename = "type")]
    pub block_type: &'static str,
    pub text: String,
}

/// Anthropic usage stats.
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// POST /v1/messages/count_tokens request.
#[derive(Debug, Clone, Deserialize)]
pub struct CountTokensRequest {
    #[allow(dead_code)] // Required for API deserialization compatibility
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
}

/// POST /v1/messages/count_tokens response.
#[derive(Debug, Clone, Serialize)]
pub struct CountTokensResponse {
    pub input_tokens: u32,
}

// --- Streaming event types ---

/// Server-sent event wrapper for Anthropic streaming.
#[derive(Debug, Clone, Serialize)]
pub struct MessageStartEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub message: MessageStartPayload,
}

#[derive(Debug, Clone, Serialize)]
pub struct MessageStartPayload {
    pub id: String,
    #[serde(rename = "type")]
    pub message_type: &'static str,
    pub role: &'static str,
    pub content: Vec<serde_json::Value>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContentBlockStartEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub index: u32,
    pub content_block: ContentBlockStartPayload,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContentBlockStartPayload {
    #[serde(rename = "type")]
    pub block_type: &'static str,
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContentBlockDeltaEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub index: u32,
    pub delta: TextDelta,
}

#[derive(Debug, Clone, Serialize)]
pub struct TextDelta {
    #[serde(rename = "type")]
    pub delta_type: &'static str,
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContentBlockStopEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub index: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct MessageDeltaEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
    pub delta: MessageDelta,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Clone, Serialize)]
pub struct MessageDelta {
    pub stop_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MessageStopEvent {
    #[serde(rename = "type")]
    pub event_type: &'static str,
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used, clippy::indexing_slicing)]
mod tests {
    use super::*;

    #[test]
    fn test_anthropic_request_deserialization() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100
        }"#;
        let req: CreateMessageRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "test");
        assert_eq!(req.max_tokens, 100);
        assert_eq!(req.messages.len(), 1);
    }

    #[test]
    fn test_anthropic_content_text() {
        let json = r#"{"role": "user", "content": "Hello"}"#;
        let msg: AnthropicMessage = serde_json::from_str(json).unwrap();
        assert!(matches!(msg.content, AnthropicContent::Text(ref s) if s == "Hello"));
    }

    #[test]
    fn test_anthropic_content_blocks() {
        let json = r#"{"role": "user", "content": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]}"#;
        let msg: AnthropicMessage = serde_json::from_str(json).unwrap();
        assert!(matches!(msg.content, AnthropicContent::Blocks(ref blocks) if blocks.len() == 2));
    }

    #[test]
    fn test_anthropic_response_serialization() {
        let resp = CreateMessageResponse {
            id: "msg_123".to_owned(),
            message_type: "message",
            role: "assistant",
            content: vec![ContentBlockResponse {
                block_type: "text",
                text: "Hello!".to_owned(),
            }],
            model: "test".to_owned(),
            stop_reason: Some("end_turn".to_owned()),
            usage: AnthropicUsage {
                input_tokens: 5,
                output_tokens: 1,
            },
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"type\":\"message\""));
        assert!(json.contains("end_turn"));
    }

    #[test]
    fn test_count_tokens_request() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "Hello"}],
            "system": "You are helpful."
        }"#;
        let req: CountTokensRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.system, Some("You are helpful.".to_owned()));
    }

    #[test]
    fn test_message_start_event_serialization() {
        let event = MessageStartEvent {
            event_type: "message_start",
            message: MessageStartPayload {
                id: "msg_123".to_owned(),
                message_type: "message",
                role: "assistant",
                content: vec![],
                model: "test".to_owned(),
                stop_reason: None,
                usage: AnthropicUsage {
                    input_tokens: 5,
                    output_tokens: 0,
                },
            },
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("message_start"));
    }

    #[test]
    fn test_anthropic_content_tool_use_block() {
        let json = r#"{"role": "assistant", "content": [{"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "London"}}]}"#;
        let msg: AnthropicMessage = serde_json::from_str(json).unwrap();
        if let AnthropicContent::Blocks(blocks) = &msg.content {
            assert_eq!(blocks.len(), 1);
            assert!(
                matches!(&blocks[0], ContentBlock::ToolUse { name, .. } if name == "get_weather")
            );
        } else {
            panic!("Expected Blocks variant");
        }
    }

    #[test]
    fn test_anthropic_content_tool_result_block() {
        let json = r#"{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "72 degrees"}]}"#;
        let msg: AnthropicMessage = serde_json::from_str(json).unwrap();
        if let AnthropicContent::Blocks(blocks) = &msg.content {
            assert_eq!(blocks.len(), 1);
            assert!(
                matches!(&blocks[0], ContentBlock::ToolResult { tool_use_id, content } if tool_use_id == "tu_1" && content == "72 degrees")
            );
        } else {
            panic!("Expected Blocks variant");
        }
    }

    #[test]
    fn test_anthropic_request_with_stop_sequences() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "stop_sequences": ["END", "STOP"]
        }"#;
        let req: CreateMessageRequest = serde_json::from_str(json).unwrap();
        let stops = req.stop_sequences.unwrap();
        assert_eq!(stops.len(), 2);
        assert_eq!(stops[0], "END");
        assert_eq!(stops[1], "STOP");
    }

    #[test]
    fn test_create_message_request_max_tokens_zero() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 0
        }"#;
        let req: CreateMessageRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.max_tokens, 0);
    }

    #[test]
    fn test_create_message_request_temperature_zero() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "temperature": 0.0
        }"#;
        let req: CreateMessageRequest = serde_json::from_str(json).unwrap();
        assert!((req.temperature.unwrap()).abs() < f32::EPSILON);
    }

    #[test]
    fn test_anthropic_content_text_deserialization() {
        let json = r#"{"role": "user", "content": "simple text"}"#;
        let msg: AnthropicMessage = serde_json::from_str(json).unwrap();
        match &msg.content {
            AnthropicContent::Text(s) => assert_eq!(s, "simple text"),
            AnthropicContent::Blocks(_) => panic!("expected Text variant"),
        }
    }

    #[test]
    fn test_anthropic_content_blocks_mixed_types() {
        let json = r#"{
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check"},
                {"type": "tool_use", "id": "tu_1", "name": "calculator", "input": {"expr": "2+2"}},
                {"type": "text", "text": "The answer is 4"}
            ]
        }"#;
        let msg: AnthropicMessage = serde_json::from_str(json).unwrap();
        match &msg.content {
            AnthropicContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 3);
                assert!(
                    matches!(&blocks[0], ContentBlock::Text { text } if text == "Let me check")
                );
                assert!(
                    matches!(&blocks[1], ContentBlock::ToolUse { name, .. } if name == "calculator")
                );
                assert!(
                    matches!(&blocks[2], ContentBlock::Text { text } if text == "The answer is 4")
                );
            }
            AnthropicContent::Text(_) => panic!("expected Blocks variant"),
        }
    }

    #[test]
    fn test_tool_use_with_complex_json_input() {
        let json = r#"{
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "tu_complex",
                "name": "database_query",
                "input": {
                    "query": "SELECT * FROM users",
                    "params": [1, "hello", null, true],
                    "nested": {"key": {"deep": [1,2,3]}}
                }
            }]
        }"#;
        let msg: AnthropicMessage = serde_json::from_str(json).unwrap();
        if let AnthropicContent::Blocks(blocks) = &msg.content {
            if let ContentBlock::ToolUse { input, .. } = &blocks[0] {
                assert_eq!(input["query"], "SELECT * FROM users");
                assert!(input["params"].is_array());
                assert!(input["nested"]["key"]["deep"].is_array());
            } else {
                panic!("expected ToolUse block");
            }
        } else {
            panic!("expected Blocks variant");
        }
    }

    #[test]
    fn test_tool_result_with_very_long_content() {
        let long_content = "x".repeat(10_000);
        let json = format!(
            r#"{{"role": "user", "content": [{{"type": "tool_result", "tool_use_id": "tu_1", "content": "{long_content}"}}]}}"#,
        );
        let msg: AnthropicMessage = serde_json::from_str(&json).unwrap();
        if let AnthropicContent::Blocks(blocks) = &msg.content {
            if let ContentBlock::ToolResult { content, .. } = &blocks[0] {
                assert_eq!(content.len(), 10_000);
            } else {
                panic!("expected ToolResult block");
            }
        } else {
            panic!("expected Blocks variant");
        }
    }

    #[test]
    fn test_count_tokens_request_empty_messages() {
        let json = r#"{
            "model": "test",
            "messages": []
        }"#;
        let req: CountTokensRequest = serde_json::from_str(json).unwrap();
        assert!(req.messages.is_empty());
        assert!(req.system.is_none());
    }

    #[test]
    fn test_count_tokens_request_with_system_prompt() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "system": "You are a helpful assistant."
        }"#;
        let req: CountTokensRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.system, Some("You are a helpful assistant.".to_owned()));
    }

    #[test]
    fn test_message_start_event_type_field() {
        let event = MessageStartEvent {
            event_type: "message_start",
            message: MessageStartPayload {
                id: "msg_1".to_owned(),
                message_type: "message",
                role: "assistant",
                content: vec![],
                model: "test".to_owned(),
                stop_reason: None,
                usage: AnthropicUsage {
                    input_tokens: 0,
                    output_tokens: 0,
                },
            },
        };
        let json: serde_json::Value = serde_json::to_value(&event).unwrap();
        assert_eq!(json["type"], "message_start");
    }

    #[test]
    fn test_content_block_start_event_type_field() {
        let event = ContentBlockStartEvent {
            event_type: "content_block_start",
            index: 0,
            content_block: ContentBlockStartPayload {
                block_type: "text",
                text: String::new(),
            },
        };
        let json: serde_json::Value = serde_json::to_value(&event).unwrap();
        assert_eq!(json["type"], "content_block_start");
    }

    #[test]
    fn test_content_block_delta_event_type_field() {
        let event = ContentBlockDeltaEvent {
            event_type: "content_block_delta",
            index: 0,
            delta: TextDelta {
                delta_type: "text_delta",
                text: "Hello".to_owned(),
            },
        };
        let json: serde_json::Value = serde_json::to_value(&event).unwrap();
        assert_eq!(json["type"], "content_block_delta");
        assert_eq!(json["delta"]["type"], "text_delta");
    }

    #[test]
    fn test_content_block_stop_event_type_field() {
        let event = ContentBlockStopEvent {
            event_type: "content_block_stop",
            index: 0,
        };
        let json: serde_json::Value = serde_json::to_value(&event).unwrap();
        assert_eq!(json["type"], "content_block_stop");
    }

    #[test]
    fn test_message_delta_event_type_field() {
        let event = MessageDeltaEvent {
            event_type: "message_delta",
            delta: MessageDelta {
                stop_reason: Some("end_turn".to_owned()),
            },
            usage: AnthropicUsage {
                input_tokens: 10,
                output_tokens: 5,
            },
        };
        let json: serde_json::Value = serde_json::to_value(&event).unwrap();
        assert_eq!(json["type"], "message_delta");
    }

    #[test]
    fn test_message_stop_event_type_field() {
        let event = MessageStopEvent {
            event_type: "message_stop",
        };
        let json: serde_json::Value = serde_json::to_value(&event).unwrap();
        assert_eq!(json["type"], "message_stop");
    }

    #[test]
    fn test_anthropic_request_with_tools() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "What is the weather?"}],
            "max_tokens": 100,
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}}
                    }
                }
            ]
        }"#;
        let req: CreateMessageRequest = serde_json::from_str(json).unwrap();
        assert!(req.tools.is_some());
        let tools = req.tools.unwrap();
        assert_eq!(tools.len(), 1);
    }
}
