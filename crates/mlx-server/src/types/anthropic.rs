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
}
