use crate::types::anthropic::{AnthropicContent, AnthropicMessage};

/// Map an OpenAI finish_reason to an Anthropic stop_reason.
pub fn openai_finish_to_anthropic_stop(finish_reason: &str) -> String {
    match finish_reason {
        "stop" => "end_turn".to_owned(),
        "length" => "max_tokens".to_owned(),
        "tool_calls" => "tool_use".to_owned(),
        other => other.to_owned(),
    }
}

/// Convert Anthropic messages to the engine's ChatMessage format.
pub fn anthropic_messages_to_engine(
    messages: &[AnthropicMessage],
    system: Option<&str>,
) -> Vec<mlx_engine::chat_template::ChatMessage> {
    let mut result = Vec::new();

    if let Some(sys) = system {
        result.push(mlx_engine::chat_template::ChatMessage {
            role: "system".to_owned(),
            content: sys.to_owned(),
            tool_calls: None,
        });
    }

    for msg in messages {
        let content = match &msg.content {
            AnthropicContent::Text(s) => s.clone(),
            AnthropicContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    crate::types::anthropic::ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        };

        result.push(mlx_engine::chat_template::ChatMessage {
            role: msg.role.clone(),
            content,
            tool_calls: None,
        });
    }

    result
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_finish_reason_mapping() {
        assert_eq!(openai_finish_to_anthropic_stop("stop"), "end_turn");
        assert_eq!(openai_finish_to_anthropic_stop("length"), "max_tokens");
        assert_eq!(openai_finish_to_anthropic_stop("tool_calls"), "tool_use");
        assert_eq!(openai_finish_to_anthropic_stop("other"), "other");
    }

    #[test]
    fn test_anthropic_messages_to_engine_with_system() {
        let messages = vec![AnthropicMessage {
            role: "user".to_owned(),
            content: AnthropicContent::Text("Hello".to_owned()),
        }];

        let result = anthropic_messages_to_engine(&messages, Some("Be helpful"));
        assert_eq!(result.len(), 2);
        assert_eq!(result.first().map(|m| m.role.as_str()), Some("system"));
        assert_eq!(
            result.first().map(|m| m.content.as_str()),
            Some("Be helpful")
        );
    }

    #[test]
    fn test_anthropic_messages_to_engine_without_system() {
        let messages = vec![AnthropicMessage {
            role: "user".to_owned(),
            content: AnthropicContent::Text("Hello".to_owned()),
        }];
        let result = anthropic_messages_to_engine(&messages, None);
        assert_eq!(result.len(), 1);
        assert_eq!(result.first().map(|m| m.role.as_str()), Some("user"));
    }

    #[test]
    fn test_anthropic_messages_to_engine_content_blocks() {
        use crate::types::anthropic::ContentBlock;

        let messages = vec![AnthropicMessage {
            role: "user".to_owned(),
            content: AnthropicContent::Blocks(vec![
                ContentBlock::Text {
                    text: "Hello ".to_owned(),
                },
                ContentBlock::Text {
                    text: "World".to_owned(),
                },
            ]),
        }];
        let result = anthropic_messages_to_engine(&messages, None);
        assert_eq!(result.len(), 1);
        assert_eq!(
            result.first().map(|m| m.content.as_str()),
            Some("Hello World")
        );
    }

    #[test]
    fn test_anthropic_messages_to_engine_mixed_blocks_filters_non_text() {
        use crate::types::anthropic::ContentBlock;

        let messages = vec![AnthropicMessage {
            role: "user".to_owned(),
            content: AnthropicContent::Blocks(vec![
                ContentBlock::Text {
                    text: "Hello".to_owned(),
                },
                ContentBlock::ToolUse {
                    id: "tu_1".to_owned(),
                    name: "get_weather".to_owned(),
                    input: serde_json::json!({}),
                },
                ContentBlock::ToolResult {
                    tool_use_id: "tu_1".to_owned(),
                    content: "72 degrees".to_owned(),
                },
            ]),
        }];
        let result = anthropic_messages_to_engine(&messages, None);
        assert_eq!(result.len(), 1);
        assert_eq!(result.first().map(|m| m.content.as_str()), Some("Hello"));
    }

    #[test]
    fn test_anthropic_messages_to_engine_empty_messages() {
        let result = anthropic_messages_to_engine(&[], None);
        assert!(result.is_empty());
    }

    #[test]
    fn test_anthropic_messages_to_engine_multiple_messages() {
        let messages = vec![
            AnthropicMessage {
                role: "user".to_owned(),
                content: AnthropicContent::Text("First".to_owned()),
            },
            AnthropicMessage {
                role: "assistant".to_owned(),
                content: AnthropicContent::Text("Second".to_owned()),
            },
            AnthropicMessage {
                role: "user".to_owned(),
                content: AnthropicContent::Text("Third".to_owned()),
            },
        ];
        let result = anthropic_messages_to_engine(&messages, None);
        assert_eq!(result.len(), 3);
        assert_eq!(result.first().map(|m| m.content.as_str()), Some("First"));
        assert_eq!(result.get(1).map(|m| m.content.as_str()), Some("Second"));
        assert_eq!(result.get(2).map(|m| m.content.as_str()), Some("Third"));
    }
}
