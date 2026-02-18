use minijinja::{Environment, Value};
use serde::Serialize;

use crate::error::EngineError;

/// A chat message for template rendering.
#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
}

/// Renders chat messages using a Jinja2 template (HuggingFace format).
pub struct ChatTemplateRenderer {
    env: Environment<'static>,
}

impl ChatTemplateRenderer {
    /// Create a renderer from a Jinja2 template string.
    pub fn new(template_source: impl Into<String>) -> Result<Self, EngineError> {
        let mut env = Environment::new();
        env.add_filter("tojson", tojson_filter);
        env.add_template_owned("chat".to_owned(), template_source.into())
            .map_err(|e| EngineError::Template(e.to_string()))?;
        Ok(Self { env })
    }

    /// Load template from a model directory (chat_template.jinja or tokenizer_config.json).
    pub fn from_model_dir(model_dir: &std::path::Path) -> Result<Self, EngineError> {
        // Prefer standalone chat_template.jinja
        let jinja_path = model_dir.join("chat_template.jinja");
        if jinja_path.exists() {
            let template = std::fs::read_to_string(&jinja_path)
                .map_err(|e| EngineError::Template(format!("Failed to read template: {e}")))?;
            return Self::new(&template);
        }

        // Fall back to tokenizer_config.json
        let config_path = model_dir.join("tokenizer_config.json");
        if config_path.exists() {
            let config_str = std::fs::read_to_string(&config_path)
                .map_err(|e| EngineError::Template(format!("Failed to read config: {e}")))?;
            let config: serde_json::Value = serde_json::from_str(&config_str)
                .map_err(|e| EngineError::Template(format!("Invalid JSON: {e}")))?;
            if let Some(template) = config.get("chat_template").and_then(|v| v.as_str()) {
                return Self::new(template);
            }
        }

        Err(EngineError::Template(
            "No chat template found in model directory".to_owned(),
        ))
    }

    /// Apply the chat template to messages, returning the formatted prompt string.
    pub fn apply(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
        add_generation_prompt: bool,
    ) -> Result<String, EngineError> {
        let tmpl = self
            .env
            .get_template("chat")
            .map_err(|e| EngineError::Template(e.to_string()))?;

        let mut context = minijinja::context! {
            messages => messages,
            add_generation_prompt => add_generation_prompt,
        };

        if let Some(tool_list) = tools {
            context = minijinja::context! {
                messages => messages,
                tools => tool_list,
                add_generation_prompt => add_generation_prompt,
            };
        }

        tmpl.render(context)
            .map_err(|e| EngineError::Template(e.to_string()))
    }
}

/// Custom tojson filter for minijinja (used by HF chat templates).
fn tojson_filter(value: Value) -> Result<String, minijinja::Error> {
    let serialized = serde_json::to_string(&value).map_err(|e| {
        minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            "JSON serialization failed",
        )
        .with_source(e)
    })?;
    Ok(serialized)
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_chatml_template() {
        let template = r#"{%- for message in messages %}
<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{%- endfor %}
{%- if add_generation_prompt %}
<|im_start|>assistant
{%- endif %}"#;

        let renderer = ChatTemplateRenderer::new(template).unwrap();
        let messages = vec![
            ChatMessage {
                role: "system".to_owned(),
                content: "You are helpful.".to_owned(),
                tool_calls: None,
            },
            ChatMessage {
                role: "user".to_owned(),
                content: "Hello!".to_owned(),
                tool_calls: None,
            },
        ];

        let result = renderer.apply(&messages, None, true).unwrap();
        assert!(result.contains("<|im_start|>system"));
        assert!(result.contains("You are helpful."));
        assert!(result.contains("<|im_start|>user"));
        assert!(result.contains("Hello!"));
        assert!(result.contains("<|im_start|>assistant"));
    }

    #[test]
    fn test_tojson_filter() {
        let template = r#"{{ value | tojson }}"#;
        let mut env = Environment::new();
        env.add_filter("tojson", tojson_filter);
        env.add_template_owned("test".to_owned(), template.to_owned())
            .unwrap();
        let tmpl = env.get_template("test").unwrap();
        let result = tmpl
            .render(minijinja::context! { value => "hello" })
            .unwrap();
        assert_eq!(result, r#""hello""#);
    }

    #[test]
    fn test_invalid_template_syntax_returns_error() {
        let result = ChatTemplateRenderer::new("{%- invalid syntax %}}}");
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_without_generation_prompt() {
        let template = r#"{%- for message in messages %}
<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{%- endfor %}
{%- if add_generation_prompt %}
<|im_start|>assistant
{%- endif %}"#;

        let renderer = ChatTemplateRenderer::new(template).unwrap();
        let messages = vec![ChatMessage {
            role: "user".to_owned(),
            content: "Hello!".to_owned(),
            tool_calls: None,
        }];

        let result = renderer.apply(&messages, None, false).unwrap();
        assert!(!result.contains("<|im_start|>assistant"));
    }

    #[test]
    fn test_apply_empty_messages() {
        let template = r#"{%- for message in messages %}
<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{%- endfor %}"#;

        let renderer = ChatTemplateRenderer::new(template).unwrap();
        let result = renderer.apply(&[], None, false).unwrap();
        assert!(!result.contains("<|im_start|>"));
    }

    #[test]
    fn test_apply_with_tools() {
        let template = r#"{%- for message in messages %}
{{ message.content }}
{%- endfor %}
{%- if tools %}
TOOLS:{{ tools | length }}
{%- endif %}"#;

        let renderer = ChatTemplateRenderer::new(template).unwrap();
        let messages = vec![ChatMessage {
            role: "user".to_owned(),
            content: "Hi".to_owned(),
            tool_calls: None,
        }];
        let tools = vec![serde_json::json!({"type": "function", "function": {"name": "test"}})];

        let result = renderer.apply(&messages, Some(&tools), false).unwrap();
        assert!(result.contains("TOOLS:1"));
    }

    #[test]
    fn test_from_model_dir_no_template_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let result = ChatTemplateRenderer::from_model_dir(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_from_model_dir_tokenizer_config_no_chat_template_field() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"model_type": "qwen2"}"#,
        )
        .unwrap();
        let result = ChatTemplateRenderer::from_model_dir(dir.path());
        assert!(result.is_err());
    }
}
