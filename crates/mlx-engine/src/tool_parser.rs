//! Parse tool calls from model-generated text.
//!
//! Qwen models emit tool calls in a specific XML-like format:
//! ```text
//! <tool_call>
//! {"name": "function_name", "arguments": {"arg1": "value1"}}
//! </tool_call>
//! ```
//!
//! This module extracts those structured tool calls from the raw text.

/// A parsed tool call extracted from model output.
#[derive(Debug, Clone)]
pub struct ParsedToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Result of parsing model output for tool calls.
#[derive(Debug, Clone)]
pub struct ToolParseResult {
    /// Text content before/outside any tool calls.
    pub text: String,
    /// Extracted tool calls (empty if none found).
    pub tool_calls: Vec<ParsedToolCall>,
}

const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";

/// Parse model output text for Qwen-format tool calls.
///
/// Returns the non-tool-call text and any extracted tool calls.
pub fn parse_tool_calls(text: &str) -> ToolParseResult {
    let mut result_text = String::new();
    let mut tool_calls = Vec::new();
    let mut remaining = text;

    loop {
        match remaining.find(TOOL_CALL_OPEN) {
            Some(start_pos) => {
                // Collect text before the tool call tag
                result_text.push_str(remaining.get(..start_pos).unwrap_or_default());

                let after_open = remaining
                    .get(start_pos + TOOL_CALL_OPEN.len()..)
                    .unwrap_or_default();

                match after_open.find(TOOL_CALL_CLOSE) {
                    Some(end_pos) => {
                        let raw_block = after_open.get(..end_pos).unwrap_or_default();
                        let call_content = raw_block.trim();

                        if let Some(parsed) = try_parse_tool_call(call_content) {
                            tool_calls.push(parsed);
                        } else {
                            result_text.push_str(TOOL_CALL_OPEN);
                            result_text.push_str(raw_block);
                            result_text.push_str(TOOL_CALL_CLOSE);
                        }

                        remaining = after_open
                            .get(end_pos + TOOL_CALL_CLOSE.len()..)
                            .unwrap_or_default();
                    }
                    None => {
                        // Unclosed tag -- treat rest as text
                        result_text.push_str(remaining.get(start_pos..).unwrap_or_default());
                        break;
                    }
                }
            }
            None => {
                result_text.push_str(remaining);
                break;
            }
        }
    }

    ToolParseResult {
        text: result_text.trim().to_owned(),
        tool_calls,
    }
}

/// Try to parse a single tool call JSON block.
fn try_parse_tool_call(content: &str) -> Option<ParsedToolCall> {
    let value: serde_json::Value = serde_json::from_str(content).ok()?;
    let obj = value.as_object()?;

    let name = obj.get("name").and_then(|v| v.as_str())?.to_owned();

    let arguments = obj
        .get("arguments")
        .cloned()
        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

    Some(ParsedToolCall { name, arguments })
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_no_tool_calls() {
        let result = parse_tool_calls("Hello, how can I help you?");
        assert_eq!(result.text, "Hello, how can I help you?");
        assert!(result.tool_calls.is_empty());
    }

    #[test]
    fn test_single_tool_call() {
        let input = r#"<tool_call>
{"name": "get_weather", "arguments": {"city": "London"}}
</tool_call>"#;
        let result = parse_tool_calls(input);
        assert!(result.text.is_empty());
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls.first().unwrap().name, "get_weather");
    }

    #[test]
    fn test_tool_call_with_surrounding_text() {
        let input = r#"Let me check the weather for you.
<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris"}}
</tool_call>
I've requested the weather."#;
        let result = parse_tool_calls(input);
        assert!(result.text.contains("Let me check"));
        assert!(result.text.contains("I've requested"));
        assert_eq!(result.tool_calls.len(), 1);
    }

    #[test]
    fn test_multiple_tool_calls() {
        let input = r#"<tool_call>
{"name": "search", "arguments": {"query": "rust"}}
</tool_call>
<tool_call>
{"name": "calculate", "arguments": {"expression": "2+2"}}
</tool_call>"#;
        let result = parse_tool_calls(input);
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls.first().unwrap().name, "search");
        assert_eq!(result.tool_calls.get(1).unwrap().name, "calculate");
    }

    #[test]
    fn test_invalid_json_in_tool_call() {
        let input = "<tool_call>\nnot valid json\n</tool_call>";
        let result = parse_tool_calls(input);
        assert!(result.tool_calls.is_empty());
        assert!(result.text.contains("not valid json"));
    }

    #[test]
    fn test_unclosed_tool_call_tag() {
        let input = "Text before <tool_call>\n{\"name\": \"test\"}";
        let result = parse_tool_calls(input);
        assert!(result.tool_calls.is_empty());
        assert!(result.text.contains("<tool_call>"));
    }

    #[test]
    fn test_tool_call_missing_arguments() {
        let input = r#"<tool_call>
{"name": "no_args_tool"}
</tool_call>"#;
        let result = parse_tool_calls(input);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls.first().unwrap().name, "no_args_tool");
        assert!(result.tool_calls.first().unwrap().arguments.is_object());
    }

    #[test]
    fn test_tool_call_missing_name() {
        let input = r#"<tool_call>
{"arguments": {"key": "value"}}
</tool_call>"#;
        let result = parse_tool_calls(input);
        assert!(result.tool_calls.is_empty());
    }

    #[test]
    fn test_empty_text() {
        let result = parse_tool_calls("");
        assert!(result.text.is_empty());
        assert!(result.tool_calls.is_empty());
    }

    #[test]
    fn test_invalid_json_preserves_original_tags() {
        let input = "<tool_call>\nnot valid json\n</tool_call>";
        let result = parse_tool_calls(input);
        assert!(result.tool_calls.is_empty());
        assert!(result.text.contains("<tool_call>"));
        assert!(result.text.contains("</tool_call>"));
        assert!(result.text.contains("not valid json"));
    }

    #[test]
    fn test_mix_of_valid_and_invalid_tool_calls() {
        let input = r#"<tool_call>
{"name": "good_tool", "arguments": {"key": "value"}}
</tool_call>
<tool_call>
this is not json
</tool_call>
<tool_call>
{"name": "another_good", "arguments": {}}
</tool_call>"#;
        let result = parse_tool_calls(input);

        // Two valid tool calls extracted
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls.first().unwrap().name, "good_tool");
        assert_eq!(result.tool_calls.get(1).unwrap().name, "another_good");

        // Invalid one preserved as raw text with tags
        assert!(result.text.contains("<tool_call>"));
        assert!(result.text.contains("this is not json"));
        assert!(result.text.contains("</tool_call>"));
    }

    #[test]
    fn test_valid_json_but_missing_name_preserved_as_raw() {
        // Valid JSON object but no "name" field -- should be treated as malformed
        let input = r#"<tool_call>
{"arguments": {"key": "value"}, "description": "no name field"}
</tool_call>"#;
        let result = parse_tool_calls(input);
        assert!(result.tool_calls.is_empty());
        assert!(result.text.contains("<tool_call>"));
        assert!(result.text.contains("</tool_call>"));
        assert!(result.text.contains("no name field"));
    }

    #[test]
    fn test_valid_json_array_not_object_preserved_as_raw() {
        // Valid JSON but an array, not an object
        let input = "<tool_call>\n[1, 2, 3]\n</tool_call>";
        let result = parse_tool_calls(input);
        assert!(result.tool_calls.is_empty());
        assert!(result.text.contains("<tool_call>"));
        assert!(result.text.contains("[1, 2, 3]"));
        assert!(result.text.contains("</tool_call>"));
    }

    #[test]
    fn test_valid_json_name_is_not_string_preserved_as_raw() {
        // "name" present but is a number, not a string
        let input = r#"<tool_call>
{"name": 42, "arguments": {}}
</tool_call>"#;
        let result = parse_tool_calls(input);
        assert!(result.tool_calls.is_empty());
        assert!(result.text.contains("<tool_call>"));
        assert!(result.text.contains("</tool_call>"));
    }

    #[test]
    fn test_text_between_multiple_tool_calls() {
        let input = r#"Before first.
<tool_call>
{"name": "tool_a", "arguments": {}}
</tool_call>
Middle text.
<tool_call>
{"name": "tool_b", "arguments": {}}
</tool_call>
After last."#;
        let result = parse_tool_calls(input);
        assert_eq!(result.tool_calls.len(), 2);
        assert!(result.text.contains("Before first."));
        assert!(result.text.contains("Middle text."));
        assert!(result.text.contains("After last."));
    }

    #[test]
    fn test_nested_tool_call_tags() {
        // A <tool_call> tag nested inside another -- the inner one becomes
        // part of the content between the first open and first close.
        let input = r#"<tool_call>
<tool_call>
{"name": "inner", "arguments": {}}
</tool_call>
</tool_call>"#;
        let result = parse_tool_calls(input);
        // The parser finds the first <tool_call>, then looks for first </tool_call>.
        // Content between them: "\n<tool_call>\n{\"name\": \"inner\", \"arguments\": {}}\n"
        // This is not valid JSON (starts with <tool_call>), so it's preserved as raw text.
        // Then the remaining "</tool_call>" is just text.
        // The outer close tag becomes trailing text.
        assert!(result.tool_calls.is_empty() || result.tool_calls.len() <= 1);
    }

    #[test]
    fn test_arguments_as_json_array() {
        let input = r#"<tool_call>
{"name": "batch_op", "arguments": [1, 2, 3]}
</tool_call>"#;
        let result = parse_tool_calls(input);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls.first().unwrap().name, "batch_op");
        assert!(result.tool_calls.first().unwrap().arguments.is_array());
        assert_eq!(
            result.tool_calls.first().unwrap().arguments,
            serde_json::json!([1, 2, 3])
        );
    }

    #[test]
    fn test_arguments_with_special_chars_and_unicode() {
        let input = r#"<tool_call>
{"name": "translate", "arguments": {"text": "Caf\u00e9 \"quotes\" \\backslash", "emoji": "\ud83d\ude00"}}
</tool_call>"#;
        let result = parse_tool_calls(input);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls.first().unwrap().name, "translate");
        let args = &result.tool_calls.first().unwrap().arguments;
        let text_val = args.get("text").unwrap().as_str().unwrap();
        assert!(text_val.contains("Caf\u{00e9}"));
        assert!(text_val.contains("\"quotes\""));
        assert!(text_val.contains("\\backslash"));
    }

    #[test]
    fn test_whitespace_only_content_between_tags() {
        let input = "<tool_call>\n   \n  \t  \n</tool_call>";
        let result = parse_tool_calls(input);
        // Whitespace-only content is not valid JSON, so no tool calls extracted
        assert!(result.tool_calls.is_empty());
        // The raw tags and whitespace are preserved in text
        assert!(result.text.contains("<tool_call>"));
    }
}
