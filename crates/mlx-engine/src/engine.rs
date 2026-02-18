/// Output from a generation request.
#[derive(Debug, Clone)]
pub struct GenerationOutput {
    pub text: String,
    pub finish_reason: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

/// Output from a streaming generation step.
#[derive(Debug, Clone)]
pub struct StreamingOutput {
    pub new_text: String,
    pub finished: bool,
    pub finish_reason: Option<String>,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn generation_output_construction_and_field_access() {
        let output = GenerationOutput {
            text: "Hello world".to_owned(),
            finish_reason: "stop".to_owned(),
            prompt_tokens: 10,
            completion_tokens: 5,
        };
        assert_eq!(output.text, "Hello world");
        assert_eq!(output.finish_reason, "stop");
        assert_eq!(output.prompt_tokens, 10);
        assert_eq!(output.completion_tokens, 5);
    }

    #[test]
    fn generation_output_empty_defaults() {
        let output = GenerationOutput {
            text: String::new(),
            finish_reason: "length".to_owned(),
            prompt_tokens: 0,
            completion_tokens: 0,
        };
        assert!(output.text.is_empty());
        assert_eq!(output.prompt_tokens, 0);
        assert_eq!(output.completion_tokens, 0);
    }

    #[test]
    fn streaming_output_finished_true() {
        let output = StreamingOutput {
            new_text: "done".to_owned(),
            finished: true,
            finish_reason: Some("stop".to_owned()),
            prompt_tokens: 20,
            completion_tokens: 15,
        };
        assert!(output.finished);
        assert_eq!(output.finish_reason.as_deref(), Some("stop"));
        assert_eq!(output.new_text, "done");
    }

    #[test]
    fn streaming_output_finished_false() {
        let output = StreamingOutput {
            new_text: "partial".to_owned(),
            finished: false,
            finish_reason: None,
            prompt_tokens: 20,
            completion_tokens: 3,
        };
        assert!(!output.finished);
        assert!(output.finish_reason.is_none());
    }

    #[test]
    fn streaming_output_empty_text_zero_tokens() {
        let output = StreamingOutput {
            new_text: String::new(),
            finished: true,
            finish_reason: Some("length".to_owned()),
            prompt_tokens: 0,
            completion_tokens: 0,
        };
        assert!(output.new_text.is_empty());
        assert_eq!(output.prompt_tokens, 0);
        assert_eq!(output.completion_tokens, 0);
    }

    #[test]
    fn generation_output_clone() {
        let output = GenerationOutput {
            text: "test".to_owned(),
            finish_reason: "stop".to_owned(),
            prompt_tokens: 5,
            completion_tokens: 3,
        };
        let cloned = output.clone();
        assert_eq!(cloned.text, output.text);
        assert_eq!(cloned.finish_reason, output.finish_reason);
    }

    #[test]
    fn streaming_output_clone() {
        let output = StreamingOutput {
            new_text: "stream".to_owned(),
            finished: false,
            finish_reason: None,
            prompt_tokens: 10,
            completion_tokens: 2,
        };
        let cloned = output.clone();
        assert_eq!(cloned.new_text, output.new_text);
        assert_eq!(cloned.finished, output.finished);
        assert_eq!(cloned.finish_reason, output.finish_reason);
    }

    #[test]
    fn generation_output_debug_format() {
        let output = GenerationOutput {
            text: "hi".to_owned(),
            finish_reason: "stop".to_owned(),
            prompt_tokens: 1,
            completion_tokens: 1,
        };
        let debug_str = format!("{:?}", output);
        assert!(debug_str.contains("GenerationOutput"));
        assert!(debug_str.contains("hi"));
    }

    #[test]
    fn streaming_output_debug_format() {
        let output = StreamingOutput {
            new_text: "token".to_owned(),
            finished: true,
            finish_reason: Some("stop".to_owned()),
            prompt_tokens: 5,
            completion_tokens: 10,
        };
        let debug_str = format!("{:?}", output);
        assert!(debug_str.contains("StreamingOutput"));
        assert!(debug_str.contains("token"));
    }
}
