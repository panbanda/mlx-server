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
