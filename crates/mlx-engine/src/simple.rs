use std::path::Path;
use std::sync::Mutex;

use mlx_models::{AnyModel, sample};
use mlx_rs::{
    Array,
    ops::indexing::{IndexOp, NewAxis},
    transforms::eval,
};
use tokenizers::Tokenizer;

use crate::{
    chat_template::{ChatMessage, ChatTemplateRenderer},
    engine::{GenerationOutput, StreamingOutput},
    error::EngineError,
    model_loader,
    prompt_cache::PrefixCache,
};

/// Default maximum number of cached prefixes.
const DEFAULT_PREFIX_CACHE_SIZE: usize = 8;

/// Simple single-request inference engine with prefix KV caching.
///
/// Serializes requests with a mutex (same pattern as vllm-mlx's SimpleEngine).
/// Reuses cached KV states for shared prompt prefixes (e.g., system prompts).
pub struct SimpleEngine {
    model: Mutex<AnyModel>,
    prefix_cache: Mutex<PrefixCache>,
    tokenizer: Tokenizer,
    template: ChatTemplateRenderer,
    model_name: String,
    eos_token_ids: Vec<u32>,
}

impl SimpleEngine {
    /// Load a model and tokenizer from a directory.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, EngineError> {
        let model_dir = dir.as_ref();
        let model_name = model_dir
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_owned());

        tracing::info!(model_dir = %model_dir.display(), "Loading model");

        let model = model_loader::load_model(model_dir)?;
        let tokenizer = model_loader::load_tokenizer(model_dir)?;
        let template = ChatTemplateRenderer::from_model_dir(model_dir)?;

        let eos_token_ids = extract_eos_tokens(model_dir);

        tracing::info!(
            model_name = %model_name,
            eos_tokens = ?eos_token_ids,
            "Engine ready"
        );

        Ok(Self {
            model: Mutex::new(model),
            prefix_cache: Mutex::new(PrefixCache::new(DEFAULT_PREFIX_CACHE_SIZE)),
            tokenizer,
            template,
            model_name,
            eos_token_ids,
        })
    }

    /// Get the model name.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Get a reference to the tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Apply chat template and tokenize messages.
    pub fn prepare_chat_prompt(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[serde_json::Value]>,
    ) -> Result<Vec<u32>, EngineError> {
        let prompt = self.template.apply(messages, tools, true)?;
        let encoding = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(|e| EngineError::Tokenization(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Generate a complete response from a token prompt.
    pub fn generate(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
        stop_sequences: &[String],
    ) -> Result<GenerationOutput, EngineError> {
        if prompt_tokens.is_empty() {
            return Err(EngineError::Generation("Prompt is empty".to_owned()));
        }
        if max_tokens == 0 {
            return Ok(GenerationOutput {
                text: String::new(),
                finish_reason: "length".to_owned(),
                prompt_tokens: prompt_tokens
                    .len()
                    .try_into()
                    .map_err(|_| EngineError::Generation("Prompt too long".to_owned()))?,
                completion_tokens: 0,
            });
        }
        let prompt_len: u32 = prompt_tokens
            .len()
            .try_into()
            .map_err(|_| EngineError::Generation("Prompt too long".to_owned()))?;

        // Step 1: Check prefix cache (brief lock)
        let prefix_match = {
            let mut pc = self
                .prefix_cache
                .lock()
                .map_err(|e| EngineError::Generation(format!("Cache lock poisoned: {e}")))?;
            pc.find_longest_prefix(prompt_tokens)
        };

        // Step 2: Lock model and generate
        let mut model = self
            .model
            .lock()
            .map_err(|e| EngineError::Generation(format!("Model lock poisoned: {e}")))?;

        let (actual_prompt_tokens, mut cache) = if let Some(matched) = prefix_match {
            tracing::debug!(
                prefix_len = matched.prefix_len,
                total_len = prompt_tokens.len(),
                "Reusing cached prefix"
            );
            let suffix = prompt_tokens.get(matched.prefix_len..).unwrap_or_default();
            if suffix.is_empty() {
                // Full prefix match -- fall back to full prompt to avoid empty forward pass
                (prompt_tokens.to_vec(), model.make_cache())
            } else {
                (suffix.to_vec(), matched.cache)
            }
        } else {
            (prompt_tokens.to_vec(), model.make_cache())
        };

        let prompt_array = Array::from(actual_prompt_tokens.as_slice()).index(NewAxis);

        // Prefill: forward pass on the prompt
        let logits = model
            .forward(&prompt_array, None, &mut cache)
            .map_err(EngineError::Mlx)?;
        let mut current_token =
            sample(&logits.index((.., -1, ..)), temperature, top_p).map_err(EngineError::Mlx)?;
        eval([&current_token]).map_err(EngineError::Mlx)?;

        // Cache the state right after prefill (before any decode tokens)
        {
            let mut pc = self
                .prefix_cache
                .lock()
                .map_err(|e| EngineError::Generation(format!("Cache lock poisoned: {e}")))?;
            pc.store(prompt_tokens.to_vec(), cache.clone());
        }

        let mut tokens: Vec<u32> = Vec::new();
        let first_token_id: u32 = current_token.item();
        tokens.push(first_token_id);

        let first_decoded = self
            .tokenizer
            .decode(&tokens, true)
            .map_err(|e| EngineError::Tokenization(e.to_string()))?;

        if self.eos_token_ids.contains(&first_token_id) {
            return Ok(GenerationOutput {
                text: first_decoded,
                finish_reason: "stop".to_owned(),
                prompt_tokens: prompt_len,
                completion_tokens: 1,
            });
        }

        if !stop_sequences.is_empty() {
            if let Some(truncated) = check_stop_sequences(&first_decoded, stop_sequences) {
                return Ok(GenerationOutput {
                    text: truncated,
                    finish_reason: "stop".to_owned(),
                    prompt_tokens: prompt_len,
                    completion_tokens: 1,
                });
            }
        }

        if max_tokens <= 1 {
            return Ok(GenerationOutput {
                text: first_decoded,
                finish_reason: "length".to_owned(),
                prompt_tokens: prompt_len,
                completion_tokens: 1,
            });
        }

        // Decode loop
        loop {
            let decode_input_array = current_token.index((.., NewAxis));
            let decode_logits = model
                .forward(&decode_input_array, None, &mut cache)
                .map_err(EngineError::Mlx)?;
            current_token = sample(&decode_logits.index((.., -1, ..)), temperature, top_p)
                .map_err(EngineError::Mlx)?;

            let token_id: u32 = current_token.item();
            tokens.push(token_id);

            if tokens.len() % 32 == 0 {
                eval([&current_token]).map_err(EngineError::Mlx)?;
            }

            let completion_len: u32 = tokens
                .len()
                .try_into()
                .map_err(|_| EngineError::Generation("Too many tokens generated".to_owned()))?;

            if self.eos_token_ids.contains(&token_id) {
                let text = self
                    .tokenizer
                    .decode(&tokens, true)
                    .map_err(|e| EngineError::Tokenization(e.to_string()))?;
                return Ok(GenerationOutput {
                    text,
                    finish_reason: "stop".to_owned(),
                    prompt_tokens: prompt_len,
                    completion_tokens: completion_len,
                });
            }

            // Check stop sequences
            if !stop_sequences.is_empty() {
                let text = self
                    .tokenizer
                    .decode(&tokens, true)
                    .map_err(|e| EngineError::Tokenization(e.to_string()))?;
                if let Some(truncated) = check_stop_sequences(&text, stop_sequences) {
                    return Ok(GenerationOutput {
                        text: truncated,
                        finish_reason: "stop".to_owned(),
                        prompt_tokens: prompt_len,
                        completion_tokens: completion_len,
                    });
                }
            }

            if completion_len >= max_tokens {
                let text = self
                    .tokenizer
                    .decode(&tokens, true)
                    .map_err(|e| EngineError::Tokenization(e.to_string()))?;
                return Ok(GenerationOutput {
                    text,
                    finish_reason: "length".to_owned(),
                    prompt_tokens: prompt_len,
                    completion_tokens: completion_len,
                });
            }
        }
    }

    /// Generate tokens one at a time, sending each via the provided channel.
    ///
    /// If the receiver is dropped (client disconnected), generation stops early.
    pub fn generate_streaming(
        &self,
        prompt_tokens: &[u32],
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
        stop_sequences: &[String],
        sender: tokio::sync::mpsc::Sender<StreamingOutput>,
    ) -> Result<(), EngineError> {
        if prompt_tokens.is_empty() {
            return Err(EngineError::Generation("Prompt is empty".to_owned()));
        }
        if max_tokens == 0 {
            let prompt_len: u32 = prompt_tokens
                .len()
                .try_into()
                .map_err(|_| EngineError::Generation("Prompt too long".to_owned()))?;
            let _ = sender.blocking_send(StreamingOutput {
                new_text: String::new(),
                finished: true,
                finish_reason: Some("length".to_owned()),
                prompt_tokens: prompt_len,
                completion_tokens: 0,
            });
            return Ok(());
        }
        let prompt_len: u32 = prompt_tokens
            .len()
            .try_into()
            .map_err(|_| EngineError::Generation("Prompt too long".to_owned()))?;

        // Step 1: Check prefix cache
        let prefix_match = {
            let mut pc = self
                .prefix_cache
                .lock()
                .map_err(|e| EngineError::Generation(format!("Cache lock poisoned: {e}")))?;
            pc.find_longest_prefix(prompt_tokens)
        };

        // Step 2: Lock model and generate
        let mut model = self
            .model
            .lock()
            .map_err(|e| EngineError::Generation(format!("Model lock poisoned: {e}")))?;

        let (actual_prompt_tokens, mut cache) = if let Some(matched) = prefix_match {
            tracing::debug!(
                prefix_len = matched.prefix_len,
                total_len = prompt_tokens.len(),
                "Reusing cached prefix (streaming)"
            );
            let suffix = prompt_tokens.get(matched.prefix_len..).unwrap_or_default();
            if suffix.is_empty() {
                (prompt_tokens.to_vec(), model.make_cache())
            } else {
                (suffix.to_vec(), matched.cache)
            }
        } else {
            (prompt_tokens.to_vec(), model.make_cache())
        };

        let prompt_array = Array::from(actual_prompt_tokens.as_slice()).index(NewAxis);

        // Prefill
        let logits = model
            .forward(&prompt_array, None, &mut cache)
            .map_err(EngineError::Mlx)?;
        let mut current_token =
            sample(&logits.index((.., -1, ..)), temperature, top_p).map_err(EngineError::Mlx)?;
        eval([&current_token]).map_err(EngineError::Mlx)?;

        // Cache state after prefill
        {
            let mut pc = self
                .prefix_cache
                .lock()
                .map_err(|e| EngineError::Generation(format!("Cache lock poisoned: {e}")))?;
            pc.store(prompt_tokens.to_vec(), cache.clone());
        }

        let mut all_tokens: Vec<u32> = Vec::new();
        // Process first token
        let first_token_id: u32 = current_token.item();
        all_tokens.push(first_token_id);

        let first_decoded = self
            .tokenizer
            .decode(&all_tokens, true)
            .map_err(|e| EngineError::Tokenization(e.to_string()))?;
        let (first_text, first_hit_stop) = if !stop_sequences.is_empty() {
            if let Some(truncated) = check_stop_sequences(&first_decoded, stop_sequences) {
                (truncated, true)
            } else {
                (first_decoded.clone(), false)
            }
        } else {
            (first_decoded.clone(), false)
        };
        let mut prev_decoded_len = first_decoded.len();

        let first_is_eos = self.eos_token_ids.contains(&first_token_id);
        let finished = first_is_eos || first_hit_stop || 1 >= max_tokens;

        // Send first token; if receiver dropped, stop early
        if sender
            .blocking_send(StreamingOutput {
                new_text: first_text,
                finished,
                finish_reason: if first_is_eos || first_hit_stop {
                    Some("stop".to_owned())
                } else if 1 >= max_tokens {
                    Some("length".to_owned())
                } else {
                    None
                },
                prompt_tokens: prompt_len,
                completion_tokens: 1,
            })
            .is_err()
        {
            return Ok(());
        }

        if finished {
            return Ok(());
        }

        // Decode loop
        loop {
            let decode_input_array = current_token.index((.., NewAxis));
            let decode_logits = model
                .forward(&decode_input_array, None, &mut cache)
                .map_err(EngineError::Mlx)?;
            current_token = sample(&decode_logits.index((.., -1, ..)), temperature, top_p)
                .map_err(EngineError::Mlx)?;

            let token_id: u32 = current_token.item();
            all_tokens.push(token_id);

            if all_tokens.len() % 32 == 0 {
                eval([&current_token]).map_err(EngineError::Mlx)?;
            }

            let completion_len: u32 = all_tokens
                .len()
                .try_into()
                .map_err(|_| EngineError::Generation("Too many tokens generated".to_owned()))?;

            let full_text = self
                .tokenizer
                .decode(&all_tokens, true)
                .map_err(|e| EngineError::Tokenization(e.to_string()))?;
            let new_text = full_text
                .get(prev_decoded_len..)
                .unwrap_or_default()
                .to_owned();
            let old_decoded_len = prev_decoded_len;
            prev_decoded_len = full_text.len();

            // Check stop sequences against the full decoded text
            let (final_new_text, hit_stop_seq) = if !stop_sequences.is_empty() {
                if let Some(truncated) = check_stop_sequences(&full_text, stop_sequences) {
                    // Emit only the text between previous position and the stop boundary
                    let emit = truncated
                        .get(old_decoded_len..)
                        .unwrap_or_default()
                        .to_owned();
                    (emit, true)
                } else {
                    (new_text, false)
                }
            } else {
                (new_text, false)
            };

            let is_eos = self.eos_token_ids.contains(&token_id);
            let is_max = completion_len >= max_tokens;
            let step_finished = is_eos || is_max || hit_stop_seq;

            let finish_reason = if is_eos || hit_stop_seq {
                Some("stop".to_owned())
            } else if is_max {
                Some("length".to_owned())
            } else {
                None
            };

            // Send token; if receiver dropped (client disconnected), stop early
            if sender
                .blocking_send(StreamingOutput {
                    new_text: final_new_text,
                    finished: step_finished,
                    finish_reason,
                    prompt_tokens: prompt_len,
                    completion_tokens: completion_len,
                })
                .is_err()
            {
                return Ok(());
            }

            if step_finished {
                break;
            }
        }

        Ok(())
    }
}

/// Check if any stop sequence appears in the generated text.
/// Returns Some(truncated_text) if a stop sequence was found, None otherwise.
fn check_stop_sequences(text: &str, stop_sequences: &[String]) -> Option<String> {
    let mut earliest: Option<usize> = None;
    for seq in stop_sequences {
        if let Some(pos) = text.find(seq.as_str()) {
            earliest = Some(earliest.map_or(pos, |prev| prev.min(pos)));
        }
    }
    earliest.map(|pos| text.get(..pos).unwrap_or_default().to_owned())
}

/// Extract EOS token IDs from config.json.
fn extract_eos_tokens(model_dir: &Path) -> Vec<u32> {
    let config_path = model_dir.join("config.json");
    let config_str = match std::fs::read_to_string(&config_path) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(path = %config_path.display(), error = %e, "Could not read config.json for EOS tokens");
            return vec![];
        }
    };

    let config: serde_json::Value = match serde_json::from_str(&config_str) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(error = %e, "Could not parse config.json for EOS tokens");
            return vec![];
        }
    };

    match config.get("eos_token_id") {
        Some(serde_json::Value::Number(n)) => n
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .map_or_else(Vec::new, |id| vec![id]),
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_u64().and_then(|val| u32::try_from(val).ok()))
            .collect(),
        Some(other) => {
            tracing::warn!(value = ?other, "Unexpected eos_token_id type in config.json");
            vec![]
        }
        None => {
            tracing::warn!(
                "No eos_token_id found in config.json, generation will rely on max_tokens"
            );
            vec![]
        }
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::check_stop_sequences;

    #[test]
    fn test_single_stop_sequence_found() {
        let text = "Hello world, goodbye!";
        let stops = vec!["goodbye".to_owned()];
        let result = check_stop_sequences(text, &stops);
        assert_eq!(result, Some("Hello world, ".to_owned()));
    }

    #[test]
    fn test_no_stop_sequence_match() {
        let text = "Hello world";
        let stops = vec!["goodbye".to_owned(), "farewell".to_owned()];
        let result = check_stop_sequences(text, &stops);
        assert!(result.is_none());
    }

    #[test]
    fn test_empty_stop_sequences_list() {
        let text = "Hello world";
        let stops: Vec<String> = vec![];
        let result = check_stop_sequences(text, &stops);
        assert!(result.is_none());
    }

    #[test]
    fn test_empty_text() {
        let text = "";
        let stops = vec!["hello".to_owned()];
        let result = check_stop_sequences(text, &stops);
        assert!(result.is_none());
    }

    #[test]
    fn test_stop_sequence_at_beginning() {
        let text = "STOP rest of text";
        let stops = vec!["STOP".to_owned()];
        let result = check_stop_sequences(text, &stops);
        assert_eq!(result, Some(String::new()));
    }

    #[test]
    fn test_stop_sequence_at_end() {
        let text = "Hello world END";
        let stops = vec!["END".to_owned()];
        let result = check_stop_sequences(text, &stops);
        assert_eq!(result, Some("Hello world ".to_owned()));
    }

    #[test]
    fn test_multiple_stop_sequences_earliest_wins() {
        let text = "aaa bbb ccc ddd";
        // "ccc" appears at position 8, "bbb" at position 4
        // "bbb" should win because it appears earlier, regardless of array order
        let stops = vec!["ccc".to_owned(), "bbb".to_owned()];
        let result = check_stop_sequences(text, &stops);
        assert_eq!(result, Some("aaa ".to_owned()));
    }

    #[test]
    fn test_multiple_stop_sequences_earliest_wins_reverse_order() {
        let text = "aaa bbb ccc ddd";
        let stops = vec!["bbb".to_owned(), "ccc".to_owned()];
        let result = check_stop_sequences(text, &stops);
        assert_eq!(result, Some("aaa ".to_owned()));
    }

    #[test]
    fn test_overlapping_stop_sequences_prefix() {
        // "ab" is a prefix of "abc". "ab" appears first at position 0.
        let text = "abc def";
        let stops = vec!["abc".to_owned(), "ab".to_owned()];
        let result = check_stop_sequences(text, &stops);
        assert_eq!(result, Some(String::new()));
    }

    #[test]
    fn test_stop_sequence_appears_multiple_times() {
        let text = "before stop middle stop after";
        let stops = vec!["stop".to_owned()];
        let result = check_stop_sequences(text, &stops);
        assert_eq!(result, Some("before ".to_owned()));
    }

    #[test]
    fn test_stop_sequence_is_entire_text() {
        let text = "STOP";
        let stops = vec!["STOP".to_owned()];
        let result = check_stop_sequences(text, &stops);
        assert_eq!(result, Some(String::new()));
    }

    #[test]
    fn test_stop_sequence_with_newlines() {
        let text = "line one\nline two\nline three";
        let stops = vec!["\n".to_owned()];
        let result = check_stop_sequences(text, &stops);
        assert_eq!(result, Some("line one".to_owned()));
    }

    #[test]
    fn test_extract_eos_tokens_single_number() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"eos_token_id": 151643}"#,
        )
        .unwrap();
        let result = super::extract_eos_tokens(dir.path());
        assert_eq!(result, vec![151643]);
    }

    #[test]
    fn test_extract_eos_tokens_array() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"eos_token_id": [151643, 151645]}"#,
        )
        .unwrap();
        let result = super::extract_eos_tokens(dir.path());
        assert_eq!(result, vec![151643, 151645]);
    }

    #[test]
    fn test_extract_eos_tokens_missing_field() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), r#"{"model_type": "qwen2"}"#).unwrap();
        let result = super::extract_eos_tokens(dir.path());
        assert!(result.is_empty());
    }

    #[test]
    fn test_extract_eos_tokens_unexpected_type() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"eos_token_id": "string"}"#,
        )
        .unwrap();
        let result = super::extract_eos_tokens(dir.path());
        assert!(result.is_empty());
    }

    #[test]
    fn test_extract_eos_tokens_missing_config_file() {
        let dir = tempfile::tempdir().unwrap();
        let result = super::extract_eos_tokens(dir.path());
        assert!(result.is_empty());
    }

    // -- Additional check_stop_sequences edge cases --

    #[test]
    fn test_stop_sequence_substring_of_another() {
        // "stop" is a substring of "stop_now"
        let text = "Hello stop_now world";
        let stops = vec!["stop_now".to_owned(), "stop".to_owned()];
        let result = check_stop_sequences(text, &stops);
        // "stop" appears at position 6, "stop_now" also at position 6
        // Both match at same position, earliest is 6
        assert_eq!(result, Some("Hello ".to_owned()));
    }

    #[test]
    fn test_stop_sequence_unicode() {
        let text = "Hello world, a]b stop here";
        let stops = vec!["\u{1F600}".to_owned()]; // emoji stop sequence
        let result = check_stop_sequences(text, &stops);
        assert!(result.is_none()); // emoji not present

        let text_with_emoji = "Hello \u{1F600} world";
        let result2 = check_stop_sequences(text_with_emoji, &stops);
        assert_eq!(result2, Some("Hello ".to_owned()));
    }

    #[test]
    fn test_stop_sequence_unicode_multibyte() {
        let text = "Bonjour le monde, arr\u{00EA}t ici";
        let stops = vec!["arr\u{00EA}t".to_owned()];
        let result = check_stop_sequences(text, &stops);
        assert_eq!(result, Some("Bonjour le monde, ".to_owned()));
    }

    #[test]
    fn test_stop_sequence_very_long_text_short_stop() {
        let long_text = "a".repeat(10_000) + "STOP" + &"b".repeat(5_000);
        let stops = vec!["STOP".to_owned()];
        let result = check_stop_sequences(&long_text, &stops);
        assert_eq!(result, Some("a".repeat(10_000)));
    }

    // -- Additional extract_eos_tokens edge cases --

    #[test]
    fn test_extract_eos_tokens_float_value() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"eos_token_id": 151643.0}"#,
        )
        .unwrap();
        let result = super::extract_eos_tokens(dir.path());
        // serde_json parses 151643.0 as a float, and as_u64() returns None for floats,
        // so the result is empty. This is a Number variant but not integer-representable.
        assert!(result.is_empty());
    }

    #[test]
    fn test_extract_eos_tokens_string_value() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"eos_token_id": "not_a_number"}"#,
        )
        .unwrap();
        let result = super::extract_eos_tokens(dir.path());
        assert!(result.is_empty());
    }

    #[test]
    fn test_extract_eos_tokens_nested_array() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"eos_token_id": [[1, 2], [3, 4]]}"#,
        )
        .unwrap();
        let result = super::extract_eos_tokens(dir.path());
        // Nested arrays: inner arrays are not numbers, so as_u64() returns None for them
        assert!(result.is_empty());
    }

    #[test]
    fn test_extract_eos_tokens_negative_number() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), r#"{"eos_token_id": -1}"#).unwrap();
        let result = super::extract_eos_tokens(dir.path());
        // as_u64() returns None for negative numbers
        assert!(result.is_empty());
    }

    #[test]
    fn test_extract_eos_tokens_very_large_number() {
        let dir = tempfile::tempdir().unwrap();
        // u32::MAX is 4294967295, use a number larger than that
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"eos_token_id": 4294967296}"#,
        )
        .unwrap();
        let result = super::extract_eos_tokens(dir.path());
        // as_u64() succeeds but u32::try_from fails for values > u32::MAX
        assert!(result.is_empty());
    }

    #[test]
    fn test_extract_eos_tokens_empty_array() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), r#"{"eos_token_id": []}"#).unwrap();
        let result = super::extract_eos_tokens(dir.path());
        assert!(result.is_empty());
    }

    #[test]
    fn test_extract_eos_tokens_mixed_types_in_array() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"eos_token_id": [1, "two", 3]}"#,
        )
        .unwrap();
        let result = super::extract_eos_tokens(dir.path());
        // Only numeric entries are extracted; "two" is skipped
        assert_eq!(result, vec![1, 3]);
    }
}
