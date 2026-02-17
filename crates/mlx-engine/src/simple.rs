use std::path::Path;
use std::sync::Mutex;

use mlx_models::{Model, sample};
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
    model: Mutex<Model>,
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
            (suffix.to_vec(), matched.kv_cache)
        } else {
            (prompt_tokens.to_vec(), Vec::new())
        };

        let prompt_array = Array::from(actual_prompt_tokens.as_slice()).index(NewAxis);

        // Prefill: forward pass on the prompt
        let logits = model
            .forward(&prompt_array, None, &mut cache)
            .map_err(EngineError::Mlx)?;
        let mut current_token =
            sample(&logits.index((.., -1, ..)), temperature, top_p).map_err(EngineError::Mlx)?;
        eval([&current_token]).map_err(EngineError::Mlx)?;

        // Cache the KV state right after prefill (before any decode tokens)
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

        // Check first token for stop
        if self.eos_token_ids.contains(&first_token_id) {
            let text = self
                .tokenizer
                .decode(&tokens, true)
                .map_err(|e| EngineError::Tokenization(e.to_string()))?;
            return Ok(GenerationOutput {
                text,
                finish_reason: "stop".to_owned(),
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
            current_token = sample(&decode_logits, temperature, top_p).map_err(EngineError::Mlx)?;

            let token_id: u32 = current_token.item();
            tokens.push(token_id);

            if tokens.len()% 32 == 0 {
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
            (suffix.to_vec(), matched.kv_cache)
        } else {
            (prompt_tokens.to_vec(), Vec::new())
        };

        let prompt_array = Array::from(actual_prompt_tokens.as_slice()).index(NewAxis);

        // Prefill
        let logits = model
            .forward(&prompt_array, None, &mut cache)
            .map_err(EngineError::Mlx)?;
        let mut current_token =
            sample(&logits.index((.., -1, ..)), temperature, top_p).map_err(EngineError::Mlx)?;
        eval([&current_token]).map_err(EngineError::Mlx)?;

        // Cache KV state after prefill
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
        let first_text = first_decoded.clone();
        let mut prev_decoded_len = first_decoded.len();

        let first_is_eos = self.eos_token_ids.contains(&first_token_id);
        let finished = first_is_eos || 1 >= max_tokens;

        // Send first token; if receiver dropped, stop early
        if sender
            .blocking_send(StreamingOutput {
                new_text: first_text,
                finished,
                finish_reason: if first_is_eos {
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
            current_token = sample(&decode_logits, temperature, top_p).map_err(EngineError::Mlx)?;

            let token_id: u32 = current_token.item();
            all_tokens.push(token_id);

            if all_tokens.len()% 32 == 0 {
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
    for seq in stop_sequences {
        if let Some(pos) = text.find(seq.as_str()) {
            return Some(text.get(..pos).unwrap_or_default().to_owned());
        }
    }
    None
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
