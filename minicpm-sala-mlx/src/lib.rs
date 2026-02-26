pub mod attention;
pub mod config;
pub mod metal_kernels;
pub mod model;
pub mod speculative;

pub use attention::{create_layer_caches, HybridAttention, LayerCache, LightningCache, SparseKVCache};
pub use config::{ModelArgs, QuantizationConfig};
pub use model::{load_model, load_tokenizer, get_model_args, sample, Model};
pub use speculative::SpeculativeDecoder;

/// EOS token ID.
pub const EOS_TOKEN_ID: u32 = 2;
/// `<|im_end|>` token ID.
pub const IM_END_TOKEN_ID: u32 = 73440;

/// Check if a token ID is a stop token (EOS or `<|im_end|>`).
pub fn is_stop_token(token_id: u32) -> bool {
    token_id == EOS_TOKEN_ID || token_id == IM_END_TOKEN_ID
}

/// Format a single-turn chat prompt in ChatML format for MiniCPM-SALA.
/// The tokenizer adds BOS (`<s>`) automatically.
pub fn format_chat_prompt(system: &str, user: &str) -> String {
    format!(
        "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    )
}

/// Format a multi-turn chat prompt in ChatML format.
/// `turns` is a list of (role, content) pairs where role is "user" or "assistant".
pub fn format_chat_prompt_multi(system: &str, turns: &[(&str, &str)]) -> String {
    let mut prompt = format!("<|im_start|>system\n{system}<|im_end|>\n");
    for (role, content) in turns {
        prompt.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

/// Strip `<think>...</think>` block from the beginning of generated text.
/// Returns the content after `</think>` if present, or the original text.
pub fn strip_thinking(text: &str) -> &str {
    if let Some(end) = text.find("</think>") {
        text[end + "</think>".len()..].trim_start_matches('\n')
    } else {
        text
    }
}

/// Incremental think-block filter for streaming output.
///
/// When `no_think` is true, suppresses all text inside `<think>...</think>` and
/// only emits text after the closing tag. When false, passes everything through.
pub struct ThinkFilter {
    think_done: bool,
    prev_text_len: usize,
}

impl ThinkFilter {
    pub fn new(no_think: bool) -> Self {
        Self {
            think_done: !no_think, // if not filtering, treat as already "done"
            prev_text_len: 0,
        }
    }

    /// Given the full decoded text so far, return the new text to print (if any).
    pub fn next<'a>(&mut self, full_text: &'a str) -> &'a str {
        if !self.think_done {
            if let Some(end) = full_text.find("</think>") {
                self.think_done = true;
                let after = &full_text[end + "</think>".len()..];
                let trimmed = after.trim_start_matches('\n');
                self.prev_text_len = full_text.len();
                return trimmed;
            }
            // Still inside <think> block — emit nothing
            return "";
        }

        if full_text.len() > self.prev_text_len {
            let new = &full_text[self.prev_text_len..];
            self.prev_text_len = full_text.len();
            new
        } else {
            ""
        }
    }
}
