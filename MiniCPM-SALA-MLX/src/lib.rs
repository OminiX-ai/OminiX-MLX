pub mod attention;
pub mod config;
pub mod metal_kernels;
pub mod model;
pub mod speculative;

pub use attention::{create_layer_caches, HybridAttention, LayerCache, LightningCache, SparseKVCache};
pub use config::{ModelArgs, QuantizationConfig};
pub use model::{load_model, load_tokenizer, get_model_args, sample, Model};
pub use speculative::SpeculativeDecoder;

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
