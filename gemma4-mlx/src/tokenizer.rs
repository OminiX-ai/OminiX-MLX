//! Tokenizer loading + Gemma 4 chat-turn formatting.
//!
//! Gemma 4 uses a different chat format than earlier Gemma versions.
//! The actual template (from chat_template.jinja) uses `<|turn>` / `<turn|>` tokens,
//! NOT `<start_of_turn>` / `<end_of_turn>` as in earlier Gemma models.
//!
//! Special tokens (confirmed from tokenizer.json):
//!   <bos>      = id 2   (BOS — NOT auto-added by the tokenizer post-processor)
//!   <eos>      = id 1
//!   <|turn>    = id 105
//!   <turn|>    = id 106 (end-of-turn; also used as generation stop)
//!   <|channel> = id 100
//!   <channel|> = id 101

use std::path::Path;
use tokenizers::Tokenizer;
use crate::error::Result;

const BOS_ID: i32 = 2;

/// Load the tokenizer from `tokenizer.json` in `model_dir`.
pub fn load_tokenizer(model_dir: &Path) -> Result<Tokenizer> {
    let path = model_dir.join("tokenizer.json");
    Tokenizer::from_file(&path).map_err(|e| crate::error::Error::Model(e.to_string()))
}

/// Render a single user turn and encode to token ids.
///
/// Format (no thinking mode, matching the Gemma 4 chat template):
///
/// ```text
/// <bos><|turn>user
/// {user_msg}<turn|>
/// <|turn>model
/// <|channel>thought
/// <channel|>
/// ```
///
/// The `<|channel>thought\n<channel|>` suffix is the thinking-suppression block
/// that the template emits when `enable_thinking=false`. Including it in the prompt
/// means the word "thought" never appears in the decoded reply output.
///
/// BOS (id 2) is prepended manually — the Gemma 4 tokenizer does NOT add it
/// automatically in its post-processor (verified: encoding with and without
/// `add_special_tokens=true` produce identical output, neither includes BOS).
pub fn encode_chat(tok: &Tokenizer, user_msg: &str) -> Result<Vec<i32>> {
    // Build the full prompt string. The thinking-suppression tokens are the
    // last four ids: <|channel>(100) + "thought"(45518) + "\n"(107) + <channel|>(101).
    let formatted = format!(
        "<|turn>user\n{user_msg}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    );
    let encoding = tok
        .encode(formatted.as_str(), false)
        .map_err(|e| crate::error::Error::Model(e.to_string()))?;

    // Prepend BOS (id=2) — the tokenizer does not add it automatically.
    let mut ids: Vec<i32> = Vec::with_capacity(encoding.len() + 1);
    ids.push(BOS_ID);
    ids.extend(encoding.get_ids().iter().map(|&id| id as i32));
    Ok(ids)
}

/// EOS ids that should stop generation: `<eos>` (1) and `<turn|>` (106).
///
/// `<turn|>` is the end-of-turn marker appended by the model at the end of each reply.
pub fn eos_ids() -> Vec<i32> {
    vec![1, 106]
}
