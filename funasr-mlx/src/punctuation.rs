//! CT-Transformer punctuation restoration via ONNX Runtime
//!
//! Adds punctuation (commas, periods, question marks) to raw ASR text output
//! using the FunASR CT-Transformer model running on ONNX Runtime.

use crate::{Error, Result};
use ort::session::Session;
use std::collections::HashMap;
use std::path::Path;

/// Punctuation classes output by CT-Transformer (6 classes)
const PUNC_SYMBOLS: &[&str] = &["<unk>", "", "，", "。", "？", "、"];

/// CT-Transformer punctuation model
pub struct PunctuationModel {
    session: Session,
    token_to_id: HashMap<String, i32>,
    unk_id: i32,
}

impl PunctuationModel {
    /// Load punctuation model from directory containing model_quant.onnx and tokens.json
    pub fn load(model_dir: &Path) -> Result<Self> {
        // Load ONNX model (prefer quantized)
        let model_path = {
            let quant = model_dir.join("model_quant.onnx");
            if quant.exists() {
                quant
            } else {
                model_dir.join("model.onnx")
            }
        };

        if !model_path.exists() {
            return Err(Error::Model(format!(
                "Punctuation ONNX model not found at {:?}",
                model_path
            )));
        }

        let session = Session::builder()
            .and_then(|b| b.with_intra_threads(2))
            .and_then(|b| b.commit_from_file(&model_path))
            .map_err(|e| Error::Model(format!("Failed to load ONNX punctuation model: {}", e)))?;

        // Load vocabulary
        let tokens_path = model_dir.join("tokens.json");
        if !tokens_path.exists() {
            return Err(Error::Model(format!(
                "Punctuation tokens.json not found at {:?}",
                tokens_path
            )));
        }

        let tokens_content = std::fs::read_to_string(&tokens_path)?;
        let tokens: Vec<String> = serde_json::from_str(&tokens_content)
            .map_err(|e| Error::Model(format!("Failed to parse tokens.json: {}", e)))?;

        let unk_id = (tokens.len() as i32) - 1; // Last token is <unk>
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (i, token) in tokens.iter().enumerate() {
            token_to_id.insert(token.clone(), i as i32);
        }

        Ok(Self {
            session,
            token_to_id,
            unk_id,
        })
    }

    /// Add punctuation to raw ASR text
    pub fn punctuate(&mut self, text: &str) -> Result<String> {
        if text.is_empty() {
            return Ok(String::new());
        }

        let tokens = segment_text(text);
        if tokens.is_empty() {
            return Ok(text.to_string());
        }

        // Convert tokens to IDs
        let token_ids: Vec<i32> = tokens
            .iter()
            .map(|t| *self.token_to_id.get(t.as_str()).unwrap_or(&self.unk_id))
            .collect();

        let seq_len = token_ids.len();

        // Build input tensors
        let inputs =
            ort::value::Tensor::from_array(([1, seq_len], token_ids.clone()))
                .map_err(|e| Error::Model(format!("Failed to create input tensor: {}", e)))?;

        let text_lengths =
            ort::value::Tensor::from_array(([1usize], vec![seq_len as i32]))
                .map_err(|e| Error::Model(format!("Failed to create lengths tensor: {}", e)))?;

        // Run inference
        let outputs = self
            .session
            .run(ort::inputs![inputs, text_lengths])
            .map_err(|e| Error::Model(format!("ONNX inference failed: {}", e)))?;

        // Extract logits [1, seq_len, 6]
        let logits_value = &outputs[0];
        let (logits_shape, logits_slice) = logits_value
            .try_extract_tensor::<f32>()
            .map_err(|e| Error::Model(format!("Failed to extract logits: {}", e)))?;

        let num_classes = if logits_shape.len() == 3 { logits_shape[2] as usize } else { 6 };
        let punc_classes: Vec<usize> = (0..seq_len)
            .map(|i| {
                let offset = i * num_classes;
                let slice = &logits_slice[offset..offset + num_classes];
                slice
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(1)
            })
            .collect();

        // Reconstruct punctuated text
        let mut result = String::with_capacity(text.len() * 2);
        for (i, token) in tokens.iter().enumerate() {
            // Add space between consecutive ASCII tokens
            if i > 0 && is_ascii_word(token) && is_ascii_word(&tokens[i - 1]) {
                result.push(' ');
            }
            result.push_str(token);

            // Append punctuation symbol (skip class 0=unk, 1=none)
            let cls = punc_classes[i];
            if cls >= 2 && cls < PUNC_SYMBOLS.len() {
                result.push_str(PUNC_SYMBOLS[cls]);
            }
        }

        // Ensure text ends with sentence-ending punctuation
        ensure_sentence_ending(&mut result);

        Ok(result)
    }
}

/// Segment text into tokens: CJK chars individually, ASCII words grouped
fn segment_text(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut ascii_buf = String::new();

    for ch in text.chars() {
        if is_cjk(ch) {
            if !ascii_buf.is_empty() {
                tokens.push(ascii_buf.clone());
                ascii_buf.clear();
            }
            tokens.push(ch.to_string());
        } else if ch.is_ascii_alphanumeric() || ch == '\'' {
            ascii_buf.push(ch);
        } else {
            // Whitespace or other — flush ASCII buffer
            if !ascii_buf.is_empty() {
                tokens.push(ascii_buf.clone());
                ascii_buf.clear();
            }
        }
    }

    if !ascii_buf.is_empty() {
        tokens.push(ascii_buf);
    }

    tokens
}

fn is_cjk(ch: char) -> bool {
    let cp = ch as u32;
    (0x4E00..=0x9FFF).contains(&cp)       // CJK Unified Ideographs
        || (0x3400..=0x4DBF).contains(&cp) // CJK Extension A
        || (0x20000..=0x2A6DF).contains(&cp) // CJK Extension B
        || (0xF900..=0xFAFF).contains(&cp) // CJK Compatibility Ideographs
        || (0x2F800..=0x2FA1F).contains(&cp) // CJK Compatibility Supplement
        || (0x3000..=0x303F).contains(&cp) // CJK Symbols and Punctuation
        || (0x3040..=0x309F).contains(&cp) // Hiragana
        || (0x30A0..=0x30FF).contains(&cp) // Katakana
        || (0xAC00..=0xD7AF).contains(&cp) // Hangul
}

fn is_ascii_word(s: &str) -> bool {
    s.chars().next().map_or(false, |c| c.is_ascii_alphanumeric())
}

fn ensure_sentence_ending(text: &mut String) {
    let trimmed = text.trim_end();
    if trimmed.is_empty() {
        return;
    }
    let last = trimmed.chars().last().unwrap();
    if last == '，' || last == '、' {
        // Replace trailing comma with period
        let end = text.trim_end_matches(|c| c == '，' || c == '、').len();
        text.truncate(end);
        text.push('。');
    } else if last != '。' && last != '？' && last != '！' && last != '.' && last != '?' && last != '!' {
        text.push('。');
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_text_chinese() {
        let tokens = segment_text("今天天气很好");
        assert_eq!(tokens, vec!["今", "天", "天", "气", "很", "好"]);
    }

    #[test]
    fn test_segment_text_mixed() {
        let tokens = segment_text("hello世界test");
        assert_eq!(tokens, vec!["hello", "世", "界", "test"]);
    }

    #[test]
    fn test_segment_text_english() {
        let tokens = segment_text("hello world");
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_is_cjk() {
        assert!(is_cjk('中'));
        assert!(is_cjk('国'));
        assert!(!is_cjk('a'));
        assert!(!is_cjk('1'));
    }

    #[test]
    fn test_ensure_sentence_ending() {
        let mut s = "你好，世界，".to_string();
        ensure_sentence_ending(&mut s);
        assert_eq!(s, "你好，世界。");

        let mut s2 = "你好。".to_string();
        ensure_sentence_ending(&mut s2);
        assert_eq!(s2, "你好。");

        let mut s3 = "你好".to_string();
        ensure_sentence_ending(&mut s3);
        assert_eq!(s3, "你好。");
    }
}
