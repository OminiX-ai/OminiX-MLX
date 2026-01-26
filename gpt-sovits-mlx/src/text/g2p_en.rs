//! G2P English - Grapheme-to-Phoneme for English words
//!
//! This module converts English words to ARPAbet phonemes, matching Python's g2p_en behavior.
//!
//! ## Architecture
//!
//! Two-stage lookup (same as Python's g2p_en):
//!
//! 1. **CMU Dictionary** (134K words) - Fast O(1) hash lookup with human-verified pronunciations
//! 2. **Neural Network** (GRU seq2seq) - For OOV (out-of-vocabulary) words not in CMU
//!
//! ## Why Both?
//!
//! The neural network learns patterns but isn't perfect:
//! - "steak" neural: `S T IY1 K` (wrong - sounds like "steek")
//! - "steak" CMU:    `S T EY1 K` (correct - sounds like "stake")
//!
//! CMU dictionary provides accurate pronunciations for common words.
//! Neural network handles rare words, misspellings, names, and neologisms.
//!
//! ## Neural Network Details
//!
//! - Architecture: GRU encoder-decoder (seq2seq)
//! - Encoder: 256-dim hidden, 29 graphemes (a-z + special tokens)
//! - Decoder: 256-dim hidden, 74 phonemes (ARPAbet + special tokens)
//! - Runtime: ONNX with CoreML acceleration on macOS
//! - Model source: https://github.com/kyubyong/g2p (checkpoint20.npz exported to ONNX)
//!
//! ## Files Required
//!
//! ```text
//! g2p_en_onnx/
//! ├── g2p_encoder.onnx   # GRU encoder (~1.6MB)
//! ├── g2p_decoder.onnx   # GRU decoder (~1.7MB)
//! ├── graphemes.json     # Input vocabulary
//! └── phonemes.json      # Output vocabulary
//! ```

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use ort::{ep, inputs, session::Session, value::Tensor};

use super::cmudict;

/// Global G2P English instance (lazy initialized)
static G2P_EN: OnceLock<Mutex<Option<G2pEnConverter>>> = OnceLock::new();

/// Grapheme vocabulary for the neural network
const GRAPHEMES: [&str; 29] = [
    "<pad>", "<unk>", "</s>", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
];

/// Phoneme vocabulary for the neural network
const PHONEMES: [&str; 74] = [
    "<pad>", "<unk>", "<s>", "</s>", "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2",
    "AO0", "AO1", "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "B", "CH", "D", "DH", "EH0",
    "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2", "F", "G", "HH", "IH0", "IH1", "IH2",
    "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N", "NG", "OW0", "OW1", "OW2", "OY0", "OY1", "OY2",
    "P", "R", "S", "SH", "T", "TH", "UH0", "UH1", "UH2", "UW", "UW0", "UW1", "UW2", "V", "W", "Y",
    "Z", "ZH",
];

/// Convert English word to phonemes using G2P (global instance)
/// Matches Python's g2p_en: CMU dictionary first, neural network for OOV
pub fn word_to_phonemes(word: &str) -> Vec<String> {
    // First try CMU dictionary (matches Python's g2p_en behavior)
    if let Some(phonemes) = cmudict::lookup(word) {
        return phonemes;
    }

    // Neural network for OOV words
    let mutex = G2P_EN.get_or_init(|| {
        let model_paths = [
            "/tmp/g2p_en_onnx",
            "/Users/yuechen/.dora/models/g2p_en_onnx",
        ];

        for path in model_paths {
            if Path::new(path).join("g2p_encoder.onnx").exists() {
                match G2pEnConverter::new(path) {
                    Ok(converter) => {
                        eprintln!("G2P-EN: Loaded from {}", path);
                        return Mutex::new(Some(converter));
                    }
                    Err(e) => {
                        eprintln!("G2P-EN: Failed to load from {}: {}", path, e);
                    }
                }
            }
        }
        eprintln!("G2P-EN: Model not found, using rule-based fallback");
        Mutex::new(None)
    });

    if let Ok(mut guard) = mutex.lock() {
        if let Some(ref mut converter) = *guard {
            if let Ok(phonemes) = converter.predict(word) {
                return phonemes;
            }
        }
    }

    // Ultimate fallback: rule-based G2P from cmudict
    cmudict::word_to_phonemes(word)
}

/// G2P English Converter using ONNX models
pub struct G2pEnConverter {
    encoder_session: Session,
    decoder_session: Session,
}

impl G2pEnConverter {
    /// Create a new G2P English converter
    pub fn new(model_dir: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let encoder_path = Path::new(model_dir).join("g2p_encoder.onnx");
        let decoder_path = Path::new(model_dir).join("g2p_decoder.onnx");

        // Load ONNX sessions with CoreML for acceleration
        let cache_dir = Path::new(model_dir).join("coreml_cache");
        std::fs::create_dir_all(&cache_dir).ok();

        let coreml_ep = ep::CoreML::default()
            .with_compute_units(ep::coreml::ComputeUnits::All)
            .with_model_format(ep::coreml::ModelFormat::NeuralNetwork)
            .with_model_cache_dir(cache_dir.to_string_lossy().to_string())
            .build();

        let encoder_session = Session::builder()?
            .with_execution_providers([coreml_ep.clone()])?
            .with_intra_threads(2)?
            .commit_from_file(&encoder_path)?;

        let decoder_session = Session::builder()?
            .with_execution_providers([coreml_ep])?
            .with_intra_threads(2)?
            .commit_from_file(&decoder_path)?;

        Ok(Self {
            encoder_session,
            decoder_session,
        })
    }

    /// Predict phonemes for an OOV word
    pub fn predict(&mut self, word: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let word_lower = word.to_lowercase();

        // Convert word to grapheme IDs
        let mut grapheme_ids: Vec<i64> = word_lower
            .chars()
            .map(|c| {
                GRAPHEMES
                    .iter()
                    .position(|&g| g.len() == 1 && g.chars().next().unwrap() == c)
                    .unwrap_or(1) as i64 // 1 = <unk>
            })
            .collect();

        // Add </s> token
        grapheme_ids.push(2); // 2 = </s>

        let seq_len = grapheme_ids.len();

        // Run encoder
        let grapheme_tensor =
            Tensor::from_array(([1, seq_len], grapheme_ids.into_boxed_slice()))?;
        let encoder_outputs = self.encoder_session.run(inputs![
            "grapheme_ids" => grapheme_tensor
        ])?;
        let (_, hidden_data) = encoder_outputs["hidden"].try_extract_tensor::<f32>()?;
        let mut hidden: Vec<f32> = hidden_data.iter().copied().collect();

        // Run decoder autoregressively
        let mut prev_token = 2i64; // <s> token
        let mut phoneme_indices = Vec::new();

        for _ in 0..30 {
            // Max 30 phonemes
            let prev_tensor = Tensor::from_array(([1], vec![prev_token].into_boxed_slice()))?;
            let hidden_tensor =
                Tensor::from_array(([1, 256], hidden.clone().into_boxed_slice()))?;

            let decoder_outputs = self.decoder_session.run(inputs![
                "prev_token" => prev_tensor,
                "hidden" => hidden_tensor
            ])?;

            // Get logits and new hidden
            let (_, logits_data) = decoder_outputs["logits"].try_extract_tensor::<f32>()?;
            let (_, new_hidden_data) = decoder_outputs["new_hidden"].try_extract_tensor::<f32>()?;

            // Find argmax
            let pred_idx = logits_data
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            if pred_idx == 3 {
                // </s>
                break;
            }

            phoneme_indices.push(pred_idx);
            prev_token = pred_idx as i64;
            hidden = new_hidden_data.iter().copied().collect();
        }

        // Convert indices to phoneme strings
        let phonemes: Vec<String> = phoneme_indices
            .iter()
            .filter_map(|&idx| PHONEMES.get(idx).map(|s| s.to_string()))
            .collect();

        Ok(phonemes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_word_to_phonemes() {
        // Known words should use CMU dictionary
        let hello = word_to_phonemes("hello");
        assert!(!hello.is_empty());

        // OOV words should use neural network
        let oov = word_to_phonemes("resturant");
        assert!(!oov.is_empty());
    }
}
