//! Reference audio/text management for voice cloning
//!
//! Handles loading and storing reference audio, text, and semantic codes.
//! This module provides storage and basic loading functionality.
//! Semantic code extraction (which requires HuBERT + VITS components) is
//! handled by VoiceCloner.

use std::path::{Path, PathBuf};

use mlx_rs::{Array, transforms::eval};

use crate::audio::{AudioConfig, load_reference_mel};
use crate::error::Error;

/// Manages reference audio, text, and semantic codes for voice cloning
///
/// This struct handles:
/// - Loading mel spectrograms from audio files
/// - Storing reference text for few-shot mode
/// - Loading pre-computed semantic codes from files
/// - Storing semantic codes set by VoiceCloner after extraction
///
/// Note: Semantic code extraction requires HuBERT + VITS components,
/// so that logic lives in VoiceCloner, not here.
#[derive(Default)]
pub struct ReferenceManager {
    /// Reference mel spectrogram
    mel: Option<Array>,
    /// Path to reference audio file
    path: Option<String>,
    /// Reference text (for few-shot mode)
    text: Option<String>,
    /// Prompt semantic codes (for few-shot mode)
    semantic: Option<Array>,
}

impl ReferenceManager {
    /// Create a new empty reference manager
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if a reference has been loaded
    pub fn is_loaded(&self) -> bool {
        self.mel.is_some()
    }

    /// Check if few-shot mode is available (text and semantic codes set)
    pub fn has_few_shot_data(&self) -> bool {
        self.text.is_some() && self.semantic.is_some()
    }

    /// Get the reference mel spectrogram
    pub fn mel(&self) -> Option<&Array> {
        self.mel.as_ref()
    }

    /// Get the reference path
    pub fn path(&self) -> Option<&str> {
        self.path.as_deref()
    }

    /// Get the reference text
    pub fn text(&self) -> Option<&str> {
        self.text.as_deref()
    }

    /// Get the prompt semantic codes
    pub fn semantic(&self) -> Option<&Array> {
        self.semantic.as_ref()
    }

    /// Clone the mel spectrogram (needed for synthesis due to borrow checker)
    pub fn clone_mel(&self) -> Option<Array> {
        self.mel.clone()
    }

    /// Clone the semantic codes (needed for synthesis due to borrow checker)
    pub fn clone_semantic(&self) -> Option<Array> {
        self.semantic.clone()
    }

    /// Load reference from audio file (zero-shot mode)
    ///
    /// This loads only the mel spectrogram. For few-shot mode, also call
    /// `set_few_shot_data()` with extracted semantic codes.
    pub fn load_audio(&mut self, path: impl AsRef<Path>, audio_config: &AudioConfig) -> Result<(), Error> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(Error::model_not_found(path));
        }

        let mel = load_reference_mel(path, audio_config)
            .map_err(|e| Error::audio(format!("Failed to load reference audio: {}", e)))?;
        eval([&mel]).map_err(|e| Error::Message(format!("Failed to evaluate mel: {}", e)))?;

        self.mel = Some(mel);
        self.path = Some(path.to_string_lossy().to_string());
        self.text = None;
        self.semantic = None;

        Ok(())
    }

    /// Set few-shot data (text and semantic codes) after loading audio
    ///
    /// Call this after `load_audio()` to enable few-shot mode. The semantic
    /// codes should be extracted by VoiceCloner using HuBERT + VITS components.
    pub fn set_few_shot_data(&mut self, text: &str, semantic: Array) {
        self.text = Some(text.to_string());
        self.semantic = Some(semantic);
    }

    /// Load reference with pre-computed semantic codes
    pub fn load_with_precomputed_codes(
        &mut self,
        audio_path: impl AsRef<Path>,
        text: &str,
        codes_path: impl AsRef<Path>,
        audio_config: &AudioConfig,
    ) -> Result<(), Error> {
        let audio_path = audio_path.as_ref();
        let codes_path = codes_path.as_ref();

        if !audio_path.exists() {
            return Err(Error::model_not_found(audio_path));
        }
        if !codes_path.exists() {
            return Err(Error::model_not_found(codes_path));
        }

        // Load mel spectrogram
        let mel = load_reference_mel(audio_path, audio_config)
            .map_err(|e| Error::audio(format!("Failed to load reference audio: {}", e)))?;
        eval([&mel]).map_err(|e| Error::Message(format!("Failed to evaluate mel: {}", e)))?;

        // Load pre-computed codes
        let codes_path_buf = PathBuf::from(codes_path);
        let codes_data = std::fs::read(&codes_path_buf)?;

        // Parse codes (NPY or raw binary)
        let codes = parse_codes_file(&codes_data, &codes_path_buf)?;

        // Create Array from codes
        let codes_array = Array::from_slice(&codes, &[1, 1, codes.len() as i32]);

        self.mel = Some(mel);
        self.path = Some(audio_path.to_string_lossy().to_string());
        self.text = Some(text.to_string());
        self.semantic = Some(codes_array);

        Ok(())
    }

    /// Clear all reference data
    pub fn clear(&mut self) {
        self.mel = None;
        self.path = None;
        self.text = None;
        self.semantic = None;
    }
}

/// Parse codes from file (supports NPY and raw binary formats)
fn parse_codes_file(data: &[u8], path: &Path) -> Result<Vec<i32>, Error> {
    const NPY_MAGIC: &[u8] = b"\x93NUMPY";
    const NPY_MIN_HEADER: usize = 10;
    const MAX_HEADER_SIZE: usize = 10_000;

    if data.len() > NPY_MIN_HEADER && data.get(..NPY_MAGIC.len()) == Some(NPY_MAGIC) {
        // Parse NPY file
        let search_end = (NPY_MIN_HEADER + MAX_HEADER_SIZE).min(data.len());
        let header_end = data[NPY_MIN_HEADER..search_end]
            .iter()
            .position(|&b| b == b'\n')
            .map(|pos| NPY_MIN_HEADER + pos + 1)
            .ok_or_else(|| Error::file_corrupted(path, "NPY header newline not found"))?;

        let data_len = data.len() - header_end;
        if data_len % 4 != 0 {
            return Err(Error::file_corrupted(path,
                format!("NPY data not aligned to 4 bytes (size={})", data_len)));
        }

        Ok(data[header_end..]
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    } else {
        // Raw binary format
        if data.len() % 4 != 0 {
            return Err(Error::file_corrupted(path,
                format!("Binary data not aligned to 4 bytes (size={})", data.len())));
        }

        Ok(data
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }
}

impl std::fmt::Debug for ReferenceManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReferenceManager")
            .field("has_mel", &self.mel.is_some())
            .field("path", &self.path)
            .field("has_text", &self.text.is_some())
            .field("has_semantic", &self.semantic.is_some())
            .finish()
    }
}
