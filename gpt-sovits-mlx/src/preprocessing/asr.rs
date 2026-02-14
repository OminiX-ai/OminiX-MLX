//! ASR (Automatic Speech Recognition) for transcription
//!
//! Uses FunASR Paraformer model for Chinese/English speech recognition

use std::path::{Path, PathBuf};
use super::{PreprocessingError, Result};
use tracing::{debug, info};

/// Configuration for ASR
#[derive(Debug, Clone)]
pub struct ASRConfig {
    /// Path to the ASR model directory
    pub model_path: PathBuf,
    /// Language for recognition (zh, en, auto)
    pub language: String,
    /// Whether to use half precision
    pub half_precision: bool,
    /// Sample rate expected by the model
    pub sample_rate: u32,
}

impl Default for ASRConfig {
    fn default() -> Self {
        // Default to FunASR model path
        let model_path = dirs::home_dir()
            .map(|h| h.join(".OminiX/models/funasr"))
            .unwrap_or_else(|| PathBuf::from("models/funasr"));

        Self {
            model_path,
            language: "zh".to_string(),
            half_precision: false,
            sample_rate: 16000,
        }
    }
}

/// Transcription result
#[derive(Debug, Clone)]
pub struct Transcript {
    /// The transcribed text
    pub text: String,
    /// Detected language (if available)
    pub language: Option<String>,
    /// Confidence score (if available)
    pub confidence: Option<f32>,
    /// Word-level timestamps (if available)
    pub timestamps: Option<Vec<WordTimestamp>>,
}

/// Word-level timestamp
#[derive(Debug, Clone)]
pub struct WordTimestamp {
    pub word: String,
    pub start_ms: u64,
    pub end_ms: u64,
}

/// ASR processor using FunASR
pub struct ASRProcessor {
    config: ASRConfig,
    // Note: In a full implementation, this would hold the loaded model
    // For now, we provide a framework that can be extended
    model_loaded: bool,
}

impl ASRProcessor {
    /// Create a new ASR processor
    pub fn new(config: ASRConfig) -> Result<Self> {
        info!(model_path = %config.model_path.display(), "Initializing ASR processor");

        // Check if model exists
        if !config.model_path.exists() {
            return Err(PreprocessingError::Config(format!(
                "ASR model not found at: {}. Please download the FunASR model.",
                config.model_path.display()
            )));
        }

        Ok(Self {
            config,
            model_loaded: false,
        })
    }

    /// Load the ASR model (lazy loading)
    fn ensure_model_loaded(&mut self) -> Result<()> {
        if self.model_loaded {
            return Ok(());
        }

        info!("Loading ASR model...");

        // TODO: Integrate with funasr-mlx crate
        // For now, we provide a placeholder that can be extended
        //
        // When funasr-mlx is integrated:
        // self.model = Some(FunASR::load(&self.config.model_path)?);

        self.model_loaded = true;
        info!("ASR model loaded");

        Ok(())
    }

    /// Transcribe audio samples
    pub fn transcribe(&mut self, samples: &[f32], sample_rate: u32) -> Result<Transcript> {
        self.ensure_model_loaded()?;

        // Resample if needed
        let samples = if sample_rate != self.config.sample_rate {
            debug!(from = sample_rate, to = self.config.sample_rate, "Resampling for ASR");
            mlx_rs_core::audio::resample(samples, sample_rate, self.config.sample_rate)
        } else {
            samples.to_vec()
        };

        // TODO: Integrate with funasr-mlx for actual transcription
        // For now, return a placeholder
        //
        // When funasr-mlx is integrated:
        // let result = self.model.as_ref().unwrap().transcribe(&samples)?;
        // return Ok(Transcript {
        //     text: result.text,
        //     language: Some(result.language),
        //     confidence: Some(result.confidence),
        //     timestamps: result.timestamps,
        // });

        // Placeholder - in production, this would use the actual model
        debug!(samples = samples.len(), "Transcribing audio (placeholder)");

        Ok(Transcript {
            text: String::new(), // Placeholder - would be actual transcription
            language: Some(self.config.language.clone()),
            confidence: None,
            timestamps: None,
        })
    }

    /// Transcribe an audio file
    pub fn transcribe_file<P: AsRef<Path>>(&mut self, path: P) -> Result<Transcript> {
        let path = path.as_ref();
        debug!(path = %path.display(), "Transcribing file");

        // Load audio
        let (samples, sr) = mlx_rs_core::audio::load_wav(path)
            .map_err(|e| PreprocessingError::Audio(format!("Failed to load audio: {}", e)))?;

        self.transcribe(&samples, sr)
    }

    /// Batch transcribe multiple audio files
    pub fn transcribe_batch<P: AsRef<Path>>(&mut self, paths: &[P]) -> Result<Vec<Transcript>> {
        paths.iter().map(|p| self.transcribe_file(p)).collect()
    }
}

/// Alternative: Use external ASR tool via subprocess
pub struct ExternalASR {
    /// Path to the ASR script/binary
    pub command: String,
    /// Additional arguments
    pub args: Vec<String>,
}

impl ExternalASR {
    /// Create an external ASR processor that calls a Python script
    pub fn python_funasr(python_path: &str, script_path: &str) -> Self {
        Self {
            command: python_path.to_string(),
            args: vec![
                "-s".to_string(),
                script_path.to_string(),
            ],
        }
    }

    /// Transcribe a directory of audio files using external tool
    pub fn transcribe_directory<P: AsRef<Path>>(
        &self,
        input_dir: P,
        output_dir: P,
        language: &str,
    ) -> Result<PathBuf> {
        let input_dir = input_dir.as_ref();
        let output_dir = output_dir.as_ref();

        std::fs::create_dir_all(output_dir)?;

        let mut cmd = std::process::Command::new(&self.command);
        cmd.args(&self.args)
            .arg("-i").arg(input_dir)
            .arg("-o").arg(output_dir)
            .arg("-l").arg(language);

        info!(command = ?cmd, "Running external ASR");

        let output = cmd.output()
            .map_err(|e| PreprocessingError::ASR(format!("Failed to run ASR: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(PreprocessingError::ASR(format!("ASR failed: {}", stderr)));
        }

        // Return path to output transcript file
        let input_name = input_dir.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("output");
        let output_file = output_dir.join(format!("{}.list", input_name));

        Ok(output_file)
    }
}

/// Parse GPT-SoVITS transcript list file
pub fn parse_transcript_list<P: AsRef<Path>>(path: P) -> Result<Vec<(PathBuf, String, String)>> {
    let content = std::fs::read_to_string(path.as_ref())?;
    let mut results = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Format: audio_path|speaker|language|text
        let parts: Vec<&str> = line.split('|').collect();
        if parts.len() >= 4 {
            let audio_path = PathBuf::from(parts[0]);
            let language = parts[2].to_string();
            let text = parts[3..].join("|"); // In case text contains |
            results.push((audio_path, language, text));
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asr_config_default() {
        let config = ASRConfig::default();
        assert_eq!(config.language, "zh");
        assert_eq!(config.sample_rate, 16000);
    }

    #[test]
    fn test_parse_transcript_list() {
        use std::io::Write;

        // Create a temporary file
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_transcript.list");

        let mut file = std::fs::File::create(&temp_file).unwrap();
        writeln!(file, "/path/to/audio1.wav|speaker|zh|你好世界").unwrap();
        writeln!(file, "/path/to/audio2.wav|speaker|en|Hello world").unwrap();
        drop(file);

        let results = parse_transcript_list(&temp_file).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].2, "你好世界");
        assert_eq!(results[1].2, "Hello world");

        // Cleanup
        std::fs::remove_file(temp_file).ok();
    }
}
