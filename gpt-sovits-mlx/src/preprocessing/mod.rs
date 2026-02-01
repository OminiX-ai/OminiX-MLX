//! Data Preprocessing for GPT-SoVITS Training
//!
//! This module provides preprocessing utilities for preparing audio data for training:
//!
//! - **AudioSlicer**: Split long audio into shorter segments based on silence detection
//! - **ASR**: Automatic speech recognition using FunASR
//! - **Denoiser**: Audio denoising using spectral subtraction
//!
//! # Example
//!
//! ```rust,no_run
//! use gpt_sovits_mlx::preprocessing::{AudioSlicer, SlicerConfig};
//!
//! let config = SlicerConfig::default();
//! let slicer = AudioSlicer::new(config);
//!
//! // Slice a long audio file
//! let chunks = slicer.slice_file("input.wav", "output_dir/")?;
//! println!("Created {} audio chunks", chunks.len());
//! ```

mod slicer;
mod asr;
mod denoise;

pub use slicer::{AudioSlicer, SlicerConfig, AudioChunk};
pub use asr::{ASRProcessor, ASRConfig, Transcript};
pub use denoise::{Denoiser, DenoiseConfig};

use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PreprocessingError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Audio error: {0}")]
    Audio(String),

    #[error("ASR error: {0}")]
    ASR(String),

    #[error("Invalid configuration: {0}")]
    Config(String),
}

pub type Result<T> = std::result::Result<T, PreprocessingError>;

/// Preprocessing pipeline configuration
#[derive(Debug, Clone)]
pub struct PreprocessingConfig {
    /// Audio slicer configuration
    pub slicer: SlicerConfig,
    /// ASR configuration
    pub asr: ASRConfig,
    /// Denoiser configuration
    pub denoise: DenoiseConfig,
    /// Whether to run denoising
    pub enable_denoise: bool,
    /// Output sample rate
    pub output_sample_rate: u32,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            slicer: SlicerConfig::default(),
            asr: ASRConfig::default(),
            denoise: DenoiseConfig::default(),
            enable_denoise: false,
            output_sample_rate: 32000,
        }
    }
}

/// Complete preprocessing pipeline
pub struct PreprocessingPipeline {
    config: PreprocessingConfig,
    slicer: AudioSlicer,
    asr: Option<ASRProcessor>,
    denoiser: Option<Denoiser>,
}

impl PreprocessingPipeline {
    /// Create a new preprocessing pipeline
    pub fn new(config: PreprocessingConfig) -> Result<Self> {
        let slicer = AudioSlicer::new(config.slicer.clone());

        Ok(Self {
            config,
            slicer,
            asr: None,
            denoiser: None,
        })
    }

    /// Initialize ASR (lazy loading)
    pub fn init_asr(&mut self) -> Result<()> {
        if self.asr.is_none() {
            self.asr = Some(ASRProcessor::new(self.config.asr.clone())?);
        }
        Ok(())
    }

    /// Initialize denoiser (lazy loading)
    pub fn init_denoiser(&mut self) -> Result<()> {
        if self.denoiser.is_none() {
            self.denoiser = Some(Denoiser::new(self.config.denoise.clone())?);
        }
        Ok(())
    }

    /// Process a single audio file through the full pipeline
    pub fn process_file<P1: AsRef<Path>, P2: AsRef<Path>>(
        &mut self,
        input_path: P1,
        output_dir: P2,
    ) -> Result<Vec<ProcessedChunk>> {
        let input_path = input_path.as_ref();
        let output_dir = output_dir.as_ref();

        // Create output directory
        std::fs::create_dir_all(output_dir)?;

        // Step 1: Slice audio
        let chunks = self.slicer.slice_file(input_path, output_dir)?;

        // Step 2: Optionally denoise
        let chunks = if self.config.enable_denoise {
            self.init_denoiser()?;
            let denoiser = self.denoiser.as_ref().unwrap();
            chunks.into_iter()
                .map(|chunk| {
                    let denoised_path = chunk.output_path.with_extension("denoised.wav");
                    denoiser.process_file(&chunk.output_path, &denoised_path)?;
                    Ok(AudioChunk {
                        output_path: denoised_path,
                        ..chunk
                    })
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            chunks
        };

        // Step 3: Run ASR
        self.init_asr()?;
        let asr = self.asr.as_mut().unwrap();

        let mut results = Vec::new();
        for chunk in chunks {
            let transcript = asr.transcribe_file(&chunk.output_path)?;
            results.push(ProcessedChunk {
                audio_path: chunk.output_path,
                start_ms: chunk.start_ms,
                end_ms: chunk.end_ms,
                transcript: transcript.text,
                language: transcript.language,
            });
        }

        Ok(results)
    }

    /// Process a directory of audio files
    pub fn process_directory<P1: AsRef<Path>, P2: AsRef<Path>>(
        &mut self,
        input_dir: P1,
        output_dir: P2,
    ) -> Result<Vec<ProcessedChunk>> {
        let input_dir = input_dir.as_ref();
        let output_dir = output_dir.as_ref();

        let mut all_results = Vec::new();

        for entry in std::fs::read_dir(input_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                if let Some(ext) = path.extension() {
                    let ext = ext.to_string_lossy().to_lowercase();
                    if ["wav", "mp3", "flac", "ogg", "m4a"].contains(&ext.as_str()) {
                        let results = self.process_file(&path, output_dir)?;
                        all_results.extend(results);
                    }
                }
            }
        }

        Ok(all_results)
    }

    /// Write transcript list file (GPT-SoVITS format)
    pub fn write_transcript_list<P: AsRef<Path>>(
        results: &[ProcessedChunk],
        output_path: P,
    ) -> Result<()> {
        use std::io::Write;

        let mut file = std::fs::File::create(output_path)?;

        for chunk in results {
            // Format: audio_path|speaker|language|transcript
            writeln!(
                file,
                "{}|speaker|{}|{}",
                chunk.audio_path.display(),
                chunk.language.as_deref().unwrap_or("zh"),
                chunk.transcript
            )?;
        }

        Ok(())
    }
}

/// Result of processing a single audio chunk
#[derive(Debug, Clone)]
pub struct ProcessedChunk {
    /// Path to the processed audio file
    pub audio_path: std::path::PathBuf,
    /// Start time in milliseconds (from original file)
    pub start_ms: u64,
    /// End time in milliseconds (from original file)
    pub end_ms: u64,
    /// Transcribed text
    pub transcript: String,
    /// Detected language
    pub language: Option<String>,
}
