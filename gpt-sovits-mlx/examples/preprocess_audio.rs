//! Audio Preprocessing CLI
//!
//! Preprocess audio files for GPT-SoVITS training:
//! - Slice long audio into shorter segments
//! - Transcribe audio using ASR
//! - Optionally denoise audio
//!
//! # Usage
//!
//! ```bash
//! # Slice audio files
//! cargo run --example preprocess_audio --release -- slice -i input_dir -o output_dir
//!
//! # Transcribe sliced audio
//! cargo run --example preprocess_audio --release -- asr -i sliced_dir -o output.list
//!
//! # Full pipeline
//! cargo run --example preprocess_audio --release -- pipeline -i raw_audio -o processed_dir
//! ```

use std::path::PathBuf;
use clap::{Parser, Subcommand};
use tracing::{info, Level};
use tracing_subscriber::EnvFilter;

use gpt_sovits_mlx::preprocessing::{
    AudioSlicer, SlicerConfig,
    ASRProcessor, ASRConfig,
    Denoiser, DenoiseConfig,
    PreprocessingPipeline, PreprocessingConfig,
};

#[derive(Parser)]
#[command(name = "preprocess_audio")]
#[command(about = "Audio preprocessing for GPT-SoVITS training")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Slice long audio files into shorter segments
    Slice {
        /// Input audio file or directory
        #[arg(short, long)]
        input: PathBuf,

        /// Output directory for sliced audio
        #[arg(short, long)]
        output: PathBuf,

        /// Silence threshold in dB (default: -40)
        #[arg(long, default_value = "-40")]
        threshold: f32,

        /// Minimum chunk length in ms (default: 5000)
        #[arg(long, default_value = "5000")]
        min_length: u32,

        /// Minimum silence interval in ms (default: 300)
        #[arg(long, default_value = "300")]
        min_interval: u32,

        /// Maximum silence to keep in ms (default: 1000)
        #[arg(long, default_value = "1000")]
        max_sil_kept: u32,

        /// Output sample rate (default: 32000)
        #[arg(long, default_value = "32000")]
        sample_rate: u32,
    },

    /// Transcribe audio files using ASR
    Asr {
        /// Input audio file or directory
        #[arg(short, long)]
        input: PathBuf,

        /// Output transcript file (.list format)
        #[arg(short, long)]
        output: PathBuf,

        /// Language (zh, en, auto)
        #[arg(short, long, default_value = "zh")]
        language: String,

        /// ASR model path
        #[arg(long)]
        model: Option<PathBuf>,
    },

    /// Denoise audio files
    Denoise {
        /// Input audio file or directory
        #[arg(short, long)]
        input: PathBuf,

        /// Output directory
        #[arg(short, long)]
        output: PathBuf,

        /// Over-subtraction factor (default: 1.0)
        #[arg(long, default_value = "1.0")]
        strength: f32,
    },

    /// Run full preprocessing pipeline
    Pipeline {
        /// Input audio file or directory
        #[arg(short, long)]
        input: PathBuf,

        /// Output directory
        #[arg(short, long)]
        output: PathBuf,

        /// Enable denoising
        #[arg(long)]
        denoise: bool,

        /// Language for ASR (zh, en, auto)
        #[arg(short, long, default_value = "zh")]
        language: String,

        /// Output sample rate
        #[arg(long, default_value = "32000")]
        sample_rate: u32,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive(Level::INFO.into())
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Slice {
            input,
            output,
            threshold,
            min_length,
            min_interval,
            max_sil_kept,
            sample_rate,
        } => {
            info!("Slicing audio files");

            let config = SlicerConfig {
                sample_rate,
                threshold_db: threshold,
                min_length_ms: min_length,
                min_interval_ms: min_interval,
                max_sil_kept_ms: max_sil_kept,
                ..Default::default()
            };

            let slicer = AudioSlicer::new(config);

            let chunks = if input.is_dir() {
                slicer.slice_directory(&input, &output)?
            } else {
                slicer.slice_file(&input, &output)?
            };

            info!(
                chunks = chunks.len(),
                output_dir = %output.display(),
                "Slicing complete"
            );

            // Print summary
            println!("\nSlicing Summary:");
            println!("================");
            println!("Input: {}", input.display());
            println!("Output: {}", output.display());
            println!("Chunks created: {}", chunks.len());

            let total_duration: f64 = chunks.iter()
                .map(|c| (c.end_ms - c.start_ms) as f64 / 1000.0)
                .sum();
            println!("Total duration: {:.1}s", total_duration);
        }

        Commands::Asr {
            input,
            output,
            language,
            model,
        } => {
            info!("Transcribing audio files");

            let config = ASRConfig {
                language: language.clone(),
                model_path: model.unwrap_or_else(|| {
                    dirs::home_dir()
                        .map(|h| h.join(".OminiX/models/funasr"))
                        .unwrap_or_else(|| PathBuf::from("models/funasr"))
                }),
                ..Default::default()
            };

            let mut asr = ASRProcessor::new(config)?;

            // Collect audio files
            let audio_files: Vec<PathBuf> = if input.is_dir() {
                std::fs::read_dir(&input)?
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| {
                        p.extension()
                            .map(|e| e.to_string_lossy().to_lowercase() == "wav")
                            .unwrap_or(false)
                    })
                    .collect()
            } else {
                vec![input.clone()]
            };

            // Transcribe and write output
            let mut results = Vec::new();
            for path in &audio_files {
                match asr.transcribe_file(path) {
                    Ok(transcript) => {
                        results.push((path.clone(), language.clone(), transcript.text));
                    }
                    Err(e) => {
                        tracing::warn!(file = %path.display(), error = %e, "Failed to transcribe");
                    }
                }
            }

            // Write transcript list
            use std::io::Write;
            let mut file = std::fs::File::create(&output)?;
            for (path, lang, text) in &results {
                writeln!(file, "{}|speaker|{}|{}", path.display(), lang, text)?;
            }

            info!(
                files = results.len(),
                output = %output.display(),
                "ASR complete"
            );

            println!("\nASR Summary:");
            println!("============");
            println!("Input: {}", input.display());
            println!("Output: {}", output.display());
            println!("Files transcribed: {}", results.len());
        }

        Commands::Denoise {
            input,
            output,
            strength,
        } => {
            info!("Denoising audio files");

            let config = DenoiseConfig {
                over_subtraction: strength,
                ..Default::default()
            };

            let denoiser = Denoiser::new(config)?;

            if input.is_dir() {
                denoiser.process_directory(&input, &output)?;
            } else {
                std::fs::create_dir_all(&output)?;
                let filename = input.file_name().unwrap();
                denoiser.process_file(&input, output.join(filename))?;
            }

            info!(output_dir = %output.display(), "Denoising complete");

            println!("\nDenoising Summary:");
            println!("==================");
            println!("Input: {}", input.display());
            println!("Output: {}", output.display());
            println!("Strength: {}", strength);
        }

        Commands::Pipeline {
            input,
            output,
            denoise,
            language,
            sample_rate,
        } => {
            info!("Running full preprocessing pipeline");

            let config = PreprocessingConfig {
                slicer: SlicerConfig {
                    sample_rate,
                    ..Default::default()
                },
                asr: ASRConfig {
                    language: language.clone(),
                    ..Default::default()
                },
                denoise: DenoiseConfig::default(),
                enable_denoise: denoise,
                output_sample_rate: sample_rate,
            };

            let mut pipeline = PreprocessingPipeline::new(config)?;

            // Create output subdirectories
            let sliced_dir = output.join("sliced");
            let transcript_file = output.join("transcript.list");

            std::fs::create_dir_all(&output)?;

            // Process
            let results = if input.is_dir() {
                pipeline.process_directory(&input, &sliced_dir)?
            } else {
                pipeline.process_file(&input, &sliced_dir)?
            };

            // Write transcript list
            PreprocessingPipeline::write_transcript_list(&results, &transcript_file)?;

            info!(
                chunks = results.len(),
                output_dir = %output.display(),
                "Pipeline complete"
            );

            println!("\nPipeline Summary:");
            println!("=================");
            println!("Input: {}", input.display());
            println!("Output: {}", output.display());
            println!("Sliced audio: {}", sliced_dir.display());
            println!("Transcript: {}", transcript_file.display());
            println!("Chunks: {}", results.len());
            println!("Denoising: {}", if denoise { "enabled" } else { "disabled" });

            let total_duration: f64 = results.iter()
                .map(|c| (c.end_ms - c.start_ms) as f64 / 1000.0)
                .sum();
            println!("Total duration: {:.1}s", total_duration);
        }
    }

    Ok(())
}
