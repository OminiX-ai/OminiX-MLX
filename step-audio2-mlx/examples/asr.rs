//! ASR (Automatic Speech Recognition) Example
//!
//! Transcribes audio files to text using Step-Audio 2 mini.
//!
//! Usage:
//!     cargo run --example asr -- <model_path> <audio_file>
//!
//! Example:
//!     cargo run --example asr -- ./Step-Audio-2-mini ./audio.wav

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use step_audio2_mlx::{StepAudio2, Result, Error};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Step-Audio 2 ASR Example");
        eprintln!("");
        eprintln!("Usage: {} <model_path> <audio_file>", args[0]);
        eprintln!("");
        eprintln!("Arguments:");
        eprintln!("  model_path  Path to Step-Audio-2-mini model directory");
        eprintln!("  audio_file  Path to audio file (WAV format, 16kHz recommended)");
        eprintln!("");
        eprintln!("Example:");
        eprintln!("  {} ./Step-Audio-2-mini ./speech.wav", args[0]);
        return Err(Error::Config("Invalid arguments".into()));
    }

    let model_path = PathBuf::from(&args[1]);
    let audio_path = PathBuf::from(&args[2]);

    // Validate paths
    if !model_path.exists() {
        return Err(Error::ModelLoad(format!(
            "Model path not found: {}",
            model_path.display()
        )));
    }
    if !audio_path.exists() {
        return Err(Error::Audio(format!(
            "Audio file not found: {}",
            audio_path.display()
        )));
    }

    println!("Step-Audio 2 ASR");
    println!("================");
    println!("");
    println!("Model: {}", model_path.display());
    println!("Audio: {}", audio_path.display());
    println!("");

    // Load model
    println!("Loading model...");
    let start = Instant::now();
    let mut model = StepAudio2::load(&model_path)?;
    println!("Model loaded in {:.2}s", start.elapsed().as_secs_f64());
    println!("");

    // Transcribe audio
    println!("Transcribing...");
    let start = Instant::now();
    let text = model.transcribe(&audio_path)?;
    let duration = start.elapsed();
    println!("");

    // Output results
    println!("Transcription:");
    println!("--------------");
    println!("{}", text);
    println!("");
    println!("Time: {:.2}s", duration.as_secs_f64());

    Ok(())
}
