//! Transcribe audio using Fun-ASR-Nano.
//!
//! Usage:
//!   cargo run --release --example transcribe [model_dir] <audio_path>
//!
//! Model path resolution:
//!   1. Command line argument (if provided)
//!   2. FUNASR_NANO_MODEL_PATH environment variable
//!   3. ~/.OminiX/models/funasr-nano (default)

use funasr_nano_mlx::{FunASRNano, default_model_path};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse arguments: [model_dir] <audio_path>
    let (model_dir, audio_path) = match args.len() {
        1 => {
            let model = default_model_path();
            let audio = model.join("example/zh.wav");
            (model, audio)
        }
        2 => {
            // Single arg: audio path, use default model
            (default_model_path(), std::path::PathBuf::from(&args[1]))
        }
        _ => {
            // Two args: model_dir and audio_path
            (std::path::PathBuf::from(&args[1]), std::path::PathBuf::from(&args[2]))
        }
    };

    println!("Loading model from {}...", model_dir.display());
    let start = Instant::now();
    let mut model = FunASRNano::load(&model_dir).expect("Failed to load model");
    println!("Model loaded in {:.2}s\n", start.elapsed().as_secs_f32());

    println!("Transcribing {}...", audio_path.display());
    let start = Instant::now();
    match model.transcribe(&audio_path) {
        Ok(text) => {
            let elapsed = start.elapsed().as_secs_f32();
            println!("\nTranscription ({:.2}s):", elapsed);
            println!("{}", text);
        }
        Err(e) => {
            eprintln!("Transcription failed: {}", e);
            std::process::exit(1);
        }
    }
}
