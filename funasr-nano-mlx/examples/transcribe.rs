//! Transcribe audio using Fun-ASR-Nano.

use funasr_nano_mlx::model::FunASRNano;
use std::time::Instant;

fn main() {
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "./Fun-ASR-Nano-2512".to_string());

    let audio_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "./Fun-ASR-Nano-2512/example/zh.wav".to_string());

    println!("Loading model from {}...", model_dir);
    let start = Instant::now();
    let mut model = FunASRNano::load(&model_dir).expect("Failed to load model");
    println!("Model loaded in {:.2}s\n", start.elapsed().as_secs_f32());

    println!("Transcribing {}...", audio_path);
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
