//! Transcribe audio using Fun-ASR-Nano.
//!
//! Usage:
//!   cargo run --release --example transcribe [model_dir] <audio_path> [--chunk SECS]
//!
//! Model path resolution:
//!   1. Command line argument (if provided)
//!   2. FUNASR_NANO_MODEL_PATH environment variable
//!   3. ~/.OminiX/models/funasr-nano (default)

use funasr_nano_mlx::{FunASRNano, default_model_path};
use funasr_nano_mlx::audio;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
    let mut model_dir = None;
    let mut audio_path = None;
    let mut chunk_secs: f32 = 20.0;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--chunk" => {
                i += 1;
                if i < args.len() {
                    chunk_secs = args[i].parse().unwrap_or(20.0);
                }
            }
            s if !s.starts_with("--") => {
                if audio_path.is_some() {
                    // Third positional = ignore
                } else if model_dir.is_some() {
                    audio_path = Some(std::path::PathBuf::from(s));
                } else {
                    // Could be model_dir or audio_path
                    // If it looks like an audio file, treat as audio_path
                    if s.ends_with(".wav") || s.ends_with(".m4a") || s.ends_with(".mp3") || s.ends_with(".flac") {
                        audio_path = Some(std::path::PathBuf::from(s));
                    } else {
                        model_dir = Some(std::path::PathBuf::from(s));
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }

    let model_dir = model_dir.unwrap_or_else(default_model_path);
    let audio_path = audio_path.unwrap_or_else(|| model_dir.join("example/zh.wav"));

    println!("Loading model from {}...", model_dir.display());
    let start = Instant::now();
    let mut model = FunASRNano::load(&model_dir).expect("Failed to load model");
    println!("Model loaded in {:.2}s\n", start.elapsed().as_secs_f32());

    // Load audio
    let (samples, sample_rate) = audio::load_wav(&audio_path).expect("Failed to load audio");
    let duration_secs = samples.len() as f32 / sample_rate as f32;
    println!("Audio: {:.1}s ({} samples @ {}Hz)", duration_secs, samples.len(), sample_rate);

    let start = Instant::now();

    let text = if duration_secs > 30.0 {
        // Chunked processing for long audio
        let chunk_size = (chunk_secs * sample_rate as f32) as usize;
        let total_chunks = (samples.len() + chunk_size - 1) / chunk_size;
        println!("Using chunked transcription ({:.0}s chunks, {} chunks)...", chunk_secs, total_chunks);

        let mut results: Vec<String> = Vec::new();
        for (i, chunk) in samples.chunks(chunk_size).enumerate() {
            if chunk.len() < (sample_rate as usize / 10) {
                break; // skip chunks shorter than 100ms
            }
            eprint!("\r  Chunk {}/{}", i + 1, total_chunks);
            match model.transcribe_samples(chunk, sample_rate) {
                Ok(text) if !text.is_empty() => results.push(text),
                Ok(_) => {} // empty result
                Err(e) => eprintln!("\n  Chunk {} error: {}", i + 1, e),
            }
        }
        eprintln!();
        results.join("")
    } else {
        model.transcribe_samples(&samples, sample_rate)
            .expect("Transcription failed")
    };

    let elapsed = start.elapsed().as_secs_f32();
    println!("\nTranscription ({:.2}s, {:.1}x realtime):", elapsed, duration_secs / elapsed);
    println!("{}", text);
}
