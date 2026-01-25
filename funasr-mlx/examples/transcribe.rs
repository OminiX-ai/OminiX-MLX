//! Simple transcription example
//!
//! Usage:
//!   cargo run --release --example transcribe -- <audio.wav> <model_dir>
//!
//! Example:
//!   cargo run --release --example transcribe -- test.wav /path/to/paraformer

use std::env;
use std::time::Instant;

use funasr_mlx::audio::{load_wav, resample};
use funasr_mlx::{load_model, parse_cmvn_file, transcribe, Vocabulary};
use mlx_rs::module::Module;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <audio.wav> <model_dir>", args[0]);
        eprintln!();
        eprintln!("Arguments:");
        eprintln!("  audio.wav  - Input audio file (16kHz mono, or will be resampled)");
        eprintln!("  model_dir  - Directory containing:");
        eprintln!("               - paraformer.safetensors (model weights)");
        eprintln!("               - am.mvn (CMVN normalization)");
        eprintln!("               - tokens.txt (vocabulary)");
        std::process::exit(1);
    }

    let audio_path = &args[1];
    let model_dir = &args[2];

    // Construct paths
    let weights_path = format!("{}/paraformer.safetensors", model_dir);
    let cmvn_path = format!("{}/am.mvn", model_dir);
    let vocab_path = format!("{}/tokens.txt", model_dir);

    // Load audio
    println!("Loading audio: {}", audio_path);
    let (samples, sample_rate) = load_wav(audio_path)?;
    let duration_secs = samples.len() as f32 / sample_rate as f32;
    println!(
        "  {} samples, {} Hz, {:.2}s",
        samples.len(),
        sample_rate,
        duration_secs
    );

    // Resample to 16kHz if needed
    let samples = if sample_rate != 16000 {
        println!("  Resampling to 16kHz...");
        resample(&samples, sample_rate, 16000)
    } else {
        samples
    };

    // Load model
    println!("\nLoading model from: {}", model_dir);
    let mut model = load_model(&weights_path)?;
    model.training_mode(false);

    // Load CMVN
    let (addshift, rescale) = parse_cmvn_file(&cmvn_path)?;
    model.set_cmvn(addshift, rescale);

    // Load vocabulary
    let vocab = Vocabulary::load(&vocab_path)?;
    println!("  {} tokens loaded", vocab.len());

    // Transcribe
    println!("\nTranscribing...");
    let start = Instant::now();
    let text = transcribe(&mut model, &samples, &vocab)?;
    let elapsed = start.elapsed();

    // Calculate metrics
    let inference_ms = elapsed.as_millis();
    let rtf = (inference_ms as f32 / 1000.0) / duration_secs;

    println!("\n=== Results ===");
    println!("Text: {}", text);
    println!();
    println!("Performance:");
    println!("  Audio duration: {:.2}s", duration_secs);
    println!("  Inference time: {} ms", inference_ms);
    println!("  RTF: {:.4}x", rtf);
    println!("  Speed: {:.1}x real-time", 1.0 / rtf);

    Ok(())
}
