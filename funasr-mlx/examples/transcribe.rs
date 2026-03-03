//! Simple transcription example with chunking support for long audio.
//!
//! Usage:
//!   cargo run --release --example transcribe -- <audio.wav> <model_dir> [--chunk SECS]

use std::env;
use std::time::Instant;

use funasr_mlx::audio::{load_wav, resample};
use funasr_mlx::{load_model, parse_cmvn_file, transcribe, Vocabulary};
use mlx_rs::module::Module;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <audio.wav> <model_dir> [--chunk SECS]", args[0]);
        std::process::exit(1);
    }

    let audio_path = &args[1];
    let model_dir = &args[2];
    let mut chunk_secs: f32 = 20.0;

    let mut i = 3;
    while i < args.len() {
        if args[i] == "--chunk" && i + 1 < args.len() {
            chunk_secs = args[i + 1].parse().unwrap_or(20.0);
            i += 1;
        }
        i += 1;
    }

    let weights_path = format!("{}/paraformer.safetensors", model_dir);
    let cmvn_path = format!("{}/am.mvn", model_dir);
    let vocab_path = format!("{}/tokens.txt", model_dir);

    // Load audio
    println!("Loading audio: {}", audio_path);
    let (samples, sample_rate) = load_wav(audio_path)?;
    let duration_secs = samples.len() as f32 / sample_rate as f32;
    println!("  {:.1}s ({} samples @ {}Hz)", duration_secs, samples.len(), sample_rate);

    let samples = if sample_rate != 16000 {
        println!("  Resampling to 16kHz...");
        resample(&samples, sample_rate, 16000)
    } else {
        samples
    };

    // Load model
    println!("Loading model from: {}", model_dir);
    let mut model = load_model(&weights_path)?;
    model.training_mode(false);
    let (addshift, rescale) = parse_cmvn_file(&cmvn_path)?;
    model.set_cmvn(addshift, rescale);
    let vocab = Vocabulary::load(&vocab_path)?;
    println!("  {} tokens loaded", vocab.len());

    // Transcribe
    let start = Instant::now();
    let text = if duration_secs > 30.0 {
        let chunk_size = (chunk_secs * 16000.0) as usize;
        let total_chunks = (samples.len() + chunk_size - 1) / chunk_size;
        println!("Using chunked transcription ({:.0}s chunks, {} chunks)...", chunk_secs, total_chunks);

        let mut results: Vec<String> = Vec::new();
        for (i, chunk) in samples.chunks(chunk_size).enumerate() {
            if chunk.len() < 1600 { break; } // skip < 100ms
            eprint!("\r  Chunk {}/{}", i + 1, total_chunks);
            match transcribe(&mut model, chunk, &vocab) {
                Ok(text) if !text.is_empty() => results.push(text),
                Ok(_) => {}
                Err(e) => eprintln!("\n  Chunk {} error: {}", i + 1, e),
            }
        }
        eprintln!();
        results.join("")
    } else {
        transcribe(&mut model, &samples, &vocab)?
    };

    let elapsed = start.elapsed();
    let rtf = elapsed.as_secs_f32() / duration_secs;

    println!("\n=== Results ===");
    println!("Text: {}", text);
    println!("\nAudio: {:.1}s | Time: {:.1}s | {:.1}x realtime | RTF: {:.4}x",
        duration_secs, elapsed.as_secs_f32(), 1.0 / rtf, rtf);

    Ok(())
}
