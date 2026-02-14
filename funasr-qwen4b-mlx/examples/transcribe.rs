//! Transcribe/translate audio file
//!
//! Supports Chinese transcription (default) and English translation.
//! Automatically uses chunked processing for audio longer than 60 seconds.
//!
//! Run: cargo run --example transcribe --release -- path/to/audio.wav
//!      cargo run --example transcribe --release -- path/to/audio.wav --lang en

use funasr_qwen4b_mlx::FunASRQwen4B;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        println!("Usage: {} <audio.wav> [options]", args[0]);
        println!("\nOptions:");
        println!("  --lang <zh|en>    Language mode (default: zh)");
        println!("  --model <dir>     Model directory (default: .)");
        println!("  --chunk <secs>    Chunk size for long audio (default: 30)");
        println!("\nExamples:");
        println!("  {} test.wav", args[0]);
        println!("  {} english_talk.wav --lang en", args[0]);
        println!("  {} long_audio.wav --chunk 20", args[0]);
        return Ok(());
    }

    let audio_path = &args[1];

    // Parse options
    let mut lang = "zh";
    let mut model_dir = ".";
    let mut chunk_secs: f32 = 30.0;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--lang" => {
                lang = if i + 1 < args.len() { i += 1; &args[i] } else { "zh" };
            }
            "--model" => {
                model_dir = if i + 1 < args.len() { i += 1; &args[i] } else { "." };
            }
            "--chunk" => {
                if i + 1 < args.len() {
                    i += 1;
                    chunk_secs = args[i].parse().unwrap_or(30.0);
                }
            }
            _ => {
                // Legacy positional args: model_dir chunk_secs
                if i == 2 && !args[i].starts_with("--") {
                    model_dir = &args[i];
                } else if i == 3 && !args[i].starts_with("--") {
                    chunk_secs = args[i].parse().unwrap_or(30.0);
                }
            }
        }
        i += 1;
    }

    let mode_desc = match lang {
        "en" => "English translation",
        _ => "Chinese transcription",
    };
    println!("Mode: {}", mode_desc);
    println!("Loading FunASR-Qwen4B model from {}...", model_dir);
    let mut model = FunASRQwen4B::load(model_dir)?;
    println!("Model loaded.\n");

    // Check audio duration to decide single-pass vs chunked
    let (samples, sample_rate) = funasr_qwen4b_mlx::audio::load_wav(audio_path)?;
    let duration_secs = samples.len() as f32 / sample_rate as f32;
    println!("Audio: {:.1}s ({} samples @ {}Hz)", duration_secs, samples.len(), sample_rate);

    let start = std::time::Instant::now();
    let text = match lang {
        "en" => {
            if duration_secs > 60.0 {
                println!("Using chunked English translation ({:.0}s chunks)...", chunk_secs);
                model.translate_long_samples(&samples, sample_rate, chunk_secs)?
            } else {
                model.translate_samples_to_english(&samples, sample_rate)?
            }
        }
        _ => {
            if duration_secs > 60.0 {
                println!("Using chunked Chinese transcription ({:.0}s chunks)...", chunk_secs);
                model.transcribe_long_samples(&samples, sample_rate, chunk_secs)?
            } else {
                model.transcribe_samples(&samples, sample_rate)?
            }
        }
    };
    let elapsed = start.elapsed();

    println!("\nResult:\n{}", text);
    println!("\nTime: {:.2?} ({:.1}x realtime)", elapsed, duration_secs / elapsed.as_secs_f32());

    Ok(())
}
