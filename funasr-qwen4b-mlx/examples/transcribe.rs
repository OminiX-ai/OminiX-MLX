//! Transcribe/translate audio file
//!
//! Supports Chinese transcription (default) and English translation.
//! Automatically uses chunked processing for audio longer than 60 seconds.
//!
//! Run: cargo run --example transcribe --release -- path/to/audio.wav
//!      cargo run --example transcribe --release -- path/to/audio.wav --lang en
//!      cargo run --example transcribe --release -- path/to/audio.wav --robust
//!      cargo run --example transcribe --release -- path/to/audio.wav --greedy

use funasr_qwen4b_mlx::{FunASRQwen4B, TranscribeConfig};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        println!("Usage: {} <audio.wav> [options]", args[0]);
        println!("\nOptions:");
        println!("  --lang <zh|en>    Language mode (default: zh)");
        println!("  --model <dir>     Model directory (default: .)");
        println!("  --chunk <secs>    Chunk size for long audio (default: 30)");
        println!("  --greedy          Use greedy decoding (best CER, may repeat)");
        println!("  --robust          Use robust config (default: sampling + VAD + overlap)");
        println!("  --temp <float>    Sampling temperature (default: 0.6)");
        println!("  --penalty <float> Presence penalty (default: 1.0)");
        println!("\nExamples:");
        println!("  {} test.wav", args[0]);
        println!("  {} english_talk.wav --lang en", args[0]);
        println!("  {} long_audio.wav --chunk 20 --robust", args[0]);
        println!("  {} benchmark.wav --greedy", args[0]);
        return Ok(());
    }

    let audio_path = &args[1];

    // Parse options
    let mut lang = "zh";
    let mut model_dir = ".";
    let mut chunk_secs: f32 = 30.0;
    let mut config = TranscribeConfig::greedy(); // backward-compatible default
    let mut explicit_config = false;

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
            "--greedy" => {
                config = TranscribeConfig::greedy();
                explicit_config = true;
            }
            "--robust" => {
                config = TranscribeConfig::default();
                explicit_config = true;
            }
            "--temp" | "--temperature" => {
                if i + 1 < args.len() {
                    i += 1;
                    config.temperature = args[i].parse().unwrap_or(0.6);
                    if !explicit_config {
                        // If setting temp, switch to robust base
                        config = TranscribeConfig::default();
                        config.temperature = args[i].parse().unwrap_or(0.6);
                    }
                }
            }
            "--penalty" => {
                if i + 1 < args.len() {
                    i += 1;
                    config.presence_penalty = args[i].parse().unwrap_or(1.0);
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
        "mix" => "Mixed Chinese+English transcription",
        _ => "Chinese transcription",
    };
    let config_desc = if config.temperature == 0.0 { "greedy" } else { "robust (sampling)" };
    println!("Mode: {} | Config: {} (temp={}, penalty={})", mode_desc, config_desc, config.temperature, config.presence_penalty);
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
                model.translate_long_samples_with_config(&samples, sample_rate, chunk_secs, &config)?
            } else {
                model.transcribe_samples_with_config(&samples, sample_rate, "Translate the speech to English:", &config)?
            }
        }
        "mix" => {
            let prompt = "语音转写成中文：";
            model.transcribe_samples_with_config(&samples, sample_rate, prompt, &config)?
        }
        _ => {
            if duration_secs > 60.0 {
                println!("Using chunked Chinese transcription ({:.0}s chunks)...", chunk_secs);
                model.transcribe_long_samples_with_config(&samples, sample_rate, chunk_secs, &config)?
            } else {
                model.transcribe_samples_with_config(&samples, sample_rate, "语音转写成中文：", &config)?
            }
        }
    };
    let elapsed = start.elapsed();

    println!("\nResult:\n{}", text);
    println!("\nTime: {:.2?} ({:.1}x realtime)", elapsed, duration_secs / elapsed.as_secs_f32());

    Ok(())
}
