//! Think Mode Example
//!
//! Demonstrates Step-Audio 2 mini-Think's reasoning capabilities.
//! The model first thinks through the problem in <think>...</think> tags
//! before providing a response.
//!
//! Usage:
//!     cargo run --example think -- <model_path> <audio_file>
//!
//! Example:
//!     cargo run --example think -- ./Step-Audio-2-mini ./question.wav

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use step_audio2_mlx::{StepAudio2, ThinkConfig, Result, Error};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Step-Audio 2 Think Mode Example");
        eprintln!("");
        eprintln!("Usage: {} <model_path> <audio_file>", args[0]);
        eprintln!("");
        eprintln!("Arguments:");
        eprintln!("  model_path  Path to Step-Audio-2-mini-Think model directory");
        eprintln!("  audio_file  Path to audio file with question (WAV format)");
        eprintln!("");
        eprintln!("Example:");
        eprintln!("  {} ./Step-Audio-2-mini-Think ./question.wav", args[0]);
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

    println!("Step-Audio 2 Think Mode");
    println!("=======================");
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

    // Configure think mode
    let think_config = ThinkConfig::default();
    println!("Think mode enabled:");
    println!("  Max thinking tokens: {}", think_config.max_think_tokens);
    println!("  Max response tokens: {}", think_config.max_response_tokens);
    println!("");

    // Process with think mode
    println!("Processing with think mode...");
    let start = Instant::now();
    let output = model.think_and_respond(&audio_path, think_config)?;
    let duration = start.elapsed();
    println!("");

    // Output results
    if let Some(thinking) = &output.thinking {
        println!("Thinking:");
        println!("---------");
        println!("{}", thinking);
        println!("");
    }

    println!("Response:");
    println!("---------");
    println!("{}", output.response_text);
    println!("");

    println!("Statistics:");
    println!("-----------");
    println!("  Think tokens:    {}", output.think_tokens);
    println!("  Response tokens: {}", output.response_tokens);
    println!("  Total tokens:    {}", output.total_tokens);
    println!("  Time:            {:.2}s", duration.as_secs_f64());
    if output.total_tokens > 0 {
        println!(
            "  Speed:           {:.1} tokens/s",
            output.total_tokens as f64 / duration.as_secs_f64()
        );
    }

    Ok(())
}
