//! Transcribe audio and translate to English
//!
//! Tests three modes:
//! 1. Chinese transcription (baseline)
//! 2. Direct audio-to-English translation (single pass)
//! 3. Two-step: transcribe Chinese, then translate (existing pipeline)
//!
//! Run: cargo run --example transcribe_translate --release -- path/to/audio.wav

use funasr_qwen4b_mlx::FunASRQwen4B;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        println!("Usage: {} <audio.wav> [model_dir]", args[0]);
        println!("\nExample:");
        println!("  {} test.wav", args[0]);
        println!("  {} test.wav ./models", args[0]);
        return Ok(());
    }

    let audio_path = &args[1];
    let model_dir = args.get(2).map(|s| s.as_str()).unwrap_or(".");

    println!("Loading FunASR-Qwen4B model from {}...", model_dir);
    let mut model = FunASRQwen4B::load(model_dir)?;
    println!("Model loaded.\n");

    // 1. Chinese transcription
    println!("=== 1. Chinese Transcription ===");
    let start = std::time::Instant::now();
    let chinese = model.transcribe(audio_path)?;
    let t1 = start.elapsed();
    println!("  Result: {}", chinese);
    println!("  Time: {:.2?}\n", t1);

    // 2. Direct audio-to-English (single pass)
    println!("=== 2. Direct Audio-to-English ===");
    let start = std::time::Instant::now();
    let direct_english = model.translate_audio_to_english(audio_path)?;
    let t2 = start.elapsed();
    println!("  Result: {}", direct_english);
    println!("  Time: {:.2?}\n", t2);

    // 3. Two-step translation (Chinese → English via text)
    println!("=== 3. Two-Step Translation ===");
    let start = std::time::Instant::now();
    let english = model.translate(&chinese)?;
    let t3 = start.elapsed();
    println!("  Result: {}", english);
    println!("  Time: {:.2?}\n", t3);

    println!("=== Summary ===");
    println!("  Chinese:         {}", chinese);
    println!("  Direct English:  {}", direct_english);
    println!("  2-Step English:  {}", english);

    Ok(())
}
