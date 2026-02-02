//! Transcribe audio and translate to English
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
    let model_dir = args.get(2).map(|s| s.as_str()).unwrap_or("models");

    println!("Loading FunASR-Qwen4B model from {}...", model_dir);
    let mut model = FunASRQwen4B::load(model_dir)?;
    println!("Model loaded.\n");

    println!("Transcribing and translating: {}", audio_path);
    let start = std::time::Instant::now();
    let (chinese, english) = model.transcribe_and_translate(audio_path)?;
    let elapsed = start.elapsed();

    println!("\nChinese: {}", chinese);
    println!("English: {}", english);
    println!("\nTime: {:.2?}", elapsed);

    Ok(())
}
