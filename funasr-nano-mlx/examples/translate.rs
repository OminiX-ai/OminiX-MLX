//! Test translation capabilities of FunASR-Nano's integrated LLM.
//!
//! This example demonstrates:
//! 1. Two-pass ASR + translation (transcribe, then translate)
//! 2. Direct audio-to-English with modified prompt
//! 3. Text-only translation using the LLM
//!
//! Usage:
//!   cargo run --release --example translate [audio.wav]

use std::time::Instant;
use funasr_nano_mlx::{FunASRNano, TaskPrompt, default_model_path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    let model_dir = default_model_path();
    let default_audio = model_dir.join("example/zh.wav");
    let audio_path = args.get(1)
        .map(|s| s.to_string())
        .unwrap_or_else(|| default_audio.to_string_lossy().to_string());

    println!("Loading model from {}...", model_dir.display());
    let start = Instant::now();
    let mut model = FunASRNano::load(&model_dir)?;
    println!("Model loaded in {:.2}s\n", start.elapsed().as_secs_f32());

    println!("Audio: {}\n", audio_path);

    // =========================================================
    // Test 1: Standard transcription (baseline)
    // =========================================================
    println!("=== Test 1: Standard Chinese Transcription ===");
    let start = Instant::now();
    let transcription = model.transcribe(&audio_path)?;
    let elapsed = start.elapsed().as_millis();
    println!("Result: {}", transcription);
    println!("Time: {}ms\n", elapsed);

    // =========================================================
    // Test 2: Two-pass transcribe + translate
    // =========================================================
    println!("=== Test 2: Two-Pass (ASR + Translation) ===");
    let start = Instant::now();
    let (chinese, english) = model.transcribe_and_translate(&audio_path)?;
    let elapsed = start.elapsed().as_millis();
    println!("Chinese: {}", chinese);
    println!("English: {}", english);
    println!("Time: {}ms\n", elapsed);

    // =========================================================
    // Test 3: Direct audio-to-English (single pass)
    // =========================================================
    println!("=== Test 3: Direct Audio-to-English ===");
    let start = Instant::now();
    let direct_english = model.transcribe_translate_direct(&audio_path)?;
    let elapsed = start.elapsed().as_millis();
    println!("Result: {}", direct_english);
    println!("Time: {}ms\n", elapsed);

    // =========================================================
    // Test 4: Text-only translation
    // =========================================================
    println!("=== Test 4: Text-Only Translation ===");
    let test_text = "今天天气很好，我们去公园散步吧。";
    println!("Input: {}", test_text);
    let start = Instant::now();
    let translation = model.translate_text(test_text)?;
    let elapsed = start.elapsed().as_millis();
    println!("Output: {}", translation);
    println!("Time: {}ms\n", elapsed);

    // =========================================================
    // Test 5: Custom prompt
    // =========================================================
    println!("=== Test 5: Custom Prompt ===");
    let custom_prompt = TaskPrompt::custom(
        "You are a professional translator.",
        "Transcribe and translate to English with natural phrasing:",
    );
    let start = Instant::now();
    let custom_result = model.transcribe_with_prompt(&audio_path, &custom_prompt)?;
    let elapsed = start.elapsed().as_millis();
    println!("Result: {}", custom_result);
    println!("Time: {}ms\n", elapsed);

    // =========================================================
    // Summary
    // =========================================================
    println!("=== Summary ===");
    println!("Note: Qwen3-0.6B is a small model (620M params) optimized for ASR.");
    println!("Translation quality may be limited. For production, consider:");
    println!("1. Using a larger translation LLM in the pipeline");
    println!("2. Fine-tuning Qwen3 on translation data");
    println!("3. Using the two-pass approach with error correction");

    Ok(())
}
