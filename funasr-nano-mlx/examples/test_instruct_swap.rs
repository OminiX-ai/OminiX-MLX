//! Test swapping Qwen3-0.6B Instruct weights into funasr-nano.
//!
//! This validates whether the Instruct model can:
//! 1. Still perform Chinese ASR (quality check)
//! 2. Follow translation prompts (capability check)
//!
//! Usage:
//!   # First, merge the weights:
//!   python scripts/merge_instruct_weights.py
//!
//!   # Then run this test:
//!   FUNASR_NANO_MODEL_PATH=~/.dora/models/funasr-nano-instruct \
//!     cargo run --release --example test_instruct_swap

use std::env;
use std::time::Instant;

use funasr_nano_mlx::{FunASRNano, TaskPrompt, default_model_path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    // Allow overriding model path via command line
    let model_dir = if args.len() > 1 {
        std::path::PathBuf::from(&args[1])
    } else {
        default_model_path()
    };

    let audio_path = if args.len() > 2 {
        args[2].clone()
    } else {
        model_dir.join("example/zh.wav").to_string_lossy().to_string()
    };

    println!("=== Instruct Model Swap Validation ===\n");
    println!("Model: {}", model_dir.display());
    println!("Audio: {}\n", audio_path);

    // Load model
    println!("Loading model...");
    let start = Instant::now();
    let mut model = FunASRNano::load(&model_dir)?;
    println!("Loaded in {:.2}s\n", start.elapsed().as_secs_f32());

    // Test 1: Standard Chinese ASR
    println!("=== Test 1: Chinese ASR (Baseline) ===");
    let start = Instant::now();
    let chinese = model.transcribe(&audio_path)?;
    let elapsed = start.elapsed().as_millis();
    println!("Output: {}", chinese);
    println!("Time: {}ms", elapsed);
    println!("Status: {}\n", if !chinese.is_empty() && !chinese.contains("!") { "PASS" } else { "FAIL" });

    // Test 2: Text-only Chinese instruction
    println!("=== Test 2: Text Instruction Following ===");
    let start = Instant::now();
    let response = model.generate_from_text(
        "请用一句话介绍自己",
        &funasr_nano_mlx::SamplingConfig::default(),
    )?;
    let elapsed = start.elapsed().as_millis();
    println!("Prompt: 请用一句话介绍自己");
    println!("Output: {}", response);
    println!("Time: {}ms", elapsed);
    let is_garbage = response.chars().all(|c| c == '!' || c == '?' || c.is_whitespace());
    println!("Status: {}\n", if !is_garbage { "PASS" } else { "FAIL" });

    // Test 3: Text translation
    println!("=== Test 3: Text Translation ===");
    let start = Instant::now();
    let translation = model.translate_text("今天天气很好，我们去公园散步吧。")?;
    let elapsed = start.elapsed().as_millis();
    println!("Input: 今天天气很好，我们去公园散步吧。");
    println!("Output: {}", translation);
    println!("Time: {}ms", elapsed);
    // Check if output looks like English
    let has_english = translation.chars().any(|c| c.is_ascii_alphabetic());
    let is_garbage = translation.chars().all(|c| c == '!' || c == '?' || c.is_whitespace());
    println!("Status: {}\n", if has_english && !is_garbage { "PASS" } else { "FAIL" });

    // Test 4: Audio-to-English (direct)
    println!("=== Test 4: Audio-to-English (Direct) ===");
    let start = Instant::now();
    let english = model.transcribe_translate_direct(&audio_path)?;
    let elapsed = start.elapsed().as_millis();
    println!("Output: {}", english);
    println!("Time: {}ms", elapsed);
    let has_english = english.chars().any(|c| c.is_ascii_alphabetic());
    let is_garbage = english.chars().all(|c| c == '!' || c == '?' || c.is_whitespace());
    println!("Status: {}\n", if has_english && !is_garbage { "PASS" } else { "FAIL" });

    // Test 5: Custom prompt
    println!("=== Test 5: Custom Prompt (Correction + Translation) ===");
    let prompt = TaskPrompt::custom(
        "You are a professional speech translator.",
        "Transcribe the speech accurately and provide an English translation:",
    );
    let start = Instant::now();
    let result = model.transcribe_with_prompt(&audio_path, &prompt)?;
    let elapsed = start.elapsed().as_millis();
    println!("Output: {}", result);
    println!("Time: {}ms", elapsed);
    println!();

    // Summary
    println!("=== Summary ===");
    println!("If Tests 1, 2, 3, and 4 all PASS:");
    println!("  -> Instruct swap successful! Proceed to Phase 2A.");
    println!();
    println!("If Test 1 PASS but Tests 2-4 FAIL:");
    println!("  -> Instruct swap broke instruction following.");
    println!("  -> The audio adaptor may not be compatible with Instruct weights.");
    println!("  -> Proceed to Phase 2B (LoRA fine-tuning).");
    println!();
    println!("If Test 1 FAIL:");
    println!("  -> Instruct swap broke ASR capability.");
    println!("  -> Weights may not be compatible. Check merge script.");

    Ok(())
}
