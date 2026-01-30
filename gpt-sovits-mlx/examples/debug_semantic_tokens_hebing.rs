//! Debug semantic token generation for "合并" cases
//!
//! This script compares the semantic tokens generated for working vs failing cases.

use std::path::Path;
use gpt_sovits_mlx::voice_clone::{VoiceCloner, VoiceClonerConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading voice cloner...");

    let config = VoiceClonerConfig {
        top_k: 15,
        top_p: 1.0,
        temperature: 1.0,
        repetition_penalty: 1.35,
        ..Default::default()
    };

    let mut cloner = VoiceCloner::new(config)?;

    // Reference audio is required for voice cloning
    let ref_audio_path = "~/.dora/models/primespeech/gpt-sovits-mlx/doubao_ref.wav";
    let ref_text = "今天天气真好啊！";

    if !Path::new(ref_audio_path).exists() {
        println!("ERROR: Reference audio not found at {}", ref_audio_path);
        println!("Please provide a reference audio file for voice cloning.");
        return Err("Reference audio not found".into());
    }

    println!("Setting reference audio for few-shot mode...");
    cloner.set_reference_audio_with_text(ref_audio_path, ref_text)?;
    let few_shot = true;

    // Test cases
    let test_cases = [
        ("合并", "WORKING - expected he2 bing4"),
        ("合并两个文件", "WORKING - expected he2 bing4 liang3 ge5 wen2 jian4"),
        ("合并，并将", "FAILING - sounds like he1 bing4 instead of he2 bing4"),
    ];

    for (text, status) in &test_cases {
        println!("\n{}", "=".repeat(60));
        println!("Text: {} - {}", text, status);
        println!("{}", "=".repeat(60));

        // Generate audio (this will print debug info including phonemes)
        let audio = cloner.synthesize(text)?;

        println!("\nGeneration stats:");
        println!("  Duration: {:.2}s", audio.duration);
        println!("  Tokens: {}", audio.num_tokens);
        if audio.duration > 0.0 {
            println!("  Tokens/sec: {:.1}", audio.num_tokens as f32 / audio.duration);
        }

        // Save the audio for listening comparison
        let filename = format!("/tmp/hebing_debug_{}.wav",
            text.replace("，", "_comma_").replace("。", "_period_"));
        cloner.save_wav(&audio, &filename)?;
        println!("  Saved: {}", filename);
    }

    println!("\n{}", "=".repeat(60));
    println!("COMPARISON INSTRUCTIONS");
    println!("{}", "=".repeat(60));
    println!();
    println!("Listen to the generated WAV files:");
    println!("  /tmp/hebing_debug_合并.wav");
    println!("  /tmp/hebing_debug_合并两个文件.wav");
    println!("  /tmp/hebing_debug_合并_comma_并将.wav");
    println!();
    println!("Compare the pronunciation of '合' (hé) in each file.");
    println!("If the first two sound like tone 2 (rising) but the third");
    println!("sounds like tone 1 (level), the issue is in T2S generation.");
    println!();
    if few_shot {
        println!("Mode: Few-shot (with reference audio)");
    } else {
        println!("Mode: Zero-shot (no reference audio)");
    }

    Ok(())
}
