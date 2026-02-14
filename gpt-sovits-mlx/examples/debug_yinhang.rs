//! Debug semantic tokens for "银行公报" to investigate pronunciation issue
//!
//! Run with: cargo run --release --example debug_yinhang

use gpt_sovits_mlx::voice_clone::{VoiceCloner, VoiceClonerConfig};
use gpt_sovits_mlx::audio::save_wav;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Debug 银行公报 Pronunciation ===\n");

    // Test just "银行公报" to isolate the issue
    let test_text = "银行公报";

    // Initialize with doubao model
    let config = VoiceClonerConfig {
        model_path: "/Users/yuechen/.OminiX/models/moyoyo".to_string(),
        voice_name: "doubao".to_string(),
        debug: true,
        ..Default::default()
    };

    let start = Instant::now();
    let mut cloner = VoiceCloner::new(config)?;
    println!("Initialized in {:?}\n", start.elapsed());

    // Load reference
    let ref_audio = "/Users/yuechen/.OminiX/models/moyoyo/ref_audios/doubao_ref_mix_new.wav";
    let ref_text = "这家resturant的steak很有名，但是vegetable salad的price有点贵";
    cloner.load_reference_with_python_codes(ref_audio, ref_text, None)?;

    // Generate with debug info
    println!("Generating for: {}\n", test_text);
    let audio = cloner.generate(test_text)?;

    println!("\nGenerated {} samples ({:.2}s at 32kHz)",
             audio.len(),
             audio.len() as f32 / 32000.0);

    // Save to file
    let output_path = "/tmp/debug_yinhang.wav";
    save_wav(&audio, 32000, output_path)?;

    println!("\nSaved to: {}", output_path);

    // Play audio
    println!("\n▶️  Playing...\n");
    std::process::Command::new("afplay")
        .arg(output_path)
        .status()?;

    Ok(())
}
