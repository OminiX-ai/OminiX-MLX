//! Debug TTS pipeline - outputs all intermediate values for comparison with Python

use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

use gpt_sovits_mlx::{
    inference::preprocess_text,
    voice_clone::{VoiceCloner, VoiceClonerConfig},
};
use mlx_rs::transforms::eval;
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let text = "1845年";
    let output_dir = Path::new("/tmp/tts_comparison");
    fs::create_dir_all(output_dir)?;

    println!("Initializing VoiceCloner...");
    let config = VoiceClonerConfig::default();
    let mut cloner = VoiceCloner::new(config)?;

    // Set reference audio
    cloner.set_reference_audio("/Users/yuechen/.OminiX/models/moyoyo/ref_audios/doubao_ref_mix_new.wav")?;

    // Get phonemes using the same preprocessing
    let (phoneme_ids, phonemes, word2ph, norm_text) = preprocess_text(text);
    eval([&phoneme_ids])?;
    let phone_ids: Vec<i32> = phoneme_ids.as_slice().to_vec();

    println!("\n{}", "=".repeat(60));
    println!("RUST TTS OUTPUT SUMMARY");
    println!("{}", "=".repeat(60));
    println!("Text: '{}'", text);
    println!("Normalized: '{}'", norm_text);
    println!("Phones ({}): {:?}", phonemes.len(), phonemes);
    println!("Phone IDs: {:?}", phone_ids);

    // Generate audio
    let audio = cloner.synthesize(text)?;

    // Save audio
    let audio_path = output_dir.join("rust_audio.wav");
    cloner.save_wav(&audio, &audio_path)?;
    println!("Audio saved: {:?}", audio_path);
    println!("Audio: {} samples, {:.2}s", audio.samples.len(), audio.samples.len() as f32 / audio.sample_rate as f32);

    // Save data as JSON
    let result = json!({
        "text": text,
        "norm_text": norm_text,
        "phones": phonemes,
        "phone_ids": phone_ids,
        "word2ph": word2ph,
        "audio_samples": audio.samples.len(),
        "audio_duration": audio.samples.len() as f32 / audio.sample_rate as f32,
    });

    let json_path = output_dir.join("rust_outputs.json");
    let mut file = File::create(&json_path)?;
    file.write_all(serde_json::to_string_pretty(&result)?.as_bytes())?;
    println!("Data saved: {:?}", json_path);
    println!("{}", "=".repeat(60));

    // Play audio
    cloner.play_blocking(&audio)?;

    Ok(())
}
