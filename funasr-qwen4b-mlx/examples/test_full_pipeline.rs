//! Test full ASR pipeline: SenseVoice → Adaptor → Qwen3-4B
//!
//! Run: cargo run --example test_full_pipeline --release

use funasr_qwen4b_mlx::sensevoice_encoder::{SenseVoiceEncoder, SenseVoiceEncoderConfig};
use funasr_qwen4b_mlx::adaptor::AudioAdaptorQwen4B;
use funasr_qwen4b_mlx::error::Result;
use mlx_rs::Array;

fn main() -> Result<()> {
    println!("=== Full ASR Pipeline Test ===\n");

    // Paths
    let sensevoice_path = "/Users/yuechen/.dora/models/funasr-nano/model.safetensors";
    let adaptor_path = "adaptor_phase2_final.safetensors";

    // Check paths
    if !std::path::Path::new(sensevoice_path).exists() {
        println!("SenseVoice weights not found: {}", sensevoice_path);
        return Ok(());
    }

    // 1. Load SenseVoice Encoder
    println!("1. Loading SenseVoice Encoder...");
    let mut encoder = SenseVoiceEncoder::new(SenseVoiceEncoderConfig::default())?;
    encoder.load_weights(sensevoice_path)?;
    println!("   SenseVoice encoder loaded (70 layers, 512-dim output)");

    // 2. Load Adaptor
    println!("\n2. Loading Adaptor...");
    let mut adaptor = AudioAdaptorQwen4B::new()?;
    if std::path::Path::new(adaptor_path).exists() {
        adaptor.load_weights(adaptor_path)?;
        println!("   Adaptor loaded (4 layers, 512→2560 dim)");
    } else {
        println!("   Warning: Adaptor weights not found, using random weights");
    }

    // 3. Test Pipeline with dummy audio features
    println!("\n3. Testing Pipeline...");
    let batch_size = 1;
    let seq_len = 100;  // ~1 second of audio after LFR
    let lfr_dim = 560;  // 80 mels * 7 LFR

    // Simulate LFR output (mel spectrogram after LFR transformation)
    let lfr_input = Array::zeros::<f32>(&[batch_size, seq_len, lfr_dim])?;
    println!("   LFR input shape: {:?}", lfr_input.shape());

    // SenseVoice encoding
    let start = std::time::Instant::now();
    let encoder_out = encoder.forward(&lfr_input)?;
    let encoder_time = start.elapsed();
    println!("   Encoder output shape: {:?} ({:.2?})", encoder_out.shape(), encoder_time);

    // Adaptor projection
    let start = std::time::Instant::now();
    let adapted = adaptor.forward(&encoder_out)?;
    let adaptor_time = start.elapsed();
    println!("   Adaptor output shape: {:?} ({:.2?})", adapted.shape(), adaptor_time);

    // Verify dimensions
    let adapted_shape = adapted.shape();
    assert_eq!(adapted_shape[0], batch_size);
    assert_eq!(adapted_shape[1], seq_len);
    assert_eq!(adapted_shape[2], 2560, "Adaptor should output 2560-dim for Qwen3-4B");

    println!("\n   Total pipeline time: {:.2?}", encoder_time + adaptor_time);

    // Check output statistics
    let output_mean = mlx_rs::ops::mean(&adapted, None)?;
    let output_mean_val: f32 = output_mean.item();
    println!("   Adapted features mean: {:.6}", output_mean_val);

    println!("\n=== Pipeline Summary ===");
    println!("Audio → LFR → SenseVoice (512) → Adaptor (2560) → [Ready for Qwen3-4B]");
    println!("\nThe adapted features can now be fed to Qwen3-4B for text generation.");
    println!("Next step: Implement multimodal embedding injection.");

    Ok(())
}
