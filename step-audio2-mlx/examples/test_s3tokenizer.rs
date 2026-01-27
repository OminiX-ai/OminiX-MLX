//! Test MLX-accelerated S3Tokenizer
//!
//! Converts mel spectrogram to discrete audio codes using pure MLX acceleration.
//!
//! Usage:
//!     cargo run --release --features tts --example test_s3tokenizer

use std::path::PathBuf;
use std::time::Instant;

use step_audio2_mlx::audio::{load_wav, samples_to_mel};
use step_audio2_mlx::config::AudioConfig;
use step_audio2_mlx::tts::S3TokenizerMLX;
use step_audio2_mlx::{Error, Result};

fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║           MLX-ACCELERATED S3TOKENIZER TEST                           ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    // Find model directory
    let model_dirs = [
        "./Step-Audio-2-mini",
        "../Step-Audio-2-mini",
        "step-audio2-mlx/Step-Audio-2-mini",
    ];

    let model_dir = model_dirs
        .iter()
        .map(PathBuf::from)
        .find(|p| p.exists())
        .ok_or_else(|| Error::Config("Step-Audio-2-mini model not found".into()))?;

    // Check for converted weights
    let weights_path = model_dir.join("tts_mlx").join("s3tokenizer.safetensors");
    if !weights_path.exists() {
        println!("S3Tokenizer weights not found at {:?}", weights_path);
        println!("Run: python scripts/convert_s3tokenizer.py");
        return Err(Error::ModelLoad("S3Tokenizer weights not found".into()));
    }

    // Load S3Tokenizer
    println!("Loading S3Tokenizer from {}...", model_dir.display());
    let load_start = Instant::now();
    let mut tokenizer = S3TokenizerMLX::load(&model_dir)?;
    let load_time = load_start.elapsed().as_secs_f64();
    println!("  Loaded in {:.2}s", load_time);
    println!("  Frame rate: {} Hz", tokenizer.frame_rate());
    println!("  Codebook size: {}", tokenizer.codebook_size());
    println!();

    // Create test audio (or load from file if available)
    let test_audio_path = PathBuf::from("./test_audio.wav");
    let (samples, sample_rate) = if test_audio_path.exists() {
        println!("Loading test audio from {:?}...", test_audio_path);
        load_wav(&test_audio_path)?
    } else {
        // Generate synthetic audio
        println!("Generating synthetic test audio (2s @ 16kHz)...");
        let sample_rate = 16000u32;
        let duration = 2.0f32;
        let num_samples = (sample_rate as f32 * duration) as usize;

        let mut samples = Vec::with_capacity(num_samples);
        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            // Mix of frequencies to create interesting patterns
            let sample = 0.3 * (440.0 * 2.0 * std::f32::consts::PI * t).sin()
                + 0.2 * (880.0 * 2.0 * std::f32::consts::PI * t).sin()
                + 0.1 * (220.0 * 2.0 * std::f32::consts::PI * t).sin();
            samples.push(sample);
        }
        (samples, sample_rate)
    };

    let audio_duration = samples.len() as f32 / sample_rate as f32;
    println!("  Duration: {:.2}s at {}Hz", audio_duration, sample_rate);
    println!();

    // Convert to mel spectrogram
    println!("Computing mel spectrogram...");
    let mel_start = Instant::now();
    let audio_config = AudioConfig::default();
    let mel = samples_to_mel(&samples, sample_rate, &audio_config)?;
    let mel_time = mel_start.elapsed().as_secs_f64();
    println!("  Mel shape: {:?}", mel.shape());
    println!("  Mel time: {:.3}s", mel_time);
    println!();

    // Encode to codes
    println!("Encoding mel to audio codes...");
    let encode_start = Instant::now();
    let codes = tokenizer.encode(&mel)?;
    // Force evaluation by reading data (MLX is lazy)
    let _ = codes.shape();
    mlx_rs::transforms::eval([&codes]).map_err(|e| Error::Mlx(e))?;
    let encode_time = encode_start.elapsed().as_secs_f64();

    let codes_shape = codes.shape();
    let num_codes = codes_shape.iter().product::<i32>() as usize;
    let codes_duration = num_codes as f32 / tokenizer.frame_rate() as f32;

    println!("  Codes shape: {:?}", codes_shape);
    println!("  Num codes: {}", num_codes);
    println!("  Codes duration: {:.2}s @ {} Hz", codes_duration, tokenizer.frame_rate());
    println!("  Encode time: {:.3}s", encode_time);
    println!();

    // Print some sample codes
    let codes_vec: Vec<i32> = codes.as_slice::<i32>().to_vec();
    println!("Sample codes (first 20):");
    for (i, code) in codes_vec.iter().take(20).enumerate() {
        print!("{:4} ", code);
        if (i + 1) % 10 == 0 {
            println!();
        }
    }
    if codes_vec.len() > 20 {
        println!("...");
    }
    println!();

    // Summary
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                          SUMMARY                                     ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║ Input audio:      {:.2}s @ {}Hz", audio_duration, sample_rate);
    println!("║ Mel spectrogram:  {:?}", mel.shape());
    println!("║ Audio codes:      {} codes @ {} Hz", num_codes, tokenizer.frame_rate());
    println!("║ Encoding time:    {:.3}s ({:.1}x real-time)", encode_time, encode_time / audio_duration as f64);
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    Ok(())
}
