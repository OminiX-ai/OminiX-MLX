//! Test Full TTS Pipeline
//!
//! Tests each component of the TTS pipeline:
//! 1. S3Tokenizer: mel → audio codes
//! 2. Flow Decoder: audio codes → mel spectrogram
//! 3. HiFi-GAN: mel spectrogram → audio waveform
//!
//! Usage:
//!     cargo run --release --features tts --example test_tts_pipeline

use std::path::PathBuf;
use std::time::Instant;

use step_audio2_mlx::audio::{load_wav, save_wav, samples_to_mel};
use step_audio2_mlx::config::AudioConfig;
use step_audio2_mlx::tts::{S3TokenizerMLX, FlowDecoder, HiFiGAN};
use step_audio2_mlx::{Error, Result};

fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║              FULL TTS PIPELINE TEST                                  ║");
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

    println!("Model directory: {}", model_dir.display());
    println!();

    // =========================================================================
    // Step 1: Load test audio and compute mel spectrogram
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("STEP 1: Load Audio & Compute Mel Spectrogram");
    println!("═══════════════════════════════════════════════════════════════════════");

    let test_audio_path = PathBuf::from("./test_audio.wav");
    let (samples, sample_rate) = if test_audio_path.exists() {
        println!("Loading audio from {:?}...", test_audio_path);
        load_wav(&test_audio_path)?
    } else {
        println!("Generating synthetic test audio (1s @ 16kHz)...");
        let sample_rate = 16000u32;
        let duration = 1.0f32;
        let num_samples = (sample_rate as f32 * duration) as usize;

        let mut samples = Vec::with_capacity(num_samples);
        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            let sample = 0.5 * (440.0 * 2.0 * std::f32::consts::PI * t).sin();
            samples.push(sample);
        }
        (samples, sample_rate)
    };

    let audio_duration = samples.len() as f32 / sample_rate as f32;
    println!("  Audio: {:.2}s @ {}Hz", audio_duration, sample_rate);

    let audio_config = AudioConfig::default();
    let mel = samples_to_mel(&samples, sample_rate, &audio_config)?;
    println!("  Mel shape: {:?}", mel.shape());
    println!();

    // =========================================================================
    // Step 2: S3Tokenizer - Mel → Audio Codes
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("STEP 2: S3Tokenizer (Mel → Audio Codes)");
    println!("═══════════════════════════════════════════════════════════════════════");

    let s3_weights = model_dir.join("tts_mlx").join("s3tokenizer.safetensors");
    if !s3_weights.exists() {
        println!("  ⚠️  S3Tokenizer weights not found. Run: python scripts/convert_s3tokenizer.py");
        return Err(Error::ModelLoad("S3Tokenizer weights not found".into()));
    }

    println!("Loading S3Tokenizer...");
    let load_start = Instant::now();
    let mut s3tokenizer = S3TokenizerMLX::load(&model_dir)?;
    println!("  Loaded in {:.2}s", load_start.elapsed().as_secs_f64());

    println!("Encoding mel to codes...");
    let encode_start = Instant::now();
    let codes = s3tokenizer.encode(&mel)?;
    mlx_rs::transforms::eval([&codes]).map_err(|e| Error::Mlx(e))?;
    let encode_time = encode_start.elapsed().as_secs_f64();

    let codes_vec: Vec<i32> = codes.as_slice::<i32>().to_vec();
    println!("  Codes: {} codes", codes_vec.len());
    println!("  Time: {:.3}s ({:.1}x real-time)", encode_time, encode_time / audio_duration as f64);
    println!("  Sample codes: {:?}", &codes_vec[..codes_vec.len().min(10)]);
    println!();

    // =========================================================================
    // Step 3: Flow Decoder - Audio Codes → Mel Spectrogram
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("STEP 3: Flow Decoder (Audio Codes → Mel)");
    println!("═══════════════════════════════════════════════════════════════════════");

    let flow_weights = model_dir.join("tts_mlx").join("flow.safetensors");
    if !flow_weights.exists() {
        println!("  ⚠️  Flow weights not found. Run: python scripts/convert_tts_weights.py");
        return Err(Error::ModelLoad("Flow weights not found".into()));
    }

    println!("Loading Flow Decoder...");
    let load_start = Instant::now();
    let flow_decoder = FlowDecoder::load(&model_dir)?;
    println!("  Loaded in {:.2}s", load_start.elapsed().as_secs_f64());
    println!("  Weights loaded: {}", flow_decoder.weights_loaded);

    println!("Generating mel from codes...");
    let flow_start = Instant::now();
    let generated_mel = flow_decoder.generate(&codes_vec)?;
    mlx_rs::transforms::eval([&generated_mel]).map_err(|e| Error::Mlx(e))?;
    let flow_time = flow_start.elapsed().as_secs_f64();

    println!("  Generated mel shape: {:?}", generated_mel.shape());
    println!("  Time: {:.3}s", flow_time);
    let mel_flat = generated_mel.flatten(None, None).unwrap();
    mlx_rs::transforms::eval([&mel_flat]).map_err(|e| Error::Mlx(e))?;
    let mel_slice: Vec<f32> = mel_flat.as_slice::<f32>().to_vec();
    let mel_min = mel_slice.iter().cloned().fold(f32::INFINITY, f32::min);
    let mel_max = mel_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mel_mean = mel_slice.iter().sum::<f32>() / mel_slice.len() as f32;
    println!("  Mel range: [{:.4}, {:.4}], mean: {:.4}", mel_min, mel_max, mel_mean);
    println!();

    // =========================================================================
    // Step 4: HiFi-GAN - Mel → Audio Waveform
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("STEP 4: HiFi-GAN (Mel → Audio)");
    println!("═══════════════════════════════════════════════════════════════════════");

    let hifigan_weights = model_dir.join("tts_mlx").join("hifigan.safetensors");
    if !hifigan_weights.exists() {
        println!("  ⚠️  HiFi-GAN weights not found. Run: python scripts/convert_tts_weights.py");
        return Err(Error::ModelLoad("HiFi-GAN weights not found".into()));
    }

    println!("Loading HiFi-GAN...");
    let load_start = Instant::now();
    let hifigan = HiFiGAN::load(&model_dir)?;
    println!("  Loaded in {:.2}s", load_start.elapsed().as_secs_f64());
    println!("  Weights loaded: {}", hifigan.weights_loaded);

    println!("Synthesizing audio from mel...");
    let hifi_start = Instant::now();
    let audio_output_arr = hifigan.synthesize(&generated_mel)?;
    mlx_rs::transforms::eval([&audio_output_arr]).map_err(|e| Error::Mlx(e))?;
    let hifi_time = hifi_start.elapsed().as_secs_f64();

    // Convert Array to Vec<f32>
    let audio_output: Vec<f32> = audio_output_arr.as_slice::<f32>().to_vec();
    let output_duration = audio_output.len() as f32 / 24000.0;
    println!("  Output samples: {}", audio_output.len());
    println!("  Output duration: {:.2}s @ 24kHz", output_duration);
    println!("  Time: {:.3}s", hifi_time);
    let audio_min = audio_output.iter().cloned().fold(f32::INFINITY, f32::min);
    let audio_max = audio_output.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let audio_mean = audio_output.iter().sum::<f32>() / audio_output.len() as f32;
    let audio_rms = (audio_output.iter().map(|x| x * x).sum::<f32>() / audio_output.len() as f32).sqrt();
    println!("  Audio range: [{:.6}, {:.6}], mean: {:.6}, RMS: {:.6}", audio_min, audio_max, audio_mean, audio_rms);
    println!();

    // =========================================================================
    // Save output
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("SAVING OUTPUT");
    println!("═══════════════════════════════════════════════════════════════════════");

    let output_path = PathBuf::from("./tts_output.wav");
    if !audio_output.is_empty() {
        save_wav(&audio_output, 24000, &output_path)?;
        println!("  Saved to: {}", output_path.display());
    } else {
        println!("  ⚠️  No audio generated");
    }
    println!();

    // =========================================================================
    // Summary
    // =========================================================================
    let total_time = encode_time + flow_time + hifi_time;

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                          SUMMARY                                     ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║ Input audio:       {:.2}s @ {}Hz", audio_duration, sample_rate);
    println!("║ Audio codes:       {} codes", codes_vec.len());
    println!("║ Output audio:      {:.2}s @ 24kHz", output_duration);
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║ S3Tokenizer:       {:.3}s", encode_time);
    println!("║ Flow Decoder:      {:.3}s", flow_time);
    println!("║ HiFi-GAN:          {:.3}s", hifi_time);
    println!("║ Total:             {:.3}s ({:.1}x real-time)", total_time, total_time / audio_duration as f64);
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    Ok(())
}
