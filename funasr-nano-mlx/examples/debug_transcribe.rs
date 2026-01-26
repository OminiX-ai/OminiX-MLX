//! Debug transcription pipeline.

use funasr_nano_mlx::model::FunASRNano;
use funasr_nano_mlx::audio::{self, AudioConfig};
use mlx_rs::transforms::eval;

fn main() {
    let model_dir = "./Fun-ASR-Nano-2512";
    let audio_path = "./Fun-ASR-Nano-2512/example/zh.wav";

    println!("Loading model...");
    let mut model = FunASRNano::load(&model_dir).expect("Failed to load model");
    println!("Model loaded!\n");

    // Load audio
    println!("Loading audio from {}...", audio_path);
    let (samples, sample_rate) = audio::load_wav(audio_path).expect("Failed to load audio");
    println!("  Loaded {} samples at {} Hz", samples.len(), sample_rate);

    // Resample
    let config = AudioConfig::default();
    let samples = audio::resample(&samples, sample_rate, config.sample_rate).expect("Resample failed");
    println!("  Resampled to {} samples at {} Hz", samples.len(), config.sample_rate);

    // Compute mel spectrogram
    let mel = audio::compute_mel_spectrogram(&samples, &config).expect("Mel failed");
    eval([&mel]).unwrap();
    println!("\nMel spectrogram shape: {:?}", mel.shape());
    
    // Check mel stats
    let mel_data: Vec<f32> = mel.try_as_slice().unwrap().to_vec();
    let mel_min = mel_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let mel_max = mel_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mel_mean: f32 = mel_data.iter().sum::<f32>() / mel_data.len() as f32;
    println!("  min: {:.4}, max: {:.4}, mean: {:.4}", mel_min, mel_max, mel_mean);

    // Apply LFR
    let mel_lfr = audio::apply_lfr(&mel, 7, 6).expect("LFR failed");
    eval([&mel_lfr]).unwrap();
    println!("\nLFR output shape: {:?}", mel_lfr.shape());

    // Encode audio
    println!("\nRunning encoder...");
    let audio_features = model.encode_audio(&mel_lfr).expect("Encode failed");
    eval([&audio_features]).unwrap();
    println!("Encoder output shape: {:?}", audio_features.shape());
    
    // Check encoder stats
    let enc_data: Vec<f32> = audio_features.try_as_slice().unwrap().to_vec();
    let enc_min = enc_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let enc_max = enc_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let enc_mean: f32 = enc_data.iter().sum::<f32>() / enc_data.len() as f32;
    println!("  min: {:.4}, max: {:.4}, mean: {:.4}", enc_min, enc_max, enc_mean);

    // Check for NaN/Inf
    let nan_count = enc_data.iter().filter(|x| x.is_nan()).count();
    let inf_count = enc_data.iter().filter(|x| x.is_infinite()).count();
    if nan_count > 0 || inf_count > 0 {
        println!("  WARNING: {} NaN, {} Inf values!", nan_count, inf_count);
    }

    println!("\nDone!");
}
