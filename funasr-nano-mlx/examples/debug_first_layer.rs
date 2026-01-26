//! Debug first encoder layer.

use funasr_nano_mlx::model::FunASRNano;
use funasr_nano_mlx::audio::{self, AudioConfig};
use mlx_rs::transforms::eval;
use mlx_rs::Dtype;

fn main() {
    // Load model with weights
    println!("Loading model...");
    let mut model = FunASRNano::load("./Fun-ASR-Nano-2512").expect("Failed to load model");
    println!("Model loaded!");
    
    // Create proper LFR input from audio
    let (samples, sample_rate) = audio::load_wav("./Fun-ASR-Nano-2512/example/zh.wav")
        .expect("Failed to load audio");
    let config = AudioConfig::default();
    let samples = audio::resample(&samples, sample_rate, config.sample_rate)
        .expect("Resample failed");
    let mel = audio::compute_mel_spectrogram(&samples, &config).expect("Mel failed");
    let mel_lfr = audio::apply_lfr(&mel, 7, 6).expect("LFR failed");
    
    eval([&mel_lfr]).unwrap();
    println!("\nInput (LFR) shape: {:?}", mel_lfr.shape());
    
    // Run just encoders0 layer
    let mut h = mel_lfr.clone();
    for layer in &mut model.encoder.encoders0 {
        h = layer.forward(&h, None).expect("Forward failed");
    }
    eval([&h]).unwrap();
    
    println!("After encoders0: shape {:?}", h.shape());
    let h_f32 = h.as_dtype(Dtype::Float32).unwrap();
    eval([&h_f32]).unwrap();
    let data: Vec<f32> = h_f32.try_as_slice().unwrap().to_vec();
    let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    println!("  min: {:.4}, max: {:.4}, mean: {:.4}", min, max, mean);
    
    // Run 10 more encoder layers
    println!("\nRunning 10 more encoder layers...");
    for (i, layer) in model.encoder.encoders.iter_mut().take(10).enumerate() {
        h = layer.forward(&h, None).expect("Forward failed");
        if i == 0 || i == 4 || i == 9 {
            eval([&h]).unwrap();
            let h_f32 = h.as_dtype(Dtype::Float32).unwrap();
            eval([&h_f32]).unwrap();
            let data: Vec<f32> = h_f32.try_as_slice().unwrap().to_vec();
            let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
            println!("After encoder layer {}: min={:.4}, max={:.4}, mean={:.4}", i+1, min, max, mean);
        }
    }
}
