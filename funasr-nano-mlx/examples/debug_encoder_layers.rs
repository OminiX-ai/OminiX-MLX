//! Debug encoder layers to find numerical issues.

use funasr_nano_mlx::audio;
use funasr_nano_mlx::model::FunASRNano;
use mlx_rs::module::Module;
use mlx_rs::transforms::eval;

fn main() {
    let model_dir = "./Fun-ASR-Nano-2512";
    let audio_path = "./Fun-ASR-Nano-2512/example/zh.wav";

    println!("Loading model from {}...", model_dir);
    let mut model = FunASRNano::load(model_dir).expect("Failed to load model");

    // Load and preprocess audio
    let (samples, sample_rate) = audio::load_wav(audio_path).expect("Failed to load audio");
    let samples = audio::resample(&samples, sample_rate, 16000).expect("Failed to resample");
    println!("Audio: {} samples at 16kHz", samples.len());

    // Compute mel spectrogram with LFR
    let audio_config = audio::AudioConfig::default();
    let mel = audio::compute_mel_spectrogram(&samples, &audio_config).expect("Failed to compute mel");
    let mel_lfr = audio::apply_lfr(&mel, 7, 6).expect("Failed to apply LFR");
    eval([&mel_lfr]).expect("eval failed");
    println!("LFR: shape={:?}, range=[{:.4}, {:.4}]",
        mel_lfr.shape(),
        mel_lfr.min(None).expect("min").item::<f32>(),
        mel_lfr.max(None).expect("max").item::<f32>());

    // Run encoder forward layer by layer
    println!("\n=== Encoder layers ===");
    let encoder = &mut model.encoder;

    // First layer (handles LFR input dimension 560 -> 512)
    let mut h = mel_lfr.clone();
    for (i, layer) in encoder.encoders0.iter_mut().enumerate() {
        h = layer.forward(&h, None).expect("encoders0 failed");
        eval([&h]).expect("eval failed");
        println!("Encoder0[{}]: shape={:?}, range=[{:.4}, {:.4}]",
            i, h.shape(),
            h.min(None).expect("min").item::<f32>(),
            h.max(None).expect("max").item::<f32>());
    }

    // Check if there are NaN or Inf values
    let h_sum = h.sum(None).expect("sum failed");
    eval([&h_sum]).expect("eval failed");
    let sum_val = h_sum.item::<f32>();
    if sum_val.is_nan() || sum_val.is_infinite() {
        println!("WARNING: After encoders0 has NaN/Inf!");
    }

    // Main encoder layers
    for (i, layer) in encoder.encoders.iter_mut().enumerate() {
        h = layer.forward(&h, None).expect("encoder layer failed");

        if i < 5 || i % 10 == 9 || i == 49 {
            eval([&h]).expect("eval failed");

            let h_sum = h.sum(None).expect("sum failed");
            eval([&h_sum]).expect("eval failed");
            let sum_val = h_sum.item::<f32>();
            let has_nan = sum_val.is_nan() || sum_val.is_infinite();

            println!("Encoder[{:2}]: range=[{:.4}, {:.4}]{}",
                i,
                h.min(None).expect("min").item::<f32>(),
                h.max(None).expect("max").item::<f32>(),
                if has_nan { " [NaN/Inf!]" } else { "" });
        }
    }

    // After norm for main encoders
    h = encoder.after_norm.forward(&h).expect("after_norm failed");
    eval([&h]).expect("eval failed");
    println!("After main encoder norm: range=[{:.4}, {:.4}]",
        h.min(None).expect("min").item::<f32>(),
        h.max(None).expect("max").item::<f32>());

    // TP encoders
    for (i, layer) in encoder.tp_encoders.iter_mut().enumerate() {
        h = layer.forward(&h, None).expect("tp_encoder failed");
        if i == 0 || i == 9 || i == 19 {
            eval([&h]).expect("eval failed");
            println!("TP[{:2}]: range=[{:.4}, {:.4}]",
                i,
                h.min(None).expect("min").item::<f32>(),
                h.max(None).expect("max").item::<f32>());
        }
    }

    // TP norm
    h = encoder.tp_norm.forward(&h).expect("tp_norm failed");
    eval([&h]).expect("eval failed");
    println!("\nAfter TP norm: range=[{:.4}, {:.4}]",
        h.min(None).expect("min").item::<f32>(),
        h.max(None).expect("max").item::<f32>());

    println!("\n=== Encoder output shape: {:?} ===", h.shape());
}
