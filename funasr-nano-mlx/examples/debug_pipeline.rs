//! Debug the transcription pipeline.

use funasr_nano_mlx::audio;
use funasr_nano_mlx::model::FunASRNano;
use mlx_rs::module::Module;
use mlx_rs::transforms::eval;

fn main() {
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "./Fun-ASR-Nano-2512".to_string());

    let audio_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "./Fun-ASR-Nano-2512/example/zh.wav".to_string());

    println!("Loading model from {}...", model_dir);
    let mut model = FunASRNano::load(&model_dir).expect("Failed to load model");

    // Load and preprocess audio
    let (samples, sample_rate) = audio::load_wav(&audio_path).expect("Failed to load audio");
    let samples = audio::resample(&samples, sample_rate, 16000).expect("Failed to resample");
    println!("Audio: {} samples at 16kHz", samples.len());

    // Compute mel spectrogram with LFR
    let audio_config = audio::AudioConfig::default();
    let mel = audio::compute_mel_spectrogram(&samples, &audio_config).expect("Failed to compute mel");
    eval([&mel]).expect("Failed to eval mel");
    println!("Mel shape: {:?}", mel.shape());
    println!("Mel range: [{:.4}, {:.4}]",
        mel.min(None).expect("min").item::<f32>(),
        mel.max(None).expect("max").item::<f32>());

    // Apply LFR
    let mel_lfr = audio::apply_lfr(&mel, 7, 6).expect("Failed to apply LFR");
    eval([&mel_lfr]).expect("Failed to eval LFR");
    println!("LFR shape: {:?}", mel_lfr.shape());
    println!("LFR range: [{:.4}, {:.4}]",
        mel_lfr.min(None).expect("min").item::<f32>(),
        mel_lfr.max(None).expect("max").item::<f32>());

    // Encode audio through encoder
    let encoder_out = model.encoder.forward(&mel_lfr).expect("Failed to encode");
    eval([&encoder_out]).expect("Failed to eval encoder");
    println!("Encoder output shape: {:?}", encoder_out.shape());
    println!("Encoder output range: [{:.4}, {:.4}]",
        encoder_out.min(None).expect("min").item::<f32>(),
        encoder_out.max(None).expect("max").item::<f32>());

    // Adaptor
    let adapted = model.adaptor.forward(&encoder_out).expect("Failed to adapt");
    eval([&adapted]).expect("Failed to eval adaptor");
    println!("Adaptor output shape: {:?}", adapted.shape());
    println!("Adaptor output range: [{:.4}, {:.4}]",
        adapted.min(None).expect("min").item::<f32>(),
        adapted.max(None).expect("max").item::<f32>());

    // Run text generation
    println!("\nGenerating text...");
    match model.generate_text(&adapted) {
        Ok(text) => println!("Result: {}", text),
        Err(e) => eprintln!("Error: {}", e),
    }
}
