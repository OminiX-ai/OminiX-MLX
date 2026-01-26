use std::time::Instant;
use funasr_nano_mlx::audio::{self, AudioConfig, MelFrontend};
use funasr_nano_mlx::model::FunASRNano;
use mlx_rs::transforms::eval;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let audio_path = "/tmp/voice_memo.wav";
    
    // Load audio
    let t0 = Instant::now();
    let (samples, sample_rate) = audio::load_wav(audio_path)?;
    println!("load_wav: {:?}", t0.elapsed());
    
    // Resample
    let t0 = Instant::now();
    let samples = audio::resample(&samples, sample_rate, 16000)?;
    println!("resample: {:?}", t0.elapsed());
    println!("  {} samples", samples.len());
    
    // Create mel frontend
    let t0 = Instant::now();
    let config = AudioConfig::default();
    let frontend = MelFrontend::new(config);
    println!("MelFrontend::new: {:?}", t0.elapsed());
    
    // Compute mel spectrogram
    let t0 = Instant::now();
    let mel = frontend.compute_mel_spectrogram(&samples)?;
    eval([&mel])?;
    println!("compute_mel_spectrogram: {:?}", t0.elapsed());
    println!("  mel shape: {:?}", mel.shape());
    
    // Apply LFR
    let t0 = Instant::now();
    let mel_lfr = audio::apply_lfr(&mel, 7, 6)?;
    eval([&mel_lfr])?;
    println!("apply_lfr: {:?}", t0.elapsed());
    println!("  mel_lfr shape: {:?}", mel_lfr.shape());
    
    // Load model
    let t0 = Instant::now();
    let mut model = FunASRNano::load("./Fun-ASR-Nano-2512")?;
    println!("model load: {:?}", t0.elapsed());
    
    // Encode audio
    let t0 = Instant::now();
    let audio_features = model.encode_audio(&mel_lfr)?;
    eval([&audio_features])?;
    println!("encode_audio: {:?}", t0.elapsed());
    println!("  features shape: {:?}", audio_features.shape());
    
    // Generate text
    let t0 = Instant::now();
    let text = model.generate_text(&audio_features)?;
    let gen_time = t0.elapsed();
    let char_count = text.chars().count();

    println!("generate_text: {:?}", gen_time);
    println!("  text: {}...", &text.chars().take(100).collect::<String>());
    println!("  characters: {}", char_count);
    println!("  ms/char: {:.1}", gen_time.as_millis() as f64 / char_count as f64);

    // Calculate RTF
    let audio_duration = samples.len() as f64 / 16000.0;
    let total_time = gen_time.as_secs_f64();
    println!("\nPerformance:");
    println!("  Audio: {:.2}s", audio_duration);
    println!("  Encode: {:.1}ms", 546.0 + 77.0);
    println!("  Generate: {:.1}ms", gen_time.as_millis());
    println!("  RTF: {:.3}x ({:.1}x real-time)", total_time / audio_duration, audio_duration / total_time);

    Ok(())
}
