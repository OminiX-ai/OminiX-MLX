//! Transcribe audio using Qwen3-ASR.
//!
//! Usage:
//!   cargo run --release --example transcribe [model_dir] <audio_path> [--language LANG]
//!
//! Model path resolution:
//!   1. Command line argument (e.g. ~/.OminiX/models/qwen3-asr-0.6b)
//!   2. QWEN3_ASR_MODEL_PATH environment variable
//!   3. ~/.OminiX/models/qwen3-asr-1.7b
//!
//! Supports: WAV, FLAC, MP3, M4A, OGG (non-WAV requires ffmpeg)

use qwen3_asr_mlx::{Qwen3ASR, default_model_path};
use qwen3_asr_mlx::audio;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut model_dir = None;
    let mut audio_path = None;
    let mut language = "Chinese".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--language" | "--lang" | "-l" => {
                i += 1;
                if i < args.len() {
                    language = args[i].clone();
                }
            }
            s if !s.starts_with("--") => {
                if audio_path.is_some() {
                    // ignore extra args
                } else if model_dir.is_some() {
                    audio_path = Some(std::path::PathBuf::from(s));
                } else {
                    if s.ends_with(".wav") || s.ends_with(".m4a") || s.ends_with(".mp3")
                        || s.ends_with(".flac") || s.ends_with(".ogg") || s.ends_with(".aac")
                    {
                        audio_path = Some(std::path::PathBuf::from(s));
                    } else {
                        model_dir = Some(std::path::PathBuf::from(s));
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }

    let model_dir = model_dir.unwrap_or_else(default_model_path);
    let audio_path = audio_path.unwrap_or_else(|| {
        eprintln!("Usage: transcribe [model_dir] <audio.wav> [--language Chinese|English|...]\n");
        eprintln!("Examples:");
        eprintln!("  cargo run --release --example transcribe -- audio.wav");
        eprintln!("  cargo run --release --example transcribe -- ~/.OminiX/models/qwen3-asr-0.6b audio.wav");
        eprintln!("  cargo run --release --example transcribe -- audio.wav --language English");
        std::process::exit(1);
    });

    println!("Loading model from {}...", model_dir.display());
    let start = Instant::now();
    let mut model = Qwen3ASR::load(&model_dir).expect("Failed to load model");
    println!("Model loaded in {:.2}s\n", start.elapsed().as_secs_f32());

    // Load audio (convert non-WAV formats via ffmpeg)
    let (samples, sample_rate) = load_audio(&audio_path);
    let duration_secs = samples.len() as f32 / sample_rate as f32;
    println!("Audio: {:.1}s ({} samples @ {}Hz)", duration_secs, samples.len(), sample_rate);

    // Resample to 16kHz
    let samples = audio::resample(&samples, sample_rate, 16000).expect("Resample failed");

    let start = Instant::now();
    let text = model.transcribe_samples(&samples, &language)
        .expect("Transcription failed");
    let elapsed = start.elapsed().as_secs_f32();

    println!("\n=== Transcription ({:.2}s, {:.1}x realtime) ===", elapsed, duration_secs / elapsed);
    println!("{}", text);
}

/// Load audio from file. For non-WAV formats, convert via ffmpeg.
fn load_audio(path: &std::path::Path) -> (Vec<f32>, u32) {
    let ext = path.extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    if ext == "wav" {
        return audio::load_wav(path).expect("Failed to load WAV");
    }

    // Convert non-WAV to WAV using ffmpeg
    eprintln!("Converting {} to WAV via ffmpeg...", ext.to_uppercase());
    let tmp_wav = std::env::temp_dir().join("qwen3_asr_temp.wav");

    let status = std::process::Command::new("ffmpeg")
        .args([
            "-i", &path.to_string_lossy(),
            "-ar", "16000",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-y",
            &tmp_wav.to_string_lossy(),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .expect("Failed to run ffmpeg. Is ffmpeg installed?");

    if !status.success() {
        eprintln!("ffmpeg conversion failed");
        std::process::exit(1);
    }

    let result = audio::load_wav(&tmp_wav).expect("Failed to load converted WAV");
    let _ = std::fs::remove_file(&tmp_wav);
    result
}
