//! Comprehensive benchmark for qwen3-tts-mlx.
//! Tests all generation modes, speed control, and voice cloning.
//! Outputs structured timing data for comparison.

use std::time::Instant;

use qwen3_tts_mlx::{normalize_audio, save_wav, Synthesizer, SynthesizeOptions};

const CUSTOM_VOICE_MODEL: &str = "../models/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit";
const BASE_MODEL: &str = "../models/Qwen3-TTS-12Hz-1.7B-Base";
const REF_AUDIO: &str = "../step-audio2-mlx/real_speech.wav";

struct BenchResult {
    name: String,
    prefill_ms: f64,
    generation_ms: f64,
    generation_frames: usize,
    decode_ms: f64,
    total_ms: f64,
    audio_duration_s: f32,
    frames_per_sec: f64,
    realtime_factor: f64,
    sample_count: usize,
}

impl BenchResult {
    fn print(&self) {
        let gen_fps = if self.generation_ms > 0.0 {
            self.generation_frames as f64 / (self.generation_ms / 1000.0)
        } else {
            0.0
        };
        eprintln!("=== {} ===", self.name);
        eprintln!("  Prefill:       {:>8.1}ms", self.prefill_ms);
        eprintln!("  Generation:    {:>8.1}ms  ({} frames, {:.1} fps)", self.generation_ms, self.generation_frames, gen_fps);
        eprintln!("  Decode:        {:>8.1}ms", self.decode_ms);
        eprintln!("  Total:         {:>8.1}ms", self.total_ms);
        eprintln!("  Audio:         {:>8.2}s  ({} samples)", self.audio_duration_s, self.sample_count);
        eprintln!("  Realtime:      {:>8.2}x", self.realtime_factor);
        eprintln!();
    }

    fn print_csv_header() {
        println!("name,prefill_ms,generation_ms,gen_frames,decode_ms,total_ms,audio_s,samples,fps,realtime_x");
    }

    fn print_csv(&self) {
        let gen_fps = if self.generation_ms > 0.0 {
            self.generation_frames as f64 / (self.generation_ms / 1000.0)
        } else {
            0.0
        };
        println!("{},{:.1},{:.1},{},{:.1},{:.1},{:.2},{},{:.1},{:.2}",
            self.name, self.prefill_ms, self.generation_ms, self.generation_frames,
            self.decode_ms, self.total_ms, self.audio_duration_s, self.sample_count,
            gen_fps, self.realtime_factor);
    }
}

fn run_test(
    synth: &mut Synthesizer,
    name: &str,
    text: &str,
    opts: &SynthesizeOptions,
) -> Result<BenchResult, Box<dyn std::error::Error>> {
    eprintln!("Running: {}", name);
    let (samples, timing) = synth.synthesize_with_timing(text, opts)?;
    let duration = samples.len() as f32 / synth.sample_rate as f32;
    let realtime = if timing.total_ms > 0.0 {
        duration as f64 / (timing.total_ms / 1000.0)
    } else {
        0.0
    };

    let result = BenchResult {
        name: name.to_string(),
        prefill_ms: timing.prefill_ms,
        generation_ms: timing.generation_ms,
        generation_frames: timing.generation_frames,
        decode_ms: timing.decode_ms,
        total_ms: timing.total_ms,
        audio_duration_s: duration,
        frames_per_sec: timing.generation_frames as f64 / (timing.generation_ms / 1000.0),
        realtime_factor: realtime,
        sample_count: samples.len(),
    };

    // Save output for functional validation
    let filename = format!("bench_{}.wav", name.replace(' ', "_").replace('=', "").to_lowercase());
    let samples = normalize_audio(&samples, 0.95);
    save_wav(&samples, synth.sample_rate, &filename)?;
    eprintln!("  Saved: {}", filename);

    Ok(result)
}

fn run_voice_clone_test(
    synth: &mut Synthesizer,
    name: &str,
    text: &str,
    ref_samples: &[f32],
    language: &str,
    opts: &SynthesizeOptions,
) -> Result<BenchResult, Box<dyn std::error::Error>> {
    eprintln!("Running: {}", name);
    let (samples, timing) = synth.synthesize_voice_clone_with_timing(text, ref_samples, language, opts)?;
    let duration = samples.len() as f32 / synth.sample_rate as f32;
    let realtime = if timing.total_ms > 0.0 {
        duration as f64 / (timing.total_ms / 1000.0)
    } else {
        0.0
    };

    let result = BenchResult {
        name: name.to_string(),
        prefill_ms: timing.prefill_ms,
        generation_ms: timing.generation_ms,
        generation_frames: timing.generation_frames,
        decode_ms: timing.decode_ms,
        total_ms: timing.total_ms,
        audio_duration_s: duration,
        frames_per_sec: timing.generation_frames as f64 / (timing.generation_ms / 1000.0),
        realtime_factor: realtime,
        sample_count: samples.len(),
    };

    let filename = format!("bench_{}.wav", name.replace(' ', "_").replace('=', "").to_lowercase());
    let samples = normalize_audio(&samples, 0.95);
    save_wav(&samples, synth.sample_rate, &filename)?;
    eprintln!("  Saved: {}", filename);

    Ok(result)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn".into()),
        )
        .init();

    let mut results: Vec<BenchResult> = Vec::new();

    // ========================================================================
    // Test Suite 1: CustomVoice mode (8-bit model)
    // ========================================================================
    eprintln!("\n============================================================");
    eprintln!("Loading CustomVoice model...");
    let load_start = Instant::now();
    let mut synth = Synthesizer::load(CUSTOM_VOICE_MODEL)?;
    let load_time = load_start.elapsed();
    eprintln!("CustomVoice model loaded in {:.1}s\n", load_time.as_secs_f32());

    let en_text = "The quick brown fox jumps over the lazy dog near the riverbank.";
    let zh_text = "今天天气很好，我们一起去公园散步吧。";
    let seed = Some(42u64);

    // Test 1: English, normal speed, seed=42
    {
        let opts = SynthesizeOptions {
            speaker: "vivian",
            language: "english",
            seed,
            ..Default::default()
        };
        results.push(run_test(&mut synth, "EN normal speed=1.0", en_text, &opts)?);
    }

    // Test 2: Chinese, normal speed, seed=42
    {
        let opts = SynthesizeOptions {
            speaker: "vivian",
            language: "chinese",
            seed,
            ..Default::default()
        };
        results.push(run_test(&mut synth, "ZH normal speed=1.0", zh_text, &opts)?);
    }

    // Test 3: English, slow speed=0.8 (EOS steering)
    {
        let opts = SynthesizeOptions {
            speaker: "vivian",
            language: "english",
            seed,
            speed_factor: Some(0.8),
            ..Default::default()
        };
        results.push(run_test(&mut synth, "EN slow speed=0.8", en_text, &opts)?);
    }

    // Test 4: English, fast speed=1.5 (WSOLA)
    {
        let opts = SynthesizeOptions {
            speaker: "vivian",
            language: "english",
            seed,
            speed_factor: Some(1.5),
            ..Default::default()
        };
        results.push(run_test(&mut synth, "EN fast speed=1.5", en_text, &opts)?);
    }

    // Test 5: English, very slow speed=0.6 (EOS steering, strong)
    {
        let opts = SynthesizeOptions {
            speaker: "vivian",
            language: "english",
            seed,
            speed_factor: Some(0.6),
            ..Default::default()
        };
        results.push(run_test(&mut synth, "EN very slow speed=0.6", en_text, &opts)?);
    }

    // Test 6: English with top-p sampling (exercises GPU top-p path)
    {
        let opts = SynthesizeOptions {
            speaker: "vivian",
            language: "english",
            seed,
            temperature: Some(0.9),
            top_p: Some(0.9),
            top_k: Some(50),
            ..Default::default()
        };
        results.push(run_test(&mut synth, "EN sampling top_p=0.9 top_k=50", en_text, &opts)?);
    }

    // Test 7: English greedy (temperature=0)
    {
        let opts = SynthesizeOptions {
            speaker: "vivian",
            language: "english",
            seed,
            temperature: Some(0.0),
            ..Default::default()
        };
        results.push(run_test(&mut synth, "EN greedy temp=0", en_text, &opts)?);
    }

    drop(synth);

    // ========================================================================
    // Test Suite 2: Voice cloning (Base model, x_vector_only)
    // ========================================================================
    let ref_audio_path = std::path::Path::new(REF_AUDIO);
    if ref_audio_path.exists() {
        eprintln!("\n============================================================");
        eprintln!("Loading Base model for voice cloning...");
        let load_start = Instant::now();
        let mut synth = Synthesizer::load(BASE_MODEL)?;
        let load_time = load_start.elapsed();
        eprintln!("Base model loaded in {:.1}s\n", load_time.as_secs_f32());

        let (ref_samples, ref_sr) = mlx_rs_core::audio::load_wav(REF_AUDIO)?;
        let ref_samples = if ref_sr != 24000 {
            eprintln!("Resampling reference audio from {}Hz to 24000Hz", ref_sr);
            mlx_rs_core::audio::resample(&ref_samples, ref_sr, 24000)
        } else {
            ref_samples
        };
        eprintln!("Reference audio: {:.2}s ({} samples)", ref_samples.len() as f32 / 24000.0, ref_samples.len());

        // Test 8: Voice clone, normal speed
        {
            let opts = SynthesizeOptions {
                speaker: "vivian",
                language: "english",
                seed,
                ..Default::default()
            };
            results.push(run_voice_clone_test(&mut synth, "VoiceClone EN speed=1.0", en_text, &ref_samples, "english", &opts)?);
        }

        // Test 9: Voice clone, slow speed=0.8
        {
            let opts = SynthesizeOptions {
                speaker: "vivian",
                language: "english",
                seed,
                speed_factor: Some(0.8),
                ..Default::default()
            };
            results.push(run_voice_clone_test(&mut synth, "VoiceClone EN speed=0.8", en_text, &ref_samples, "english", &opts)?);
        }

        drop(synth);
    } else {
        eprintln!("\nSkipping voice clone tests: {} not found", REF_AUDIO);
    }

    // ========================================================================
    // Summary
    // ========================================================================
    eprintln!("\n============================================================");
    eprintln!("BENCHMARK RESULTS SUMMARY");
    eprintln!("============================================================\n");

    for r in &results {
        r.print();
    }

    // CSV output for easy comparison
    eprintln!("--- CSV output ---");
    BenchResult::print_csv_header();
    for r in &results {
        r.print_csv();
    }

    Ok(())
}
