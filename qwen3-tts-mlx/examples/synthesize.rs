use clap::Parser;
use std::time::Instant;

use qwen3_tts_mlx::{normalize_audio, save_wav, Synthesizer, SynthesizeOptions};

#[derive(Parser)]
#[command(name = "qwen3-tts", about = "Qwen3-TTS text-to-speech synthesis")]
struct Args {
    /// Path to the model directory
    #[arg(short, long)]
    model_dir: String,

    /// Text to synthesize
    text: String,

    /// Output WAV file path
    #[arg(short, long, default_value = "output.wav")]
    output: String,

    /// Speaker name (e.g. vivian, serena, ryan, aiden, uncle_fu)
    #[arg(short, long, default_value = "vivian")]
    speaker: String,

    /// Language (e.g. english, chinese, japanese, korean, french)
    #[arg(short, long, default_value = "english")]
    language: String,

    /// Temperature for sampling (0.0 = greedy)
    #[arg(short, long)]
    temperature: Option<f32>,

    /// Top-k for sampling
    #[arg(short = 'k', long)]
    top_k: Option<i32>,

    /// Top-p (nucleus) sampling threshold (0.0-1.0, e.g. 0.9)
    #[arg(short = 'p', long)]
    top_p: Option<f32>,

    /// Maximum number of codec frames to generate
    #[arg(short = 'n', long)]
    max_tokens: Option<i32>,

    /// Random seed for deterministic output
    #[arg(long)]
    seed: Option<u64>,

    /// Enable streaming mode (generate and decode in chunks)
    #[arg(long)]
    streaming: bool,

    /// Number of frames per streaming chunk (default 10 = ~833ms)
    #[arg(long, default_value = "10")]
    chunk_frames: usize,

    /// Voice design instruction (enables VoiceDesign mode).
    /// Describes desired voice, e.g. "A young woman with a warm, gentle voice"
    #[arg(long)]
    instruct: Option<String>,

    /// Path to reference audio WAV file for voice cloning (enables VoiceClone mode).
    /// Requires a Base model with speaker encoder.
    #[arg(long)]
    reference_audio: Option<String>,

    /// Transcript of the reference audio (enables ICL voice cloning mode).
    /// When provided with --reference-audio, uses full ICL mode for higher quality.
    /// Without this, falls back to x_vector_only mode.
    #[arg(long)]
    reference_text: Option<String>,
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();

    let start = Instant::now();
    let mut synth = Synthesizer::load(&args.model_dir)?;
    let load_time = start.elapsed();
    eprintln!("Model loaded in {:.1}s", load_time.as_secs_f32());

    eprintln!("Model type: {}", synth.model_type());
    eprintln!("Available speakers: {:?}", synth.speakers());
    eprintln!("Available languages: {:?}", synth.languages());

    if let Some(seed) = args.seed {
        eprintln!("Using seed: {}", seed);
    }

    let opts = SynthesizeOptions {
        speaker: &args.speaker,
        language: &args.language,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        max_new_tokens: args.max_tokens,
        seed: args.seed,
    };

    if let Some(ref ref_audio_path) = args.reference_audio {
        // Voice cloning mode
        eprintln!("VoiceClone mode: reference audio from {}", ref_audio_path);
        let (ref_samples, ref_sr) = mlx_rs_core::audio::load_wav(ref_audio_path)?;
        // Resample to 24kHz if needed
        let ref_samples = if ref_sr != 24000 {
            eprintln!("Resampling reference audio from {}Hz to 24000Hz", ref_sr);
            mlx_rs_core::audio::resample(&ref_samples, ref_sr, 24000)
        } else {
            ref_samples
        };
        eprintln!("Reference audio: {:.2}s ({} samples at 24kHz)", ref_samples.len() as f32 / 24000.0, ref_samples.len());

        let (samples, timing) = if let Some(ref ref_text) = args.reference_text {
            // ICL mode (full quality): uses both speaker embedding + reference codes
            eprintln!("ICL mode: reference text = \"{}\"", ref_text);
            synth.synthesize_voice_clone_icl_with_timing(
                &args.text,
                &ref_samples,
                ref_text,
                &args.language,
                &opts,
            )?
        } else {
            // x_vector_only mode: uses only speaker embedding
            eprintln!("x_vector_only mode (no --reference-text provided)");
            synth.synthesize_voice_clone_with_timing(
                &args.text,
                &ref_samples,
                &args.language,
                &opts,
            )?
        };

        if samples.is_empty() {
            eprintln!("No audio generated.");
            return Ok(());
        }

        let duration = samples.len() as f32 / synth.sample_rate as f32;
        let total_secs = timing.total_ms / 1000.0;
        eprintln!(
            "Generated {:.2}s of audio in {:.1}s ({:.1}x realtime)",
            duration,
            total_secs,
            duration as f64 / total_secs,
        );
        eprintln!(
            "Timing: prefill {:.0}ms, generation {:.0}ms ({} frames, {:.1} frames/s), decode {:.0}ms",
            timing.prefill_ms,
            timing.generation_ms,
            timing.generation_frames,
            timing.generation_frames as f64 / (timing.generation_ms / 1000.0),
            timing.decode_ms,
        );

        let samples = normalize_audio(&samples, 0.95);
        save_wav(&samples, synth.sample_rate, &args.output)?;
        eprintln!("Saved to {}", args.output);
    } else if let Some(ref instruct) = args.instruct {
        // VoiceDesign mode
        eprintln!("VoiceDesign mode: \"{}\"", instruct);
        let (samples, timing) = synth.synthesize_voice_design_with_timing(
            &args.text,
            instruct,
            &args.language,
            &opts,
        )?;

        if samples.is_empty() {
            eprintln!("No audio generated.");
            return Ok(());
        }

        let duration = samples.len() as f32 / synth.sample_rate as f32;
        let total_secs = timing.total_ms / 1000.0;
        eprintln!(
            "Generated {:.2}s of audio in {:.1}s ({:.1}x realtime)",
            duration,
            total_secs,
            duration as f64 / total_secs,
        );
        eprintln!(
            "Timing: prefill {:.0}ms, generation {:.0}ms ({} frames, {:.1} frames/s), decode {:.0}ms",
            timing.prefill_ms,
            timing.generation_ms,
            timing.generation_frames,
            timing.generation_frames as f64 / (timing.generation_ms / 1000.0),
            timing.decode_ms,
        );

        let samples = normalize_audio(&samples, 0.95);
        save_wav(&samples, synth.sample_rate, &args.output)?;
        eprintln!("Saved to {}", args.output);
    } else if args.streaming {
        // Streaming mode: generate and decode in chunks
        let start = Instant::now();
        let mut session = synth.start_streaming(&args.text, &opts, args.chunk_frames)?;
        let mut all_samples: Vec<f32> = Vec::new();
        let mut chunk_idx = 0;

        while let Some(chunk_samples) = session.next_chunk()? {
            eprintln!(
                "Chunk {}: {} samples ({:.2}s cumulative, {} frames total)",
                chunk_idx,
                chunk_samples.len(),
                session.duration_secs(),
                session.total_frames(),
            );
            all_samples.extend_from_slice(&chunk_samples);
            chunk_idx += 1;
        }

        let total_time = start.elapsed();

        if all_samples.is_empty() {
            eprintln!("No audio generated.");
            return Ok(());
        }

        let duration = all_samples.len() as f32 / synth.sample_rate as f32;
        eprintln!(
            "Streaming complete: {:.2}s of audio in {:.1}s ({:.1}x realtime), {} chunks",
            duration,
            total_time.as_secs_f32(),
            duration / total_time.as_secs_f32(),
            chunk_idx,
        );

        let all_samples = normalize_audio(&all_samples, 0.95);
        save_wav(&all_samples, synth.sample_rate, &args.output)?;
        eprintln!("Saved to {}", args.output);
    } else {
        // Non-streaming mode (CustomVoice)
        let (samples, timing) = synth.synthesize_with_timing(&args.text, &opts)?;

        if samples.is_empty() {
            eprintln!("No audio generated.");
            return Ok(());
        }

        let duration = samples.len() as f32 / synth.sample_rate as f32;
        let total_secs = timing.total_ms / 1000.0;
        eprintln!(
            "Generated {:.2}s of audio in {:.1}s ({:.1}x realtime)",
            duration,
            total_secs,
            duration as f64 / total_secs,
        );
        eprintln!(
            "Timing: prefill {:.0}ms, generation {:.0}ms ({} frames, {:.1} frames/s), decode {:.0}ms",
            timing.prefill_ms,
            timing.generation_ms,
            timing.generation_frames,
            timing.generation_frames as f64 / (timing.generation_ms / 1000.0),
            timing.decode_ms,
        );

        let samples = normalize_audio(&samples, 0.95);
        save_wav(&samples, synth.sample_rate, &args.output)?;
        eprintln!("Saved to {}", args.output);
    }

    Ok(())
}
