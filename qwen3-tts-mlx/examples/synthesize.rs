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

    /// Speaker name (e.g. vivian, serena, ryan, aiden, uncle_fu).
    /// When combined with --instruct, uses speaker+instruct mode.
    /// Without --instruct, defaults to "vivian" if not specified.
    #[arg(short, long)]
    speaker: Option<String>,

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

    /// Speed factor for speech (0.5-2.0). <1.0 = slower, >1.0 = faster.
    #[arg(long)]
    speed: Option<f32>,

    /// Repetition penalty (e.g. 1.05)
    #[arg(long)]
    repetition_penalty: Option<f32>,
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

    let default_speaker = "vivian".to_string();
    let speaker = args.speaker.as_deref().unwrap_or(&default_speaker);

    let opts = SynthesizeOptions {
        speaker,
        language: &args.language,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        max_new_tokens: args.max_tokens,
        seed: args.seed,
        speed_factor: args.speed,
        repetition_penalty: args.repetition_penalty,
    };

    if let Some(ref ref_audio_path) = args.reference_audio {
        // Voice cloning mode
        eprintln!("VoiceClone mode: reference audio from {}", ref_audio_path);
        let (ref_samples, ref_sr) = mlx_rs_core::audio::load_wav(ref_audio_path)?;
        let ref_samples = if ref_sr != 24000 {
            eprintln!("Resampling reference audio from {}Hz to 24000Hz", ref_sr);
            mlx_rs_core::audio::resample(&ref_samples, ref_sr, 24000)
        } else {
            ref_samples
        };
        eprintln!("Reference audio: {:.2}s ({} samples at 24kHz)", ref_samples.len() as f32 / 24000.0, ref_samples.len());

        if args.reference_text.is_some() {
            eprintln!("Warning: --reference-text ignored (ICL mode disabled). Using x-vector mode.");
        }

        let (samples, timing) = if let Some(ref instruct) = args.instruct {
            // Voice clone + instruct (emotion/style control)
            eprintln!("VoiceClone+Instruct mode: instruct=\"{}\"", instruct);
            synth.synthesize_voice_clone_instruct_with_timing(
                &args.text,
                &ref_samples,
                instruct,
                &args.language,
                &opts,
            )?
        } else {
            eprintln!("x_vector mode");
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
        // Voice design / combined speaker+instruct mode
        let (samples, timing) = if args.speaker.is_some() {
            // Combined: preset speaker + style instruction
            eprintln!("Speaker+Instruct mode: speaker={}, instruct=\"{}\"", speaker, instruct);
            synth.synthesize_with_speaker_instruct_with_timing(
                &args.text,
                instruct,
                &opts,
            )?
        } else {
            // Pure voice design (no specific speaker)
            eprintln!("VoiceDesign mode: \"{}\"", instruct);
            synth.synthesize_voice_design_with_timing(
                &args.text,
                instruct,
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
