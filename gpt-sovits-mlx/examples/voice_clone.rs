//! Voice Cloning Example
//!
//! Demonstrates the high-level VoiceCloner API for GPT-SoVITS.
//!
//! # Usage
//!
//! ```bash
//! # Basic usage with default reference voice
//! cargo run --example voice_clone --release -- "你好，世界！"
//!
//! # With custom reference audio
//! cargo run --example voice_clone --release -- "你好，世界！" --ref /path/to/reference.wav
//!
//! # Save to file
//! cargo run --example voice_clone --release -- "你好，世界！" --output /tmp/output.wav
//!
//! # Interactive mode
//! cargo run --example voice_clone --release -- --interactive
//!
//! # List available voices
//! cargo run --example voice_clone --release -- --list-voices
//! ```

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

use gpt_sovits_mlx::voice_clone::{VoiceCloner, VoiceClonerConfig};
use serde::Deserialize;
use tracing::{debug, error, info, warn};

// Default config file location (can be overridden by VOICES_CONFIG env var)
const DEFAULT_VOICES_CONFIG: &str = "~/.OminiX/models/voices.json";
const VOICES_CONFIG_ENV: &str = "VOICES_CONFIG";

/// Voice configuration from JSON
#[derive(Debug, Deserialize, Clone)]
struct VoiceConfig {
    ref_audio: String,
    ref_text: String,
    #[serde(default)]
    vits_onnx: Option<String>,
    #[serde(default)]
    codes_path: Option<String>,
    #[serde(default)]
    aliases: Vec<String>,
    // Optional fields for feature parity with PrimeSpeech
    #[serde(default)]
    speed_factor: Option<f32>,
    #[serde(default)]
    text_lang: Option<String>,
    #[serde(default)]
    prompt_lang: Option<String>,
}

/// Root configuration structure
#[derive(Debug, Deserialize)]
struct VoicesConfig {
    #[serde(default = "default_voice")]
    default_voice: String,
    #[serde(default = "default_base_path")]
    models_base_path: String,
    // Use BTreeMap for deterministic iteration order when listing voices
    voices: BTreeMap<String, VoiceConfig>,
}

fn default_voice() -> String {
    "doubao".to_string()
}

fn default_base_path() -> String {
    "~/.OminiX/models".to_string()
}

impl VoicesConfig {
    /// Load voices config from JSON file
    /// Returns None if file doesn't exist or parsing fails (with error logging)
    fn load(path: &str) -> Option<Self> {
        let expanded = expand_path(path);
        let content = match fs::read_to_string(&expanded) {
            Ok(c) => c,
            Err(e) => {
                if e.kind() != std::io::ErrorKind::NotFound {
                    warn!("Failed to read voices config '{}': {}", expanded, e);
                }
                return None;
            }
        };
        match serde_json::from_str(&content) {
            Ok(config) => Some(config),
            Err(e) => {
                warn!("Failed to parse voices config '{}': {} (line {}, col {})",
                      expanded, e, e.line(), e.column());
                None
            }
        }
    }

    /// Get config file path from env var or default
    fn config_path() -> String {
        env::var(VOICES_CONFIG_ENV).unwrap_or_else(|_| DEFAULT_VOICES_CONFIG.to_string())
    }

    /// Find a voice by name or alias
    fn find_voice(&self, name: &str) -> Option<&VoiceConfig> {
        let name_lower = name.to_lowercase();

        // Direct match
        if let Some(voice) = self.voices.get(&name_lower) {
            return Some(voice);
        }

        // Search aliases
        for (_, voice) in &self.voices {
            if voice.aliases.iter().any(|a| a.to_lowercase() == name_lower) {
                return Some(voice);
            }
        }

        None
    }

    /// Resolve a relative path to absolute using base_path
    fn resolve_path(&self, relative: &str) -> String {
        if relative.starts_with('/') || relative.starts_with('~') {
            expand_path(relative)
        } else {
            let base = expand_path(&self.models_base_path);
            format!("{}/{}", base, relative)
        }
    }

    /// List all available voices (ordered alphabetically due to BTreeMap)
    fn list_voices(&self) {
        println!("Available voices:");
        println!("{:-<60}", "");
        for (name, voice) in &self.voices {
            let aliases = if voice.aliases.is_empty() {
                String::new()
            } else {
                format!(" (aliases: {})", voice.aliases.join(", "))
            };
            let onnx = if voice.vits_onnx.is_some() { " [custom ONNX]" } else { "" };
            let ref_exists = Path::new(&self.resolve_path(&voice.ref_audio)).exists();
            let status = if ref_exists { "" } else { " [ref missing]" };
            println!("  {}{}{}{}", name, aliases, onnx, status);
        }
        println!("{:-<60}", "");
        println!("Default: {}", self.default_voice);
    }

    /// Validate paths and log warnings for missing files
    #[allow(dead_code)]
    fn validate_paths(&self) {
        for (name, voice) in &self.voices {
            let ref_path = self.resolve_path(&voice.ref_audio);
            if !Path::new(&ref_path).exists() {
                warn!(voice = name, path = ref_path, "ref_audio not found");
            }
            if let Some(ref onnx) = voice.vits_onnx {
                let onnx_path = self.resolve_path(onnx);
                if !Path::new(&onnx_path).exists() {
                    warn!(voice = name, path = onnx_path, "vits_onnx not found");
                }
            }
        }
    }
}

/// Expand ~ to home directory
fn expand_path(path: &str) -> String {
    if let Some(home) = dirs::home_dir() {
        if path == "~" {
            return home.display().to_string();
        } else if path.starts_with("~/") {
            return format!("{}{}", home.display(), &path[1..]);
        }
    }
    path.to_string()
}

fn print_help() {
    println!("Voice Clone - GPT-SoVITS TTS");
    println!("============================");
    println!();
    println!("Usage:");
    println!("  voice_clone \"text to speak\"              Synthesize and play text");
    println!("  voice_clone \"text\" --voice NAME          Use a configured voice preset");
    println!("  voice_clone \"text\" --ref FILE            Use custom reference audio");
    println!("  voice_clone \"text\" --ref-text \"text\"     Reference transcript (enables few-shot mode)");
    println!("  voice_clone \"text\" --codes FILE.bin      Use pre-computed prompt semantic codes");
    println!("  voice_clone \"text\" --output FILE.wav     Save to WAV file");
    println!("  voice_clone \"text\" --vits FILE.safetensors  Use custom finetuned VITS model");
    println!("  voice_clone \"text\" --pretrained FILE.safetensors  Pretrained base for finetuned VITS");
    println!("  voice_clone --interactive                 Interactive mode");
    println!("  voice_clone --list-voices                 List available voice presets");
    println!("  voice_clone --mlx-vits                    Use MLX VITS (not recommended)");
    println!("  voice_clone --help                        Show this help");
    println!();
    println!("Voice Configuration:");
    println!("  Default config: {}", DEFAULT_VOICES_CONFIG);
    println!("  Override with: export {}=/path/to/voices.json", VOICES_CONFIG_ENV);
    println!("  Use --list-voices to see available presets");
    println!();
    println!("VITS Backend:");
    println!("  Default: ONNX VITS (batched decode, matches Python, best quality)");
    println!("  --mlx-vits: Force MLX VITS (per-chunk decode, may have artifacts)");
    println!();
    println!("Examples:");
    println!("  voice_clone \"你好，世界！\"");
    println!("  voice_clone \"今天天气真好\" --voice marc");
    println!("  voice_clone \"今天天气真好\" --ref my_voice.wav");
    println!("  voice_clone \"测试语音\" --output test.wav");
    println!();
    println!("Few-shot mode (better quality with reference transcript):");
    println!("  voice_clone \"你好\" --ref voice.wav --ref-text \"这是参考音频的文本\"");
    println!();
    println!("Few-shot with Python-extracted codes (best quality):");
    println!("  # First extract codes with Python:");
    println!("  python scripts/extract_prompt_semantic.py voice.wav codes.bin");
    println!("  # Then use them:");
    println!("  voice_clone \"你好\" --ref voice.wav --ref-text \"参考文本\" --codes codes.bin");
}

/// Parsed command line arguments
struct Args {
    text: Option<String>,
    ref_audio: Option<String>,
    ref_text: Option<String>,
    codes_path: Option<String>,
    tokens_path: Option<String>,  // Pre-computed semantic tokens (for testing)
    output: Option<String>,
    t2s_model: Option<String>,    // Custom T2S model path
    vits_model: Option<String>,   // Custom VITS model path (finetuned)
    vits_onnx_model: Option<String>,  // Custom VITS ONNX model path
    pretrained_model: Option<String>,  // Pretrained base for finetuned VITS
    speed_factor: Option<f32>,    // Speed factor (1.0 = normal, >1 = faster)
    interactive: bool,
    greedy: bool,
    mlx_vits: bool,  // Force MLX VITS instead of default ONNX
    list_voices: bool,  // List available voice presets
    voices_config: Option<VoicesConfig>,  // Loaded voice configuration
}

fn parse_args() -> Args {
    let args: Vec<String> = env::args().skip(1).collect();

    // Load voice configuration (env var VOICES_CONFIG overrides default path)
    let config_path = VoicesConfig::config_path();
    let voices_config = VoicesConfig::load(&config_path);

    let mut text = None;
    let mut ref_audio = None;
    let mut ref_text = None;
    let mut codes_path = None;
    let mut tokens_path = None;
    let mut output = None;
    let mut t2s_model = None;
    let mut vits_model = None;
    let mut vits_onnx_model = None;
    let mut pretrained_model = None;
    let mut speed_factor = None;
    let mut interactive = false;
    let mut greedy = false;
    let mut mlx_vits = false;
    let mut list_voices = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            "--list-voices" | "--voices" => {
                list_voices = true;
            }
            "--ref" | "-r" => {
                if i + 1 < args.len() {
                    ref_audio = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--ref-text" | "-t" => {
                if i + 1 < args.len() {
                    ref_text = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--codes" | "-c" => {
                if i + 1 < args.len() {
                    codes_path = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--tokens" => {
                if i + 1 < args.len() {
                    tokens_path = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--output" | "-o" => {
                if i + 1 < args.len() {
                    output = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--t2s" => {
                if i + 1 < args.len() {
                    t2s_model = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--vits" => {
                if i + 1 < args.len() {
                    vits_model = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--vits-onnx" => {
                if i + 1 < args.len() {
                    vits_onnx_model = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--pretrained" => {
                if i + 1 < args.len() {
                    pretrained_model = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--text" => {
                if i + 1 < args.len() {
                    text = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--voice" => {
                // Look up voice from config
                if i + 1 < args.len() {
                    let voice_name = &args[i + 1];
                    if let Some(ref config) = voices_config {
                        if let Some(voice) = config.find_voice(voice_name) {
                            if ref_audio.is_none() {
                                ref_audio = Some(config.resolve_path(&voice.ref_audio));
                            }
                            if ref_text.is_none() {
                                ref_text = Some(voice.ref_text.clone());
                            }
                            if vits_onnx_model.is_none() {
                                if let Some(ref onnx) = voice.vits_onnx {
                                    vits_onnx_model = Some(config.resolve_path(onnx));
                                }
                            }
                            if codes_path.is_none() {
                                if let Some(ref codes) = voice.codes_path {
                                    let resolved = config.resolve_path(codes);
                                    if Path::new(&resolved).exists() {
                                        codes_path = Some(resolved);
                                    }
                                }
                            }
                            // Get speed_factor from voice config if not already set
                            if speed_factor.is_none() {
                                speed_factor = voice.speed_factor;
                            }
                        } else {
                            warn!(voice = voice_name.as_str(), "Unknown voice. Use --list-voices to see available voices.");
                        }
                    } else {
                        warn!(path = config_path, "Could not load voices config");
                    }
                    i += 1;
                }
            }
            "--play" => {
                // Play is default behavior, ignore
            }
            "--interactive" | "-i" => {
                interactive = true;
            }
            "--greedy" => {
                greedy = true;
            }
            "--speed" => {
                if i + 1 < args.len() {
                    if let Ok(s) = args[i + 1].parse::<f32>() {
                        speed_factor = Some(s);
                    }
                    i += 1;
                }
            }
            "--mlx-vits" => {
                // Force MLX VITS instead of default ONNX (not recommended)
                mlx_vits = true;
            }
            arg if !arg.starts_with('-') => {
                if text.is_none() {
                    text = Some(arg.to_string());
                }
            }
            _ => {}
        }
        i += 1;
    }

    Args { text, ref_audio, ref_text, codes_path, tokens_path, output, t2s_model, vits_model, vits_onnx_model, pretrained_model, speed_factor, interactive, greedy, mlx_vits, list_voices, voices_config }
}

fn synthesize_and_play(cloner: &mut VoiceCloner, text: &str, output: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    info!(text = text, reference = cloner.reference_path().unwrap_or("none"), "Synthesizing");

    let start = Instant::now();
    let audio = cloner.synthesize(text)?;
    let gen_time = start.elapsed();

    info!(
        tokens = audio.num_tokens,
        elapsed_ms = gen_time.as_secs_f64() * 1000.0,
        duration_secs = audio.duration_secs(),
        samples = audio.samples.len(),
        "Generated audio"
    );

    // Save if output specified
    if let Some(path) = output {
        cloner.save_wav(&audio, path)?;
        info!(path = path, "Saved audio");
    }

    // Play audio
    debug!("Playing audio...");
    cloner.play_blocking(&audio)?;

    Ok(())
}

fn interactive_mode(cloner: &mut VoiceCloner) -> Result<(), Box<dyn std::error::Error>> {
    // Interactive mode uses println! for user-facing output (not logging)
    println!("\nVoice Clone Interactive Mode");
    println!("============================");
    println!("Commands:");
    println!("  /ref <path>    - Change reference audio");
    println!("  /save <path>   - Save last audio to file");
    println!("  /quit          - Exit");
    println!("  <text>         - Synthesize and play text");
    println!();

    let mut last_audio = None;

    loop {
        print!("voice> ");
        io::stdout().flush()?;

        let mut input = String::new();
        if io::stdin().read_line(&mut input)? == 0 {
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        if input.starts_with("/ref ") {
            let path = &input[5..].trim();
            match cloner.set_reference_audio(path) {
                Ok(()) => info!(path = path, "Reference audio changed"),
                Err(e) => error!(error = %e, "Failed to set reference audio"),
            }
        } else if input.starts_with("/save ") {
            let path = &input[6..].trim();
            if let Some(ref audio) = last_audio {
                match cloner.save_wav(audio, path) {
                    Ok(()) => info!(path = path, "Saved audio"),
                    Err(e) => error!(error = %e, "Failed to save audio"),
                }
            } else {
                warn!("No audio to save. Generate some text first.");
            }
        } else if input == "/quit" || input == "/exit" || input == "/q" {
            info!("Goodbye!");
            break;
        } else if input.starts_with('/') {
            warn!(command = input, "Unknown command. Try /ref, /save, or /quit");
        } else {
            // Synthesize text
            match cloner.synthesize(input) {
                Ok(audio) => {
                    info!(tokens = audio.num_tokens, duration_secs = audio.duration_secs(), "Synthesized");
                    if let Err(e) = cloner.play_blocking(&audio) {
                        error!(error = %e, "Playback error");
                    }
                    last_audio = Some(audio);
                }
                Err(e) => error!(error = %e, "Synthesis error"),
            }
        }
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing subscriber (respects RUST_LOG env var)
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into())
        )
        .with_target(false)
        .init();

    let args = parse_args();

    // Handle --list-voices
    if args.list_voices {
        let config_path = VoicesConfig::config_path();
        if let Some(ref config) = args.voices_config {
            config.list_voices();
        } else {
            error!(path = %expand_path(&config_path), "Could not load voices config");
            println!("Create a voices.json file or set {} env var.", VOICES_CONFIG_ENV);
        }
        return Ok(());
    }

    // Initialize voice cloner
    info!("Initializing VoiceCloner...");
    let start = Instant::now();
    let mut config = VoiceClonerConfig::default();
    if let Some(t2s_path) = &args.t2s_model {
        config.t2s_weights = t2s_path.clone();
        info!(path = t2s_path, "Using custom T2S model");
    }
    if let Some(vits_path) = &args.vits_model {
        config.vits_weights = vits_path.clone();
        config.use_mlx_vits = true;  // Custom VITS requires MLX backend
        config.vits_onnx_path = None;  // Disable ONNX
        info!(path = vits_path, "Using custom VITS model");

        // If pretrained base is specified, use it for frozen layer weights
        if let Some(pretrained_path) = &args.pretrained_model {
            config.vits_pretrained_base = Some(pretrained_path.clone());
            info!(path = pretrained_path, "Using pretrained base");
        }
    }
    if let Some(vits_onnx_path) = &args.vits_onnx_model {
        config.vits_onnx_path = Some(vits_onnx_path.clone());
        info!(path = vits_onnx_path, "Using custom VITS ONNX model");
    }
    if args.greedy {
        config.top_k = 1;
        config.temperature = 0.001;
        config.top_p = 1.0;
        config.repetition_penalty = 1.0;
        config.noise_scale = 0.0;  // Deterministic VITS too
        debug!("Greedy mode: top_k=1, temperature=0.001, noise_scale=0.0");
    }
    if args.mlx_vits {
        config.use_mlx_vits = true;
        warn!("Using MLX VITS (per-chunk decode) - not recommended");
    }
    if let Some(speed) = args.speed_factor {
        config.speed = speed;
        info!(speed = speed, "Using speed factor");
    }
    let mut cloner = VoiceCloner::new(config)?;
    info!(elapsed_ms = start.elapsed().as_secs_f64() * 1000.0, "Models loaded");

    // Check HuBERT availability for few-shot mode
    if cloner.few_shot_available() {
        info!("HuBERT available (few-shot mode supported)");
    } else {
        info!("HuBERT not available (zero-shot mode only)");
    }

    // Set reference audio - use default voice from config if not specified
    let ref_path = if let Some(ref path) = args.ref_audio {
        path.clone()
    } else if let Some(ref config) = args.voices_config {
        // Use default voice from config
        if let Some(voice) = config.find_voice(&config.default_voice) {
            config.resolve_path(&voice.ref_audio)
        } else {
            error!(voice = config.default_voice, "Default voice not found in config");
            return Ok(());
        }
    } else {
        let config_path = VoicesConfig::config_path();
        error!("No reference audio specified and no voices config available");
        println!("Use --ref to specify reference audio, or create {}", expand_path(&config_path));
        return Ok(());
    };

    if !Path::new(&ref_path).exists() {
        error!(path = ref_path, "Reference audio not found");
        return Ok(());
    }

    let start = Instant::now();

    // Get reference text - use default from config if not specified
    let ref_text = if args.ref_text.is_some() {
        args.ref_text.clone()
    } else if let Some(ref config) = args.voices_config {
        // Get default voice's ref_text
        config.find_voice(&config.default_voice)
            .map(|v| v.ref_text.clone())
    } else {
        None
    };

    // Use few-shot mode if reference text is available
    if let Some(ref text) = ref_text {
        // Check if pre-computed codes are provided
        if let Some(ref codes_path) = args.codes_path {
            if !Path::new(codes_path).exists() {
                error!(path = codes_path, "Codes file not found");
                return Ok(());
            }
            cloner.set_reference_with_precomputed_codes(&ref_path, text, codes_path)?;
            info!(
                elapsed_ms = start.elapsed().as_secs_f64() * 1000.0,
                ref_text = text,
                codes_path = codes_path,
                "Reference loaded (few-shot with Python codes)"
            );
        } else {
            if !cloner.few_shot_available() {
                error!("Few-shot mode requires HuBERT model. Tip: Use --codes with pre-computed codes from Python");
                return Ok(());
            }
            cloner.set_reference_audio_with_text(&ref_path, text)?;
            info!(
                elapsed_ms = start.elapsed().as_secs_f64() * 1000.0,
                ref_text = text,
                "Reference loaded (few-shot mode)"
            );
        }
    } else {
        cloner.set_reference_audio(&ref_path)?;
        info!(
            elapsed_ms = start.elapsed().as_secs_f64() * 1000.0,
            "Reference loaded (zero-shot mode)"
        );
    }

    if args.interactive {
        interactive_mode(&mut cloner)?;
    } else if let Some(ref tokens_path) = args.tokens_path {
        // Use pre-computed tokens (for testing/debugging)
        let text = args.text.as_deref().unwrap_or("从季节上看，主要是增在秋粮");
        let bytes = fs::read(tokens_path)?;
        let tokens: Vec<i32> = bytes.chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        info!(
            text = text,
            tokens_path = tokens_path,
            num_tokens = tokens.len(),
            "Using pre-computed tokens"
        );
        debug!(first_10 = ?&tokens[..tokens.len().min(10)], "Token preview");

        let start = std::time::Instant::now();
        let audio = cloner.synthesize_from_tokens(text, &tokens)?;
        let gen_time = start.elapsed();

        info!(
            elapsed_ms = gen_time.as_secs_f64() * 1000.0,
            duration_secs = audio.duration_secs(),
            samples = audio.samples.len(),
            "Vocoded audio"
        );

        debug!("Playing audio...");
        cloner.play_blocking(&audio)?;
    } else if let Some(text) = args.text {
        synthesize_and_play(&mut cloner, &text, args.output.as_deref())?;
    } else {
        // Default demo
        let demo_texts = [
            "你好，欢迎使用语音克隆系统。",
            "今天天气真好，我们一起出去玩吧！",
            "这是一个测试句子，用来验证语音合成的效果。",
        ];

        info!("Voice Clone Demo");

        for text in demo_texts {
            synthesize_and_play(&mut cloner, text, None)?;
            println!();
        }
    }

    Ok(())
}
