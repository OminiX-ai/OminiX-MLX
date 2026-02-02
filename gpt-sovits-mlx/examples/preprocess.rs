//! Preprocessing CLI for GPT-SoVITS Training
//!
//! This tool extracts all features needed for training from raw audio + transcript.
//!
//! Usage:
//! ```bash
//! # Single file
//! cargo run --release --example preprocess -- \
//!     --audio /path/to/audio.wav \
//!     --text "这是我说的话" \
//!     --output /path/to/training_data
//!
//! # Directory with transcript file
//! cargo run --release --example preprocess -- \
//!     --input-dir /path/to/audio_files \
//!     --transcript /path/to/transcript.list \
//!     --output /path/to/training_data
//! ```
//!
//! Output format:
//! ```
//! training_data/
//! ├── metadata.json
//! ├── phoneme_ids/*.npy      # [seq_len] int32
//! ├── bert_features/*.npy    # [1024, seq_len] float32
//! ├── semantic_ids/*.npy     # [seq_len] int32
//! └── hubert_features/*.npy  # [768, seq_len] float32 (optional)
//! ```

use std::fs;
use std::path::{Path, PathBuf};

use clap::Parser;
use mlx_rs::{
    module::Module,
    ops::swap_axes,
    transforms::eval,
    Array,
};

use gpt_sovits_mlx::{
    audio::load_audio_for_hubert,
    error::Error,
    models::hubert::load_hubert_model,
    models::vits::load_vits_model,
    text::{BertFeatureExtractor, TextPreprocessor, PreprocessorConfig},
};

/// Preprocessing CLI for GPT-SoVITS Training
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Single audio file to process
    #[arg(long)]
    audio: Option<PathBuf>,

    /// Text transcript for single audio
    #[arg(long)]
    text: Option<String>,

    /// Input directory containing audio files
    #[arg(long)]
    input_dir: Option<PathBuf>,

    /// Transcript file (format: path|speaker|lang|text)
    #[arg(long)]
    transcript: Option<PathBuf>,

    /// Output directory for training data
    #[arg(long)]
    output: PathBuf,

    /// Path to HuBERT weights
    #[arg(long, default_value = "~/.dora/models/primespeech/gpt-sovits-mlx/hubert.safetensors")]
    hubert_weights: String,

    /// Path to VITS weights (for quantizer codebook)
    #[arg(long, default_value = "~/.dora/models/primespeech/gpt-sovits-mlx/vits_pretrained_v2.safetensors")]
    vits_weights: String,

    /// Path to BERT weights
    #[arg(long, default_value = "~/.dora/models/primespeech/gpt-sovits-mlx/bert.safetensors")]
    bert_weights: String,

    /// Path to BERT tokenizer
    #[arg(long, default_value = "~/.dora/models/primespeech/gpt-sovits-mlx/chinese-roberta-tokenizer/tokenizer.json")]
    bert_tokenizer: String,

    /// Also save HuBERT features (for VITS training)
    #[arg(long, default_value = "false")]
    save_hubert: bool,
}

/// Expand ~ in paths
fn expand_path(path: &str) -> PathBuf {
    if path.starts_with("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(&path[2..]);
        }
    }
    PathBuf::from(path)
}

/// Sample metadata (matches training dataset format)
#[derive(serde::Serialize)]
struct SampleMeta {
    id: String,
    audio_path: String,
    transcript: String,
    phoneme_len: usize,
    semantic_len: usize,
}

/// Training data metadata
#[derive(serde::Serialize)]
struct Metadata {
    num_samples: usize,
    samples: Vec<SampleMeta>,
}

/// Process a single audio file
fn process_sample(
    audio_path: &Path,
    text: &str,
    sample_id: &str,
    hubert: &mut gpt_sovits_mlx::models::hubert::HuBertEncoder,
    vits: &mut gpt_sovits_mlx::models::vits::SynthesizerTrn,
    bert: &mut BertFeatureExtractor,
    text_processor: &TextPreprocessor,
    output_dir: &Path,
    save_hubert: bool,
) -> Result<SampleMeta, Error> {
    println!("Processing: {} - \"{}\"", sample_id, text);

    // 1. Load audio and extract HuBERT features
    let audio = load_audio_for_hubert(audio_path)
        .map_err(|e| Error::Message(format!("Failed to load audio: {}", e)))?;
    eval([&audio])?;

    let hubert_features = hubert.forward(&audio)
        .map_err(|e| Error::Message(format!("HuBERT forward failed: {}", e)))?;
    eval([&hubert_features])?;

    // hubert_features: [1, time, 768] -> [1, 768, time] for quantizer
    let hubert_ncl = swap_axes(&hubert_features, 1, 2)
        .map_err(|e| Error::Message(e.to_string()))?;
    eval([&hubert_ncl])?;

    // Debug: print HuBERT feature stats
    let hubert_shape = hubert_ncl.shape();
    println!("  HuBERT features shape: {:?}", hubert_shape);
    let hubert_mean: f32 = hubert_ncl.mean(false)?.item();
    let hubert_max: f32 = hubert_ncl.max(false)?.item();
    let hubert_min: f32 = hubert_ncl.min(false)?.item();
    println!("  HuBERT features mean/min/max: {:.4} / {:.4} / {:.4}", hubert_mean, hubert_min, hubert_max);

    // 2. Extract semantic tokens via ssl_proj + quantizer
    // This applies the ssl_proj Conv1d (kernel=2, stride=2 for 25hz) before quantization
    let semantic_ids = vits.extract_semantic_codes(&hubert_ncl)
        .map_err(|e| Error::Message(format!("extract_semantic_codes failed: {}", e)))?;
    eval([&semantic_ids])?;

    // Debug: print semantic token stats
    let semantic_shape = semantic_ids.shape();
    println!("  Semantic IDs shape: {:?}", semantic_shape);

    // semantic_ids: [1, 1, time] -> [time] for saving (squeeze both batch dims)
    let semantic_ids_1d = semantic_ids.squeeze_axes(&[0, 1])
        .map_err(|e| Error::Message(e.to_string()))?;
    let semantic_len = semantic_ids_1d.dim(0) as usize;

    // 3. Extract phonemes
    let preprocessor_output = text_processor.preprocess(text, None);
    let phoneme_ids: Vec<i32> = preprocessor_output.phoneme_ids.iter().map(|&x| x as i32).collect();
    let phoneme_len = phoneme_ids.len();

    // 4. Extract BERT features (raw, without word2ph alignment)
    // For training, we need features per token which will be aligned during batching
    let bert_features = bert.extract_raw_features(&preprocessor_output.text_normalized, true)
        .map_err(|e| Error::Message(format!("BERT extraction failed: {}", e)))?;
    eval([&bert_features])?;

    // bert_features: [1, seq_len, 1024] -> [1024, seq_len] for saving
    let bert_squeezed = bert_features.squeeze_axes(&[0])
        .map_err(|e| Error::Message(e.to_string()))?;
    let bert_ncl = swap_axes(&bert_squeezed, 0, 1)
        .map_err(|e| Error::Message(e.to_string()))?;

    // 5. Save all features
    let phoneme_dir = output_dir.join("phoneme_ids");
    let bert_dir = output_dir.join("bert_features");
    let semantic_dir = output_dir.join("semantic_ids");

    fs::create_dir_all(&phoneme_dir)?;
    fs::create_dir_all(&bert_dir)?;
    fs::create_dir_all(&semantic_dir)?;

    // Save phoneme IDs as numpy
    let phoneme_array = Array::from_slice(&phoneme_ids, &[phoneme_ids.len() as i32]);
    phoneme_array.save_numpy(&phoneme_dir.join(format!("{}.npy", sample_id)))
        .map_err(|e| Error::Message(format!("Failed to save phonemes: {}", e)))?;

    // Save BERT features
    bert_ncl.save_numpy(&bert_dir.join(format!("{}.npy", sample_id)))
        .map_err(|e| Error::Message(format!("Failed to save BERT: {}", e)))?;

    // Save semantic IDs
    semantic_ids_1d.save_numpy(&semantic_dir.join(format!("{}.npy", sample_id)))
        .map_err(|e| Error::Message(format!("Failed to save semantic: {}", e)))?;

    // Optionally save HuBERT features (for VITS training)
    if save_hubert {
        let hubert_dir = output_dir.join("hubert_features");
        fs::create_dir_all(&hubert_dir)?;

        // hubert_ncl: [1, 768, time] -> [768, time]
        let hubert_2d = hubert_ncl.squeeze_axes(&[0])
            .map_err(|e| Error::Message(e.to_string()))?;
        hubert_2d.save_numpy(&hubert_dir.join(format!("{}.npy", sample_id)))
            .map_err(|e| Error::Message(format!("Failed to save HuBERT: {}", e)))?;
    }

    Ok(SampleMeta {
        id: sample_id.to_string(),
        audio_path: audio_path.to_string_lossy().to_string(),
        transcript: text.to_string(),
        phoneme_len,
        semantic_len,
    })
}

/// Parse transcript file
/// Supports formats:
/// - GPT-SoVITS: filename<TAB>phonemes<TAB>word2ph<TAB>text
/// - Pipe-separated: path|speaker|lang|text or path|text
fn parse_transcript(path: &Path, input_dir: Option<&Path>) -> Result<Vec<(PathBuf, String)>, Error> {
    let content = fs::read_to_string(path)
        .map_err(|e| Error::Message(format!("Failed to read transcript: {}", e)))?;

    let mut samples = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Try TAB-separated first (GPT-SoVITS format)
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 4 {
            // Format: filename<TAB>phonemes<TAB>word2ph<TAB>text
            let filename = parts[0];
            let text = parts[parts.len() - 1].to_string();

            // Resolve path relative to input_dir if provided
            let audio_path = if let Some(dir) = input_dir {
                dir.join(filename)
            } else {
                PathBuf::from(filename)
            };
            samples.push((audio_path, text));
            continue;
        }

        // Try pipe-separated format
        let parts: Vec<&str> = line.split('|').collect();
        if parts.len() >= 2 {
            let audio_path = PathBuf::from(parts[0]);
            let text = parts[parts.len() - 1].to_string();
            samples.push((audio_path, text));
        }
    }

    Ok(samples)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("GPT-SoVITS Preprocessing");
    println!("========================");

    // Expand paths
    let hubert_path = expand_path(&args.hubert_weights);
    let vits_path = expand_path(&args.vits_weights);
    let bert_path = expand_path(&args.bert_weights);
    let tokenizer_path = expand_path(&args.bert_tokenizer);

    // Validate inputs
    let samples: Vec<(PathBuf, String)> = if let (Some(audio), Some(text)) = (&args.audio, &args.text) {
        vec![(audio.clone(), text.clone())]
    } else if let (Some(input_dir), Some(transcript)) = (&args.input_dir, &args.transcript) {
        parse_transcript(transcript, Some(input_dir))?
    } else {
        return Err("Must provide either --audio + --text, or --input-dir + --transcript".into());
    };

    println!("Samples to process: {}", samples.len());
    println!("Output directory: {:?}", args.output);
    println!();

    // Create output directory
    fs::create_dir_all(&args.output)?;

    // Load models
    println!("Loading HuBERT model...");
    let mut hubert = load_hubert_model(&hubert_path)?;

    println!("Loading VITS model (for ssl_proj + quantizer)...");
    let mut vits = load_vits_model(&vits_path)?;

    // Debug: check quantizer codebook
    println!("  Quantizer codebook shape: {:?}", vits.quantizer.embed.shape());
    // Check if codebook is loaded (not all zeros)
    eval([vits.quantizer.embed.as_ref()])?;
    let embed_sum: f32 = vits.quantizer.embed.sum(false)?.item();
    println!("  Quantizer codebook sum: {}", embed_sum);

    println!("Loading BERT model...");
    let mut bert = BertFeatureExtractor::new(&tokenizer_path, &bert_path, -3)?;

    // Create text preprocessor
    let text_processor = TextPreprocessor::new(PreprocessorConfig::default());

    println!();
    println!("Processing samples...");
    println!();

    // Process all samples
    let mut metadata = Metadata {
        num_samples: 0,
        samples: Vec::new(),
    };

    for (i, (audio_path, text)) in samples.iter().enumerate() {
        let sample_id = format!("sample_{:04}", i);

        match process_sample(
            audio_path,
            text,
            &sample_id,
            &mut hubert,
            &mut vits,
            &mut bert,
            &text_processor,
            &args.output,
            args.save_hubert,
        ) {
            Ok(sample_meta) => {
                metadata.samples.push(sample_meta);
                metadata.num_samples += 1;
            }
            Err(e) => {
                eprintln!("  Error processing {}: {}", audio_path.display(), e);
            }
        }
    }

    // Save metadata
    let metadata_json = serde_json::to_string_pretty(&metadata)?;
    fs::write(args.output.join("metadata.json"), metadata_json)?;

    println!();
    println!("========================");
    println!("Preprocessing complete!");
    println!("  Processed: {} samples", metadata.num_samples);
    println!("  Output: {:?}", args.output);
    println!();
    println!("To train T2S:");
    println!("  cargo run --release --example train_t2s -- \\");
    println!("      --data-dir {:?} \\", args.output);
    println!("      --pretrained ~/.dora/models/primespeech/gpt-sovits-mlx/t2s.safetensors \\");
    println!("      --output t2s_finetuned.safetensors");

    Ok(())
}
