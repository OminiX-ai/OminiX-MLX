//! Qwen-Image generation example (4-bit quantized)
//!
//! Model path can be set via environment variable:
//!   export DORA_MODELS_PATH=~/.OminiX/models
//!
//! Expected directory structure:
//!   $DORA_MODELS_PATH/qwen-image-2512-4bit/
//!   ├── transformer/    (4-bit quantized)
//!   ├── text_encoder/   (full precision)
//!   ├── vae/            (full precision)
//!   └── tokenizer/
//!
//! Falls back to HuggingFace cache if not found.
//!
//! Usage:
//!   cargo run --release --example generate_qwen_image -- --prompt "a cat sitting on a couch"

use std::collections::HashMap;
use std::path::PathBuf;

use clap::Parser;
use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(name = "generate_qwen_image")]
#[command(about = "Generate images with Qwen-Image")]
struct Args {
    /// Text prompt for image generation
    #[arg(short, long, default_value = "a cat sitting on a couch")]
    prompt: String,

    /// Output image path
    #[arg(short, long, default_value = "output_qwen.png")]
    output: PathBuf,

    /// Image height (must be divisible by 16)
    #[arg(long, default_value_t = 512)]
    height: i32,

    /// Image width (must be divisible by 16)
    #[arg(long, default_value_t = 512)]
    width: i32,

    /// Number of inference steps
    #[arg(long, default_value_t = 50)]
    steps: i32,

    /// Guidance scale for CFG
    #[arg(long, default_value_t = 5.0)]
    guidance: f32,

    /// Random seed
    #[arg(long)]
    seed: Option<u64>,

    /// Skip text encoder (use dummy embeddings for testing transformer/VAE only)
    #[arg(long)]
    skip_text_encoder: bool,

    /// Use 8-bit quantization (requires /tmp/qwen_image_8bit.safetensors)
    #[arg(long)]
    use_8bit: bool,

    /// Custom model path (overrides default HuggingFace cache location)
    #[arg(long)]
    model_path: Option<PathBuf>,
}

fn get_hf_cache_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();

    // Check ~/.cache/huggingface (Linux/default location)
    if let Some(home) = dirs::home_dir() {
        dirs.push(home.join(".cache").join("huggingface").join("hub"));
    }

    // Check Library/Caches/huggingface (macOS)
    if let Some(cache_dir) = dirs::cache_dir() {
        let macos_path = cache_dir.join("huggingface").join("hub");
        if !dirs.contains(&macos_path) {
            dirs.push(macos_path);
        }
    }

    dirs
}

/// Get model directory from DORA_MODELS_PATH or HuggingFace cache
fn get_model_dir(repo_id: &str) -> std::io::Result<PathBuf> {
    // Determine model name for dora path
    let dora_model_name = if repo_id.contains("8bit") {
        "qwen-image-2512-8bit"
    } else {
        "qwen-image-2512-4bit"
    };

    // Check DORA_MODELS_PATH environment variable first
    if let Ok(models_path) = std::env::var("DORA_MODELS_PATH") {
        let model_dir = PathBuf::from(models_path).join(dora_model_name);
        if model_dir.join("transformer").exists() {
            return Ok(model_dir);
        }
    }

    // Check default dora location
    if let Some(home) = dirs::home_dir() {
        let dora_path = home.join(".OminiX").join("models").join(dora_model_name);
        if dora_path.join("transformer").exists() {
            return Ok(dora_path);
        }
    }

    // Fall back to HuggingFace cache
    let cache_dirs = get_hf_cache_dirs();
    let repo_name = format!("models--{}", repo_id.replace('/', "--"));

    for cache_dir in &cache_dirs {
        let repo_dir = cache_dir.join(&repo_name);
        let snapshots_dir = repo_dir.join("snapshots");

        if snapshots_dir.exists() {
            let mut entries: Vec<_> = std::fs::read_dir(&snapshots_dir)?
                .filter_map(|e| e.ok())
                .filter(|e| {
                    let has_text_encoder = e.path().join("text_encoder").exists();
                    let has_tokenizer = e.path().join("tokenizer").exists();
                    has_text_encoder && has_tokenizer
                })
                .collect();

            if entries.is_empty() {
                entries = std::fs::read_dir(&snapshots_dir)?
                    .filter_map(|e| e.ok())
                    .collect();
            }

            entries.sort_by_key(|e| e.path());

            if let Some(entry) = entries.last() {
                return Ok(entry.path());
            }
        }
    }

    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!(
            "Model not found. Please either:\n  \
             1. Set DORA_MODELS_PATH and place model in $DORA_MODELS_PATH/{}/\n  \
             2. Place model in ~/.OminiX/models/{}/\n  \
             3. Download with: huggingface-cli download {}",
            dora_model_name, dora_model_name, repo_id
        ),
    ))
}

/// Load safetensors weights from a single file
fn load_safetensors<P: AsRef<std::path::Path>>(path: P) -> Result<HashMap<String, Array>, Box<dyn std::error::Error>> {
    use safetensors::SafeTensors;

    let data = std::fs::read(path)?;
    let tensors = SafeTensors::deserialize(&data)?;

    let mut weights = HashMap::new();
    for (name, tensor) in tensors.tensors() {
        let array = Array::try_from(tensor)?;
        weights.insert(name.to_string(), array);
    }

    Ok(weights)
}

/// Load safetensors weights from multiple shards
fn load_sharded_weights(paths: &[PathBuf]) -> Result<HashMap<String, Array>, Box<dyn std::error::Error>> {
    let mut all_weights = HashMap::new();

    for path in paths {
        println!("  Loading {} ...", path.display());
        let weights = load_safetensors(path)?;
        all_weights.extend(weights);
    }

    Ok(all_weights)
}

/// Load tokenizer from model directory
fn load_tokenizer(model_dir: &std::path::Path) -> Result<Tokenizer, Box<dyn std::error::Error>> {
    let tokenizer_path = model_dir.join("tokenizer/tokenizer.json");
    if !tokenizer_path.exists() {
        return Err(format!("Tokenizer not found at: {}", tokenizer_path.display()).into());
    }
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("Failed to load tokenizer: {}", e))?;
    Ok(tokenizer)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("=== Qwen-Image MLX Generator ===");
    println!("Prompt: {}", args.prompt);
    println!("Size: {}x{}", args.width, args.height);
    println!("Steps: {}", args.steps);
    println!();

    // Select model based on quantization bits or custom path
    let model_dir = if let Some(custom_path) = &args.model_path {
        println!("Using custom model path: {}", custom_path.display());
        custom_path.clone()
    } else {
        let repo_id = if args.use_8bit {
            "mlx-community/Qwen-Image-2512-8bit"
        } else {
            "mlx-community/Qwen-Image-2512-4bit"
        };

        // Find model directory
        println!("Looking for model...");
        let dir = get_model_dir(repo_id)?;
        println!("  Found: {}", dir.display());
        dir
    };

    // Load transformer weights (8-bit or 4-bit)
    use qwen_image_mlx::{QwenQuantizedTransformer, QwenConfig, load_transformer_weights};

    // Load transformer weights from HuggingFace model (4-bit or 8-bit)
    let transformer_dir = model_dir.join("transformer");
    let mut transformer_files: Vec<PathBuf> = Vec::new();
    for i in 0..20 {  // Check up to 20 shards (8-bit has 11 shards: 0-10)
        let path = transformer_dir.join(format!("{}.safetensors", i));
        if path.exists() {
            transformer_files.push(path);
        }
    }
    println!("  Found {} transformer shards", transformer_files.len());

    let bits = if args.use_8bit { 8 } else { 4 };
    println!("\nLoading {}-bit transformer weights...", bits);
    let transformer_weights = load_sharded_weights(&transformer_files)?;
    println!("  Loaded {} tensors", transformer_weights.len());

    let config = if args.use_8bit {
        QwenConfig::with_8bit()
    } else {
        QwenConfig::default()
    };

    // Create quantized transformer model
    println!("\nCreating quantized transformer model ({}-bit)...", config.quantization_bits);
    println!("  Config: {:?}", config);

    // Save quantization settings before moving config
    let quant_bits = config.quantization_bits;
    let quant_group_size = config.quantization_group_size;

    let mut transformer = QwenQuantizedTransformer::new(config)?;
    println!("  Model created successfully");

    // Load weights
    println!("\nLoading weights into model...");

    // Debug: print sample weight keys from file
    println!("  Sample weight keys from file:");
    let mut sorted_keys: Vec<_> = transformer_weights.keys().collect();
    sorted_keys.sort();
    for key in sorted_keys.iter().take(10) {
        let w = &transformer_weights[*key];
        println!("    {} {:?}", key, w.shape());
    }

    // Debug: print model parameter names
    use mlx_rs::module::ModuleParameters;
    let model_params = transformer.parameters();
    let flat_params = model_params.flatten();
    println!("\n  Sample model parameter names ({} total):", flat_params.len());
    let mut param_names: Vec<_> = flat_params.keys().collect();
    param_names.sort();
    for name in param_names.iter().take(10) {
        println!("    {}", name);
    }

    // Check for mismatches
    let weight_keys: std::collections::HashSet<_> = transformer_weights.keys().collect();
    let model_keys: std::collections::HashSet<String> = flat_params.keys().map(|k| k.to_string()).collect();
    let mut missing_in_model = 0;
    let mut missing_in_weights = 0;
    for k in weight_keys.iter() {
        if !model_keys.contains(*k) {
            if missing_in_model == 0 {
                println!("\n  First few weights missing in model:");
            }
            if missing_in_model < 5 {
                println!("    {}", k);
            }
            missing_in_model += 1;
        }
    }
    for k in model_keys.iter() {
        if !weight_keys.iter().any(|wk| *wk == k) {
            if missing_in_weights == 0 {
                println!("\n  First few model params missing in weights:");
            }
            if missing_in_weights < 5 {
                println!("    {}", k);
            }
            missing_in_weights += 1;
        }
    }
    println!("\n  Weights missing in model: {}, Model params missing in weights: {}",
        missing_in_model, missing_in_weights);

    load_transformer_weights(&mut transformer, transformer_weights)?;
    println!("  Weights loaded successfully!");

    // Debug: verify img_in weight shape (confirms correct bits)
    {
        let params = transformer.parameters().flatten();
        for (name, param) in params.iter() {
            if name.as_ref() == "img_in.inner.weight" {
                let p: &mlx_rs::Array = param;
                let expected_cols = 64 / (32 / quant_bits);  // 8 for 4-bit, 16 for 8-bit
                println!("  [VERIFY] img_in.weight shape: {:?} (expected cols={} for {}-bit)",
                    p.shape(), expected_cols, quant_bits);
                if p.dim(1) != expected_cols {
                    println!("  [WARNING] Weight shape mismatch! Model may produce incorrect output.");
                }
                break;
            }
        }
    }

    // Debug: verify timestep embedder weights and dequantize
    {
        let params = transformer.parameters().flatten();
        let mut weight: Option<mlx_rs::Array> = None;
        let mut scales: Option<mlx_rs::Array> = None;
        let mut biases: Option<mlx_rs::Array> = None;

        for (name, param) in params.iter() {
            if name.contains("timestep_embedder") && name.contains("linear_2") {
                let p: &mlx_rs::Array = param;
                mlx_rs::transforms::eval([p]).ok();
                println!("  [VERIFY] {}: shape={:?}, dtype={:?}",
                    name, p.shape(), p.dtype());

                if name.ends_with(".inner.weight") {
                    weight = Some(p.clone());
                } else if name.ends_with(".scales") {
                    scales = Some(p.clone());
                } else if name.ends_with(".biases") {
                    biases = Some(p.clone());
                }
            }
        }

        // Dequantize and check values
        if let (Some(w), Some(s), Some(b)) = (weight, scales, biases) {
            if let Ok(dequant) = mlx_rs::ops::dequantize(&w, &s, &b, quant_group_size, quant_bits, None::<&str>) {
                mlx_rs::transforms::eval([&dequant]).ok();
                println!("  [DEQUANT] linear_2 weight: shape={:?}, min={:.4}, max={:.4}",
                    dequant.shape(),
                    dequant.min(None).unwrap().item::<f32>(),
                    dequant.max(None).unwrap().item::<f32>());
            }
        }

        // Also check bias
        for (name, param) in params.iter() {
            if name.as_ref() == "time_text_embed.timestep_embedder.linear_2.inner.bias" {
                let p: &mlx_rs::Array = param;
                mlx_rs::transforms::eval([p]).ok();
                println!("  [BIAS] linear_2 bias: min={:.4}, max={:.4}, mean={:.4}",
                    p.min(None).unwrap().item::<f32>(),
                    p.max(None).unwrap().item::<f32>(),
                    p.mean(None).unwrap().item::<f32>());
            }
        }
    }


    // Get text embeddings (conditional and unconditional for CFG)
    // Returns (cond_states, uncond_states, cond_mask, uncond_mask, cond_seq_len, uncond_seq_len)
    let (cond_hidden_states, uncond_hidden_states, _cond_mask, _uncond_mask, cond_txt_len, uncond_txt_len) = if args.skip_text_encoder {
        println!("\n=== Using Dummy Text Embeddings ===");
        let seed = args.seed.unwrap_or(42);
        let txt_key = mlx_rs::random::key(seed + 1)?;
        let txt_len = 11usize;  // Use same length as typical prompt
        let cond = mlx_rs::random::normal::<f32>(&[1, txt_len as i32, 3584], None, None, Some(&txt_key))?;
        let uncond = mlx_rs::Array::zeros::<f32>(&[1, txt_len as i32, 3584])?;
        // For dummy, use all ones mask (all real)
        let cond_mask = Array::ones::<f32>(&[1, txt_len as i32])?;
        let uncond_mask = Array::ones::<f32>(&[1, txt_len as i32])?;
        (cond, uncond, cond_mask, uncond_mask, txt_len, txt_len)
    } else {
        println!("\n=== Loading Text Encoder ===");

        // Load tokenizer
        println!("Loading tokenizer...");
        let tokenizer = load_tokenizer(&model_dir)?;
        println!("  Tokenizer loaded");

        // Use the Qwen VL template format (matching mflux)
        // The template adds 34 tokens of system/user wrapper that get dropped later
        let template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n";

        // Tokenize the prompt (conditional)
        println!("Tokenizing prompt with template: \"{}\"", args.prompt);
        let formatted_prompt = template.replace("{}", &args.prompt);
        let encoding = tokenizer.encode(formatted_prompt.as_str(), false)
            .map_err(|e| format!("Tokenization error: {}", e))?;
        let token_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
        let cond_total_tokens = token_ids.len();
        println!("  Total tokens (with template): {}", cond_total_tokens);

        // mflux uses max_length of 77 + 34 = 111, then drops first 34 after encoding
        // This leaves 77 tokens for the actual output
        let max_input_len = 77 + 34;  // 111
        let drop_idx = 34;  // Skip first 34 template tokens
        let max_output_len = 77;

        let mut padded_ids = token_ids.clone();
        if padded_ids.len() > max_input_len {
            padded_ids.truncate(max_input_len);
        }
        while padded_ids.len() < max_input_len {
            padded_ids.push(0);
        }

        // After dropping 34 tokens, we have: cond_total_tokens - 34 real tokens (or max 77)
        let cond_num_output_tokens = (cond_total_tokens.saturating_sub(drop_idx)).min(max_output_len);

        // Create attention mask for OUTPUT (after dropping 34): 1 for real tokens, 0 for padding
        let cond_mask_data: Vec<f32> = (0..max_output_len)
            .map(|i| if i < cond_num_output_tokens { 1.0 } else { 0.0 })
            .collect();
        let _cond_mask = Array::from_slice(&cond_mask_data, &[1, max_output_len as i32]);

        // Tokenize empty prompt for unconditional (CFG) - use same template with space
        println!("Tokenizing empty prompt for CFG...");
        let formatted_uncond = template.replace("{}", " ");
        let uncond_encoding = tokenizer.encode(formatted_uncond.as_str(), false)
            .map_err(|e| format!("Tokenization error: {}", e))?;
        let uncond_total_tokens = uncond_encoding.get_ids().len();
        let mut uncond_ids: Vec<i32> = uncond_encoding.get_ids().iter().map(|&id| id as i32).collect();
        while uncond_ids.len() < max_input_len {
            uncond_ids.push(0);
        }
        if uncond_ids.len() > max_input_len {
            uncond_ids.truncate(max_input_len);
        }

        let uncond_num_output_tokens = (uncond_total_tokens.saturating_sub(drop_idx)).min(max_output_len);

        // Create unconditional mask for OUTPUT
        let uncond_mask_data: Vec<f32> = (0..max_output_len)
            .map(|i| if i < uncond_num_output_tokens { 1.0 } else { 0.0 })
            .collect();
        let _uncond_mask = Array::from_slice(&uncond_mask_data, &[1, max_output_len as i32]);

        println!("  Conditional: {} input tokens -> {} output tokens (after dropping {})",
            cond_total_tokens, cond_num_output_tokens, drop_idx);
        println!("  Unconditional: {} input tokens -> {} output tokens",
            uncond_total_tokens, uncond_num_output_tokens);

        let cond_input_ids = Array::from_slice(&padded_ids, &[1, max_input_len as i32]);
        let uncond_input_ids = Array::from_slice(&uncond_ids, &[1, max_input_len as i32]);

        // Create attention masks (1 for valid, 0 for padding) - same as mflux
        // Note: We only have cond_total_tokens valid tokens, rest are padding
        let cond_attn_mask: Vec<i32> = (0..max_input_len)
            .map(|i| if i < cond_total_tokens { 1 } else { 0 })
            .collect();
        let cond_attn_mask = Array::from_slice(&cond_attn_mask, &[1, max_input_len as i32]);

        let uncond_attn_mask: Vec<i32> = (0..max_input_len)
            .map(|i| if i < uncond_total_tokens { 1 } else { 0 })
            .collect();
        let uncond_attn_mask = Array::from_slice(&uncond_attn_mask, &[1, max_input_len as i32]);

        // Load text encoder
        println!("\nLoading text encoder...");
        let mut text_encoder = qwen_image_mlx::load_text_encoder(&model_dir)?;
        println!("  Text encoder loaded successfully!");

        // Encode prompts with attention mask (causal attention + padding mask like mflux)
        println!("\nEncoding prompts with causal attention...");
        let start = std::time::Instant::now();
        let cond_states_full = text_encoder.forward_with_mask(&cond_input_ids, &cond_attn_mask)?;
        let uncond_states_full = text_encoder.forward_with_mask(&uncond_input_ids, &uncond_attn_mask)?;
        mlx_rs::transforms::eval([&cond_states_full, &uncond_states_full])?;
        println!("  Text encoding completed in {:.2?}", start.elapsed());
        println!("  Encoded shapes (before drop): cond={:?}, uncond={:?}",
            cond_states_full.shape(), uncond_states_full.shape());

        // Debug: print raw text encoder output range (should match mflux's ~[-150, 150])
        println!("  Raw cond encoder output: min={:.2}, max={:.2}, mean={:.4}",
            cond_states_full.min(None)?.item::<f32>(),
            cond_states_full.max(None)?.item::<f32>(),
            cond_states_full.mean(None)?.item::<f32>());
        println!("  Raw uncond encoder output: min={:.2}, max={:.2}, mean={:.4}",
            uncond_states_full.min(None)?.item::<f32>(),
            uncond_states_full.max(None)?.item::<f32>(),
            uncond_states_full.mean(None)?.item::<f32>());

        // Match mflux's _process_text_embeddings_mlx:
        // 1. Extract only valid tokens (cond_total_tokens from encoder)
        // 2. Drop first 34 (template)
        // 3. NO padding - use actual valid length like mflux

        // For conditional: extract tokens 34 to cond_total_tokens
        let cond_valid_end = cond_total_tokens.min(max_input_len);
        let cond_valid_start = drop_idx;
        let cond_valid_len = cond_valid_end.saturating_sub(cond_valid_start);

        let cond_states = cond_states_full.index((.., cond_valid_start as i32..cond_valid_end as i32, ..));

        // For unconditional
        let uncond_valid_end = uncond_total_tokens.min(max_input_len);
        let uncond_valid_start = drop_idx;
        let uncond_valid_len = uncond_valid_end.saturating_sub(uncond_valid_start);

        let uncond_states = uncond_states_full.index((.., uncond_valid_start as i32..uncond_valid_end as i32, ..));

        // Cast to bfloat16 to match mflux's _process_text_embeddings_mlx which does:
        // prompt_embeds = prompt_embeds.astype(dtype) where dtype=mx.bfloat16
        let cond_states = cond_states.as_dtype(mlx_rs::Dtype::Bfloat16)?;
        let uncond_states = uncond_states.as_dtype(mlx_rs::Dtype::Bfloat16)?;

        mlx_rs::transforms::eval([&cond_states, &uncond_states])?;
        println!("  After processing (cast to bfloat16): cond={:?} ({} valid), uncond={:?} ({} valid)",
            cond_states.shape(), cond_valid_len,
            uncond_states.shape(), uncond_valid_len);
        println!("  Dtype: cond={:?}, uncond={:?}", cond_states.dtype(), uncond_states.dtype());

        // Debug: print embeddings range
        println!("  Cond embeddings: min={:.3}, max={:.3}, mean={:.3}",
            cond_states.min(None)?.item::<f32>(),
            cond_states.max(None)?.item::<f32>(),
            cond_states.mean(None)?.item::<f32>());
        println!("  Uncond embeddings: min={:.3}, max={:.3}, mean={:.3}",
            uncond_states.min(None)?.item::<f32>(),
            uncond_states.max(None)?.item::<f32>(),
            uncond_states.mean(None)?.item::<f32>());

        // Masks are now all 1s since we only have valid tokens
        // mflux returns None for mask when all positions are valid
        // For cond: [1, cond_valid_len], all 1s
        // For uncond: [1, uncond_valid_len], all 1s
        let cond_mask = Array::ones::<f32>(&[1, cond_valid_len as i32])?;
        let uncond_mask = Array::ones::<f32>(&[1, uncond_valid_len as i32])?;

        // Store the text sequence lengths for RoPE computation
        (cond_states, uncond_states, cond_mask, uncond_mask, cond_valid_len, uncond_valid_len)
    };

    // CFG scale
    let cfg_scale = args.guidance;

    // Generate image
    println!("\n=== Generating Image ===");

    // Image parameters
    let height = args.height;
    let width = args.width;
    let num_steps = args.steps;
    let latent_h = height / 16;
    let latent_w = width / 16;
    let num_patches = latent_h * latent_w;

    // Generate RoPE embeddings matching mflux's QwenEmbedRopeMLX with scale_rope=True
    // Config: theta=10000, axes_dim=[16, 56, 56], scale_rope=True
    // With scale_rope=True, positions are CENTERED: [-h/2, ..., +h/2-1]
    let theta = 10000.0f32;
    let axes_dim = [16i32, 56i32, 56i32];  // frame, height, width dimensions

    // Precompute frequencies for each axis (half the dim for complex pairs)
    fn compute_freqs(dim: i32, theta: f32) -> Vec<f32> {
        (0..dim/2).map(|i| {
            let scale = (i as f32 * 2.0) / dim as f32;
            1.0 / theta.powf(scale)
        }).collect()
    }

    let frame_freqs = compute_freqs(axes_dim[0], theta);
    let height_freqs = compute_freqs(axes_dim[1], theta);
    let width_freqs = compute_freqs(axes_dim[2], theta);

    // Precompute cos/sin lookup tables for positive and negative indices
    // Like mflux: pos_freqs for indices 0..4096, neg_freqs for indices -1..-4096
    let max_pos = 4096i32;
    fn compute_rope_table(freqs: &[f32], max_len: i32) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let half_dim = freqs.len();
        let mut pos_cos = vec![vec![0.0f32; half_dim]; max_len as usize];
        let mut pos_sin = vec![vec![0.0f32; half_dim]; max_len as usize];

        for idx in 0..max_len {
            for (f_idx, &freq) in freqs.iter().enumerate() {
                let angle = (idx as f32) * freq;
                pos_cos[idx as usize][f_idx] = angle.cos();
                pos_sin[idx as usize][f_idx] = angle.sin();
            }
        }
        (pos_cos, pos_sin)
    }

    fn compute_neg_rope_table(freqs: &[f32], max_len: i32) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let half_dim = freqs.len();
        let mut neg_cos = vec![vec![0.0f32; half_dim]; max_len as usize];
        let mut neg_sin = vec![vec![0.0f32; half_dim]; max_len as usize];

        for idx in 0..max_len {
            // neg_index[i] = -(i+1) for i in 0..max_len, then reversed
            // So neg_index = [-1, -2, ..., -max_len][::-1] = [-max_len, ..., -2, -1]
            let neg_idx = -((max_len - idx) as f32);
            for (f_idx, &freq) in freqs.iter().enumerate() {
                let angle = neg_idx * freq;
                neg_cos[idx as usize][f_idx] = angle.cos();
                neg_sin[idx as usize][f_idx] = angle.sin();
            }
        }
        (neg_cos, neg_sin)
    }

    let (pos_frame_cos, pos_frame_sin) = compute_rope_table(&frame_freqs, max_pos);
    let (pos_height_cos, pos_height_sin) = compute_rope_table(&height_freqs, max_pos);
    let (pos_width_cos, pos_width_sin) = compute_rope_table(&width_freqs, max_pos);
    let (neg_height_cos, neg_height_sin) = compute_neg_rope_table(&height_freqs, max_pos);
    let (neg_width_cos, neg_width_sin) = compute_neg_rope_table(&width_freqs, max_pos);

    // Compute video/image RoPE frequencies with scale_rope=True
    // Frame: always uses positive indices starting from 0
    // Height/Width: CENTERED using negative for first half, positive for second half
    let frame = 1;  // Single image
    let half_height = (latent_h / 2) as usize;
    let half_width = (latent_w / 2) as usize;

    // Build RoPE for each patch position
    let mut img_cos_data: Vec<f32> = Vec::with_capacity((num_patches * 64) as usize);
    let mut img_sin_data: Vec<f32> = Vec::with_capacity((num_patches * 64) as usize);

    for f in 0..frame {
        for h in 0..latent_h as usize {
            for w in 0..latent_w as usize {
                // Frame: use positive index f
                let frame_cos = &pos_frame_cos[f];
                let frame_sin = &pos_frame_sin[f];

                // Height: scale_rope=True, centered positions
                // First half (h < half_height): use neg_freqs[-(latent_h - half_height) + h]
                // Second half (h >= half_height): use pos_freqs[h - half_height]
                let (height_cos, height_sin) = if h < half_height {
                    // Negative index region
                    let neg_idx = (max_pos as usize) - (latent_h as usize - half_height) + h;
                    (&neg_height_cos[neg_idx], &neg_height_sin[neg_idx])
                } else {
                    // Positive index region
                    let pos_idx = h - half_height;
                    (&pos_height_cos[pos_idx], &pos_height_sin[pos_idx])
                };

                // Width: same logic
                let (width_cos, width_sin) = if w < half_width {
                    let neg_idx = (max_pos as usize) - (latent_w as usize - half_width) + w;
                    (&neg_width_cos[neg_idx], &neg_width_sin[neg_idx])
                } else {
                    let pos_idx = w - half_width;
                    (&pos_width_cos[pos_idx], &pos_width_sin[pos_idx])
                };

                // Concatenate [frame, height, width] freqs -> [8, 28, 28] = 64
                img_cos_data.extend(frame_cos);
                img_cos_data.extend(height_cos);
                img_cos_data.extend(width_cos);

                img_sin_data.extend(frame_sin);
                img_sin_data.extend(height_sin);
                img_sin_data.extend(width_sin);
            }
        }
    }

    let img_cos = Array::from_slice(&img_cos_data, &[num_patches, 64]);
    let img_sin = Array::from_slice(&img_sin_data, &[num_patches, 64]);

    // Text RoPE: positions start after max_vid_index
    // With scale_rope=True, max_vid_index = max(height/2, width/2)
    // mflux computes text RoPE for max(txt_seq_lens) across batch
    let max_vid_index = (half_height.max(half_width)) as i32;
    let max_txt_len = cond_txt_len.max(uncond_txt_len);

    // Text uses all 3 axes but with flat positions [max_vid_index..max_vid_index+max_txt_len]
    let mut txt_cos_data: Vec<f32> = Vec::with_capacity(max_txt_len * 64);
    let mut txt_sin_data: Vec<f32> = Vec::with_capacity(max_txt_len * 64);

    for i in 0..max_txt_len {
        let pos = (max_vid_index as usize) + i;
        // Text uses positive indices for all axes
        txt_cos_data.extend(&pos_frame_cos[pos]);
        txt_cos_data.extend(&pos_height_cos[pos]);
        txt_cos_data.extend(&pos_width_cos[pos]);

        txt_sin_data.extend(&pos_frame_sin[pos]);
        txt_sin_data.extend(&pos_height_sin[pos]);
        txt_sin_data.extend(&pos_width_sin[pos]);
    }

    // Create RoPE arrays - we'll slice to actual length when needed
    let txt_cos_full = Array::from_slice(&txt_cos_data, &[max_txt_len as i32, 64]);
    let txt_sin_full = Array::from_slice(&txt_sin_data, &[max_txt_len as i32, 64]);

    // Slice to actual lengths for cond and uncond
    let cond_txt_cos = txt_cos_full.index((..cond_txt_len as i32, ..));
    let cond_txt_sin = txt_sin_full.index((..cond_txt_len as i32, ..));
    let uncond_txt_cos = txt_cos_full.index((..uncond_txt_len as i32, ..));
    let uncond_txt_sin = txt_sin_full.index((..uncond_txt_len as i32, ..));

    println!("RoPE embeddings generated");
    println!("  Image RoPE: {:?}", img_cos.shape());
    println!("  Cond text RoPE: {:?}", cond_txt_cos.shape());
    println!("  Uncond text RoPE: {:?}", uncond_txt_cos.shape());
    let num_patches = latent_h * latent_w;

    println!("Image size: {}x{}", width, height);
    println!("Latent size: {}x{}", latent_w, latent_h);
    println!("Num patches: {}", num_patches);
    println!("Steps: {}", num_steps);

    // Create random latents [batch=1, num_patches, patch_dim=64]
    let seed = args.seed.unwrap_or(42);
    println!("Seed: {}", seed);

    let key = mlx_rs::random::key(seed)?;
    let mut latents = mlx_rs::random::normal::<f32>(&[1, num_patches, 64], None, None, Some(&key))?;

    // Dynamic flow matching schedule (Qwen-Image-2512 scheduler config)
    // Uses exponential time shift with resolution-dependent mu
    const BASE_SHIFT: f32 = 0.5;
    const MAX_SHIFT: f32 = 0.9;
    const BASE_IMAGE_SEQ_LEN: f32 = 256.0;
    const MAX_IMAGE_SEQ_LEN: f32 = 8192.0;

    // Calculate mu (shift amount) based on image sequence length
    // Linear interpolation: smaller images -> base_shift, larger images -> max_shift
    fn calculate_shift(image_seq_len: i32) -> f32 {
        let image_seq_len = image_seq_len as f32;
        let m = (MAX_SHIFT - BASE_SHIFT) / (MAX_IMAGE_SEQ_LEN - BASE_IMAGE_SEQ_LEN);
        let b = BASE_SHIFT - m * BASE_IMAGE_SEQ_LEN;
        image_seq_len * m + b
    }

    // Exponential time shift (Qwen-Image uses "exponential" type)
    // Formula: exp(mu) / (exp(mu) + (1/t - 1)^sigma)
    fn time_shift_exponential(mu: f32, sigma: f32, t: f32) -> f32 {
        if t <= 0.0 || t >= 1.0 {
            return t;  // Don't shift endpoints
        }
        let exp_mu = mu.exp();
        exp_mu / (exp_mu + (1.0 / t - 1.0).powf(sigma))
    }

    // Calculate dynamic shift based on number of patches
    let mu = calculate_shift(num_patches);
    println!("Dynamic scheduler: mu = {:.4} for {} patches", mu, num_patches);

    // Generate sigmas with dynamic exponential shift
    let sigmas: Vec<f32> = (0..=num_steps).map(|i| {
        let t = 1.0 - (i as f32 / num_steps as f32);  // Linear from 1.0 to 0.0
        time_shift_exponential(mu, 1.0, t)
    }).collect();

    // Debug: print first few sigmas
    println!("Sigma schedule (first 5): {:?}", &sigmas[..5.min(sigmas.len())]);

    println!("\nRunning diffusion loop...");
    let start = std::time::Instant::now();

    for step in 0..num_steps {
        let sigma = sigmas[step as usize];
        let sigma_next = sigmas[(step + 1) as usize];

        // Timestep for this step
        let timestep = mlx_rs::Array::from_slice(&[sigma], &[1]);

        // Get velocity predictions for CFG (conditional and unconditional)
        // Use separate RoPE for cond/uncond text (different lengths)
        // Pass None for mask since all positions are valid (like mflux)
        let cond_velocity = transformer.forward(
            &latents,
            &cond_hidden_states,
            &timestep,
            Some((&img_cos, &img_sin)),
            Some((&cond_txt_cos, &cond_txt_sin)),
            None,  // No mask needed - all positions are valid
        )?;

        let uncond_velocity = transformer.forward(
            &latents,
            &uncond_hidden_states,
            &timestep,
            Some((&img_cos, &img_sin)),
            Some((&uncond_txt_cos, &uncond_txt_sin)),
            None,  // No mask needed - all positions are valid
        )?;

        // Apply normalized CFG (matching mflux compute_guided_noise):
        // combined = uncond + cfg_scale * (cond - uncond)
        // Then rescale combined to have the same norm as cond
        let velocity_diff = mlx_rs::ops::subtract(&cond_velocity, &uncond_velocity)?;
        let cfg_arr = Array::from_f32(cfg_scale);
        let scaled_diff = mlx_rs::ops::multiply(&velocity_diff, &cfg_arr)?;
        let combined = mlx_rs::ops::add(&uncond_velocity, &scaled_diff)?;

        // Compute norms for rescaling (along last axis)
        let eps = Array::from_f32(1e-12);
        let cond_sq = mlx_rs::ops::multiply(&cond_velocity, &cond_velocity)?;
        let cond_sum_sq = mlx_rs::ops::sum_axis(&cond_sq, -1, true)?;
        let cond_norm = mlx_rs::ops::sqrt(&mlx_rs::ops::add(&cond_sum_sq, &eps)?)?;

        let combined_sq = mlx_rs::ops::multiply(&combined, &combined)?;
        let combined_sum_sq = mlx_rs::ops::sum_axis(&combined_sq, -1, true)?;
        let combined_norm = mlx_rs::ops::sqrt(&mlx_rs::ops::add(&combined_sum_sq, &eps)?)?;

        // Rescale: velocity = combined * (cond_norm / combined_norm)
        let scale_factor = mlx_rs::ops::divide(&cond_norm, &combined_norm)?;
        let velocity = mlx_rs::ops::multiply(&combined, &scale_factor)?;

        // Euler step: latents = latents + (sigma_next - sigma) * velocity
        let dt = mlx_rs::Array::from_f32(sigma_next - sigma);
        let delta = mlx_rs::ops::multiply(&velocity, &dt)?;
        latents = mlx_rs::ops::add(&latents, &delta)?;

        // Print progress (minimal overhead - no eval needed)
        if (step + 1) % 5 == 0 || step == 0 {
            println!("  Step {}/{} (sigma: {:.3})", step + 1, num_steps, sigma);
        }
    }

    // Force evaluation to get accurate timing for diffusion vs VAE
    mlx_rs::transforms::eval([&latents])?;

    let gen_elapsed = start.elapsed();
    println!("Diffusion completed in {:.2?}", gen_elapsed);
    println!("  {:.2?} per step", gen_elapsed / num_steps as u32);

    // Unpatchify latents: [1, num_patches, 64] -> [1, 16, vae_h, vae_w]
    println!("\nUnpatchifying latents...");
    let patch_size = 2i32;
    let out_channels = 16i32;
    let vae_h = latent_h * patch_size;  // 64
    let vae_w = latent_w * patch_size;  // 64

    // Unpatchify: [1, num_patches, 64] -> [1, H_p, W_p, C, p_h, p_w] -> [1, C, H, W]
    let latents_reshaped = latents.reshape(&[1, latent_h, latent_w, out_channels, patch_size, patch_size])?;
    let latents_permuted = latents_reshaped.transpose_axes(&[0, 3, 1, 4, 2, 5])?;
    let vae_latents = latents_permuted.reshape(&[1, out_channels, vae_h, vae_w])?;
    mlx_rs::transforms::eval([&vae_latents])?;
    println!("  VAE latent shape: {:?}", vae_latents.shape());
    println!("  VAE latent range: [{:.3}, {:.3}], mean={:.4}",
        vae_latents.min(None)?.item::<f32>(),
        vae_latents.max(None)?.item::<f32>(),
        vae_latents.mean(None)?.item::<f32>());

    // Load VAE decoder
    println!("\nLoading VAE decoder...");
    let mut vae = qwen_image_mlx::load_vae_from_dir(&model_dir)?;

    // Decode latents to image
    println!("\nDecoding latents to image...");
    let decode_start = std::time::Instant::now();

    // Denormalize latents (VAE expects denormalized input)
    let denorm_latents = qwen_image_mlx::QwenVAE::denormalize_latent(&vae_latents)?;

    // Decode: [1, 16, 64, 64] -> [1, 3, 512, 512]
    let decoded = vae.decode(&denorm_latents)?;
    mlx_rs::transforms::eval([&decoded])?;

    println!("  Decoded in {:.2?}", decode_start.elapsed());
    println!("  Output shape: {:?}", decoded.shape());

    // Convert to RGB image and save
    println!("\nSaving image...");

    // decoded: [1, 3, H, W] - convert to RGB image
    // Clamp to [-1, 1] and rescale to [0, 255]
    let img = decoded.index((0, .., .., ..));  // [3, H, W]
    let img = mlx_rs::ops::clip(&img, (-1.0f32, 1.0f32))?;
    let img = mlx_rs::ops::add(&img, &Array::from_f32(1.0))?;  // [0, 2]
    let img = mlx_rs::ops::multiply(&img, &Array::from_f32(127.5))?;  // [0, 255]
    let img = img.as_dtype(mlx_rs::Dtype::Uint8)?;

    // Transpose from [3, H, W] to [H, W, 3] for PPM
    let img = img.transpose_axes(&[1, 2, 0])?;
    mlx_rs::transforms::eval([&img])?;

    let img_shape = img.shape();
    let img_h = img_shape[0] as u32;
    let img_w = img_shape[1] as u32;

    // Force contiguous memory layout (transpose creates strided view, as_slice ignores strides)
    let numel = img_h as i32 * img_w as i32 * 3;
    let img = img.reshape(&[numel])?;
    let img = img.reshape(&[img_h as i32, img_w as i32, 3])?;
    mlx_rs::transforms::eval([&img])?;

    // Save as PNG using the image crate
    let img_data: Vec<u8> = img.as_slice().to_vec();

    use image::RgbImage;
    let rgb_image = RgbImage::from_raw(img_w, img_h, img_data)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Failed to create image from data"))?;
    rgb_image.save(&args.output)?;

    println!("Saved image to: {}", args.output.display());
    println!("  Image size: {}x{}", img_w, img_h);

    println!("\n=== Generation Complete ===");

    Ok(())
}
