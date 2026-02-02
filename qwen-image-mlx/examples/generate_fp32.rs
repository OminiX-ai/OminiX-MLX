//! Full Precision Qwen-Image generation example
//!
//! Uses full precision BF16 weights for highest quality.
//!
//! Model path can be set via environment variable:
//!   export DORA_MODELS_PATH=~/.dora/models
//!
//! Expected directory structure:
//!   $DORA_MODELS_PATH/qwen-image-2512/
//!   ├── transformer/    (full precision BF16 weights)
//!   ├── text_encoder/   (MLX format)
//!   ├── vae/            (MLX format)
//!   └── tokenizer/
//!
//! Usage:
//!   cargo run --release --example generate_fp32 -- --prompt "a fluffy cat"

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use image::{ImageBuffer, Rgb};
use mlx_rs::Array;
use mlx_rs::ops::indexing::IndexOp;
use safetensors::SafeTensors;
use tokenizers::Tokenizer;

// For cache clearing
extern crate mlx_sys;

use qwen_image_mlx::qwen_full_precision::{QwenFullConfig, QwenFullTransformer, load_full_precision_weights};

#[derive(Parser, Debug)]
#[command(name = "generate_fp32")]
#[command(about = "Generate images with full precision Qwen-Image")]
struct Args {
    /// Text prompt for image generation
    #[arg(short, long, default_value = "a fluffy cat")]
    prompt: String,

    /// Output image path
    #[arg(short, long, default_value = "output_fp32.png")]
    output: PathBuf,

    /// Image height (must be divisible by 16)
    #[arg(long, default_value_t = 512)]
    height: i32,

    /// Image width (must be divisible by 16)
    #[arg(long, default_value_t = 512)]
    width: i32,

    /// Number of inference steps
    #[arg(long, default_value_t = 20)]
    steps: i32,

    /// Guidance scale for CFG
    #[arg(long, default_value_t = 5.0)]
    guidance: f32,

    /// Random seed
    #[arg(long)]
    seed: Option<u64>,
}

/// Get model directory from DORA_MODELS_PATH environment variable or default location
fn get_model_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    // Check environment variable first
    if let Ok(models_path) = std::env::var("DORA_MODELS_PATH") {
        let model_dir = PathBuf::from(models_path).join("qwen-image-2512");
        if model_dir.join("transformer").exists() {
            return Ok(model_dir);
        }
    }

    // Fall back to default location
    let home = std::env::var("HOME")?;
    let default_path = PathBuf::from(format!("{}/.dora/models/qwen-image-2512", home));
    if default_path.join("transformer").exists() {
        return Ok(default_path);
    }

    Err("Model not found. Please set DORA_MODELS_PATH or place models in ~/.dora/models/qwen-image-2512/\n\
         Expected structure:\n\
         qwen-image-2512/\n\
         ├── transformer/\n\
         ├── text_encoder/\n\
         ├── vae/\n\
         └── tokenizer/".into())
}

fn load_safetensors_weights(dir: &Path) -> Result<HashMap<String, Array>, Box<dyn std::error::Error>> {
    let mut weights = HashMap::new();

    for i in 1..=9 {
        let filename = format!("diffusion_pytorch_model-0000{}-of-00009.safetensors", i);
        let path = dir.join(&filename);
        if !path.exists() {
            continue;
        }

        println!("  Loading {}...", filename);
        let data = std::fs::read(&path)?;
        let tensors = SafeTensors::deserialize(&data)?;

        for (name, view) in tensors.tensors() {
            let shape: Vec<i32> = view.shape().iter().map(|&s| s as i32).collect();
            let dtype = view.dtype();

            let array = match dtype {
                safetensors::Dtype::BF16 => {
                    let bf16_data = view.data();
                    let f32_data: Vec<f32> = bf16_data
                        .chunks(2)
                        .map(|chunk| {
                            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                            half::bf16::from_bits(bits).to_f32()
                        })
                        .collect();
                    Array::from_slice(&f32_data, &shape)
                }
                safetensors::Dtype::F32 => {
                    let f32_data: Vec<f32> = view
                        .data()
                        .chunks(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect();
                    Array::from_slice(&f32_data, &shape)
                }
                safetensors::Dtype::F16 => {
                    let f16_data = view.data();
                    let f32_data: Vec<f32> = f16_data
                        .chunks(2)
                        .map(|chunk| {
                            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                            half::f16::from_bits(bits).to_f32()
                        })
                        .collect();
                    Array::from_slice(&f32_data, &shape)
                }
                _ => {
                    eprintln!("  Warning: Unsupported dtype {:?} for {}", dtype, name);
                    continue;
                }
            };

            weights.insert(name.to_string(), array);
        }
    }

    Ok(weights)
}

fn load_safetensors<P: AsRef<std::path::Path>>(path: P) -> Result<HashMap<String, Array>, Box<dyn std::error::Error>> {
    let data = std::fs::read(path)?;
    let tensors = SafeTensors::deserialize(&data)?;

    let mut weights = HashMap::new();
    for (name, tensor) in tensors.tensors() {
        let array = Array::try_from(tensor)?;
        weights.insert(name.to_string(), array);
    }

    Ok(weights)
}

fn load_sharded_weights(paths: &[PathBuf]) -> Result<HashMap<String, Array>, Box<dyn std::error::Error>> {
    let mut all_weights = HashMap::new();

    for path in paths {
        println!("  Loading {} ...", path.display());
        let weights = load_safetensors(path)?;
        all_weights.extend(weights);
    }

    Ok(all_weights)
}

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

    println!("=== Full Precision Qwen-Image Generator ===");
    println!("Prompt: {}", args.prompt);
    println!("Size: {}x{}", args.width, args.height);
    println!("Steps: {}", args.steps);
    println!();

    // Find model directory
    println!("Looking for model...");
    let model_dir = get_model_dir()?;
    println!("  Found: {}", model_dir.display());

    // Load full precision transformer
    println!("\nLoading full precision transformer weights...");
    let transformer_dir = model_dir.join("transformer");
    let start = Instant::now();
    let weights = load_safetensors_weights(&transformer_dir)?;
    println!("  Loaded {} weights in {:.2}s", weights.len(), start.elapsed().as_secs_f32());

    println!("\nInitializing full precision transformer...");
    let config = QwenFullConfig::default();
    let start = Instant::now();
    let mut transformer = QwenFullTransformer::new(config)?;
    println!("  Created in {:.2}s", start.elapsed().as_secs_f32());

    let start = Instant::now();
    load_full_precision_weights(&mut transformer, weights)?;
    println!("  Weights loaded in {:.2}s", start.elapsed().as_secs_f32());

    // Load text encoder and tokenizer from MLX model
    println!("\n=== Loading Text Encoder ===");
    let tokenizer = load_tokenizer(&model_dir)?;
    println!("  Tokenizer loaded");

    // Encode prompt
    let template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n";
    // Support longer prompts - Qwen text encoder can handle up to 512 tokens
    let max_input_len = 512;
    let drop_idx = 34;

    println!("Tokenizing prompt: \"{}\"", args.prompt);
    let formatted_prompt = template.replace("{}", &args.prompt);
    let encoding = tokenizer.encode(formatted_prompt.as_str(), false)
        .map_err(|e| format!("Tokenization error: {}", e))?;
    let token_ids: Vec<i32> = encoding.get_ids().iter().map(|&id| id as i32).collect();
    let cond_total_tokens = token_ids.len();

    let mut padded_ids = token_ids.clone();
    if padded_ids.len() > max_input_len {
        padded_ids.truncate(max_input_len);
    }
    while padded_ids.len() < max_input_len {
        padded_ids.push(0);
    }

    // Unconditional (empty prompt)
    let formatted_uncond = template.replace("{}", " ");
    let uncond_encoding = tokenizer.encode(formatted_uncond.as_str(), false)
        .map_err(|e| format!("Tokenization error: {}", e))?;
    let uncond_total_tokens = uncond_encoding.get_ids().len();
    let mut uncond_ids: Vec<i32> = uncond_encoding.get_ids().iter().map(|&id| id as i32).collect();
    if uncond_ids.len() > max_input_len {
        uncond_ids.truncate(max_input_len);
    }
    while uncond_ids.len() < max_input_len {
        uncond_ids.push(0);
    }

    println!("  Cond: {} tokens, Uncond: {} tokens", cond_total_tokens, uncond_total_tokens);

    let cond_input_ids = Array::from_slice(&padded_ids, &[1, max_input_len as i32]);
    let uncond_input_ids = Array::from_slice(&uncond_ids, &[1, max_input_len as i32]);

    let cond_attn_mask: Vec<i32> = (0..max_input_len)
        .map(|i| if i < cond_total_tokens { 1 } else { 0 })
        .collect();
    let cond_attn_mask = Array::from_slice(&cond_attn_mask, &[1, max_input_len as i32]);

    let uncond_attn_mask: Vec<i32> = (0..max_input_len)
        .map(|i| if i < uncond_total_tokens { 1 } else { 0 })
        .collect();
    let uncond_attn_mask = Array::from_slice(&uncond_attn_mask, &[1, max_input_len as i32]);

    println!("\nLoading text encoder...");
    let mut text_encoder = qwen_image_mlx::load_text_encoder(&model_dir)?;
    println!("  Text encoder loaded!");

    println!("\nEncoding prompts...");
    let start = Instant::now();
    let cond_states_full = text_encoder.forward_with_mask(&cond_input_ids, &cond_attn_mask)?;
    let uncond_states_full = text_encoder.forward_with_mask(&uncond_input_ids, &uncond_attn_mask)?;
    mlx_rs::transforms::eval([&cond_states_full, &uncond_states_full])?;
    println!("  Text encoding completed in {:.2?}", start.elapsed());

    // Drop template tokens
    let cond_valid_end = cond_total_tokens.min(max_input_len);
    let cond_valid_len = cond_valid_end.saturating_sub(drop_idx);
    let cond_states = cond_states_full.index((.., drop_idx as i32..cond_valid_end as i32, ..));

    let uncond_valid_end = uncond_total_tokens.min(max_input_len);
    let uncond_valid_len = uncond_valid_end.saturating_sub(drop_idx);
    let uncond_states = uncond_states_full.index((.., drop_idx as i32..uncond_valid_end as i32, ..));

    // Keep as f32 for full precision
    mlx_rs::transforms::eval([&cond_states, &uncond_states])?;
    println!("  Embeddings: cond={:?}, uncond={:?}", cond_states.shape(), uncond_states.shape());

    // Free text encoder to save ~2-3GB RAM (from flux2.c pattern)
    drop(text_encoder);
    unsafe { mlx_sys::mlx_clear_cache(); }
    println!("  Text encoder released to free memory");

    // Image parameters
    let height = args.height;
    let width = args.width;
    let num_steps = args.steps;
    let cfg_scale = args.guidance;
    let latent_h = height / 16;
    let latent_w = width / 16;
    let num_patches = latent_h * latent_w;

    println!("\n=== Generating Image ===");
    println!("Image size: {}x{}", width, height);
    println!("Latent size: {}x{}", latent_w, latent_h);
    println!("Num patches: {}", num_patches);

    // Generate RoPE embeddings
    let theta = 10000.0f32;
    let axes_dim = [16i32, 56i32, 56i32];

    fn compute_freqs(dim: i32, theta: f32) -> Vec<f32> {
        (0..dim/2).map(|i| {
            let scale = (i as f32 * 2.0) / dim as f32;
            1.0 / theta.powf(scale)
        }).collect()
    }

    let frame_freqs = compute_freqs(axes_dim[0], theta);
    let height_freqs = compute_freqs(axes_dim[1], theta);
    let width_freqs = compute_freqs(axes_dim[2], theta);

    let half_height = (latent_h / 2) as usize;
    let half_width = (latent_w / 2) as usize;

    let mut img_cos_data: Vec<f32> = Vec::with_capacity((num_patches * 64) as usize);
    let mut img_sin_data: Vec<f32> = Vec::with_capacity((num_patches * 64) as usize);

    for h in 0..latent_h as usize {
        for w in 0..latent_w as usize {
            // Frame: index 0
            for &freq in &frame_freqs {
                img_cos_data.push(1.0);  // cos(0) = 1
                img_sin_data.push(0.0);  // sin(0) = 0
            }

            // Height: centered
            let h_pos = if h < half_height {
                -(((latent_h as usize - half_height) - h) as i32) as f32
            } else {
                (h - half_height) as f32
            };
            for &freq in &height_freqs {
                img_cos_data.push((h_pos * freq).cos());
                img_sin_data.push((h_pos * freq).sin());
            }

            // Width: centered
            let w_pos = if w < half_width {
                -(((latent_w as usize - half_width) - w) as i32) as f32
            } else {
                (w - half_width) as f32
            };
            for &freq in &width_freqs {
                img_cos_data.push((w_pos * freq).cos());
                img_sin_data.push((w_pos * freq).sin());
            }
        }
    }

    let img_cos = Array::from_slice(&img_cos_data, &[num_patches, 64]);
    let img_sin = Array::from_slice(&img_sin_data, &[num_patches, 64]);

    // Text RoPE
    let max_vid_index = half_height.max(half_width) as i32;
    let max_txt_len = cond_valid_len.max(uncond_valid_len);

    let mut txt_cos_data: Vec<f32> = Vec::with_capacity(max_txt_len * 64);
    let mut txt_sin_data: Vec<f32> = Vec::with_capacity(max_txt_len * 64);

    for i in 0..max_txt_len {
        let pos = (max_vid_index as usize) + i;
        for &freq in &frame_freqs {
            txt_cos_data.push((pos as f32 * freq).cos());
            txt_sin_data.push((pos as f32 * freq).sin());
        }
        for &freq in &height_freqs {
            txt_cos_data.push((pos as f32 * freq).cos());
            txt_sin_data.push((pos as f32 * freq).sin());
        }
        for &freq in &width_freqs {
            txt_cos_data.push((pos as f32 * freq).cos());
            txt_sin_data.push((pos as f32 * freq).sin());
        }
    }

    let txt_cos_full = Array::from_slice(&txt_cos_data, &[max_txt_len as i32, 64]);
    let txt_sin_full = Array::from_slice(&txt_sin_data, &[max_txt_len as i32, 64]);

    let cond_txt_cos = txt_cos_full.index((..cond_valid_len as i32, ..));
    let cond_txt_sin = txt_sin_full.index((..cond_valid_len as i32, ..));
    let uncond_txt_cos = txt_cos_full.index((..uncond_valid_len as i32, ..));
    let uncond_txt_sin = txt_sin_full.index((..uncond_valid_len as i32, ..));

    println!("RoPE embeddings generated");
    println!("  Image RoPE: {:?}", img_cos.shape());
    println!("  Text RoPE: {:?}", cond_txt_cos.shape());

    // Create random latents
    let seed = args.seed.unwrap_or(42);
    println!("Seed: {}", seed);
    let key = mlx_rs::random::key(seed)?;
    let mut latents = mlx_rs::random::normal::<f32>(&[1, num_patches, 64], None, None, Some(&key))?;

    // Scheduler (FlowMatchEulerDiscreteScheduler)
    let base_shift = 0.5f32;
    let max_shift = 0.9f32;
    let base_image_seq_len = 256.0f32;
    let max_image_seq_len = 8192.0f32;
    let shift_terminal = 0.02f32;
    let image_seq_len = num_patches as f32;

    let m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len);
    let b = base_shift - m * base_image_seq_len;
    let mu = m * image_seq_len + b;
    let shift_sigma = 1.0f32;
    println!("Flow matching mu: {:.6}", mu);

    let exp_mu = mu.exp();
    let input_sigmas: Vec<f32> = (0..num_steps).map(|i| {
        1.0 - (i as f32 / (num_steps - 1) as f32) * (1.0 - 1.0 / num_steps as f32)
    }).collect();

    let shifted_sigmas: Vec<f32> = input_sigmas.iter().map(|&t| {
        if t >= 1.0 { 1.0 }
        else if t <= 0.0 { 0.0 }
        else { exp_mu / (exp_mu + (1.0 / t - 1.0).powf(shift_sigma)) }
    }).collect();

    let last_sigma = shifted_sigmas[shifted_sigmas.len() - 1];
    let scale_factor = (1.0 - last_sigma) / (1.0 - shift_terminal);
    let sigmas: Vec<f32> = shifted_sigmas.iter().map(|&t| {
        1.0 - (1.0 - t) / scale_factor
    }).collect();

    let mut sigmas_with_terminal = sigmas.clone();
    sigmas_with_terminal.push(0.0);

    println!("Sigma schedule: [{:.4}, {:.4}, ..., {:.4}, 0.0]", sigmas[0], sigmas[1], sigmas[sigmas.len()-1]);

    println!("\nRunning diffusion loop...");
    let start = Instant::now();

    for step in 0..num_steps {
        let sigma = sigmas[step as usize];
        let sigma_next = sigmas_with_terminal[(step + 1) as usize];
        let timestep = mlx_rs::Array::from_slice(&[sigma], &[1]);

        // Conditional velocity
        let cond_velocity = transformer.forward(
            &latents,
            &cond_states,
            &timestep,
            Some((&img_cos, &img_sin)),
            Some((&cond_txt_cos, &cond_txt_sin)),
        )?;

        // Unconditional velocity
        let uncond_velocity = transformer.forward(
            &latents,
            &uncond_states,
            &timestep,
            Some((&img_cos, &img_sin)),
            Some((&uncond_txt_cos, &uncond_txt_sin)),
        )?;

        // CFG with rescaling
        let velocity_diff = mlx_rs::ops::subtract(&cond_velocity, &uncond_velocity)?;
        let cfg_arr = Array::from_f32(cfg_scale);
        let scaled_diff = mlx_rs::ops::multiply(&velocity_diff, &cfg_arr)?;
        let combined = mlx_rs::ops::add(&uncond_velocity, &scaled_diff)?;

        // Rescale to match original velocity magnitude
        let eps = Array::from_f32(1e-12);
        let cond_sq = mlx_rs::ops::multiply(&cond_velocity, &cond_velocity)?;
        let cond_sum_sq = mlx_rs::ops::sum_axis(&cond_sq, -1, true)?;
        let cond_norm = mlx_rs::ops::sqrt(&mlx_rs::ops::add(&cond_sum_sq, &eps)?)?;

        let combined_sq = mlx_rs::ops::multiply(&combined, &combined)?;
        let combined_sum_sq = mlx_rs::ops::sum_axis(&combined_sq, -1, true)?;
        let combined_norm = mlx_rs::ops::sqrt(&mlx_rs::ops::add(&combined_sum_sq, &eps)?)?;

        let scale_factor = mlx_rs::ops::divide(&cond_norm, &combined_norm)?;
        let velocity = mlx_rs::ops::multiply(&combined, &scale_factor)?;

        // Euler step
        let dt = mlx_rs::Array::from_f32(sigma_next - sigma);
        let delta = mlx_rs::ops::multiply(&velocity, &dt)?;
        latents = mlx_rs::ops::add(&latents, &delta)?;

        // Only eval at reporting steps (every 5 steps) to avoid unnecessary sync
        if (step + 1) % 5 == 0 || step == 0 {
            mlx_rs::transforms::eval([&latents])?;
            println!("  Step {}/{} (sigma: {:.3})", step + 1, num_steps, sigma);
        }
    }

    // Final eval already done in loop
    let gen_elapsed = start.elapsed();
    println!("Diffusion completed in {:.2?}", gen_elapsed);
    println!("  {:.2?} per step", gen_elapsed / num_steps as u32);

    // Unpatchify
    println!("\nUnpatchifying latents...");
    let patch_size = 2i32;
    let out_channels = 16i32;
    let vae_h = latent_h * patch_size;
    let vae_w = latent_w * patch_size;

    let latents_reshaped = latents.reshape(&[1, latent_h, latent_w, out_channels, patch_size, patch_size])?;
    let latents_permuted = latents_reshaped.transpose_axes(&[0, 3, 1, 4, 2, 5])?;
    let vae_latents = latents_permuted.reshape(&[1, out_channels, vae_h, vae_w])?;
    mlx_rs::transforms::eval([&vae_latents])?;
    println!("  VAE latent shape: {:?}", vae_latents.shape());
    println!("  VAE latent range: [{:.3}, {:.3}]",
        vae_latents.min(None)?.item::<f32>(),
        vae_latents.max(None)?.item::<f32>());

    // Load VAE
    println!("\nLoading VAE decoder...");
    let mut vae = qwen_image_mlx::load_vae_from_dir(&model_dir)?;

    // Decode
    println!("\nDecoding latents...");
    let decode_start = Instant::now();
    let denorm_latents = qwen_image_mlx::QwenVAE::denormalize_latent(&vae_latents)?;
    let decoded = vae.decode(&denorm_latents)?;
    mlx_rs::transforms::eval([&decoded])?;
    println!("  Decoded in {:.2?}", decode_start.elapsed());
    println!("  Output shape: {:?}", decoded.shape());

    // Save image
    println!("\nSaving image...");
    let img = decoded.index((0, .., .., ..));
    let img = mlx_rs::ops::clip(&img, (-1.0f32, 1.0f32))?;
    let img = mlx_rs::ops::add(&img, &Array::from_f32(1.0))?;
    let img = mlx_rs::ops::multiply(&img, &Array::from_f32(127.5))?;
    let img = img.as_dtype(mlx_rs::Dtype::Uint8)?;

    let img = img.transpose_axes(&[1, 2, 0])?;
    mlx_rs::transforms::eval([&img])?;

    let img_shape = img.shape();
    let img_h = img_shape[0] as u32;
    let img_w = img_shape[1] as u32;

    let numel = img_h as i32 * img_w as i32 * 3;
    let img = img.reshape(&[numel])?;
    let img = img.reshape(&[img_h as i32, img_w as i32, 3])?;
    mlx_rs::transforms::eval([&img])?;

    let img_data: Vec<u8> = img.as_slice().to_vec();

    // Save as PNG (or PPM as fallback)
    let output_path = &args.output;
    let ext = output_path.extension().and_then(|s| s.to_str()).unwrap_or("png");

    if ext == "ppm" {
        // PPM format (simple, no dependencies)
        let mut file = std::fs::File::create(output_path)?;
        writeln!(file, "P6")?;
        writeln!(file, "{} {}", img_w, img_h)?;
        writeln!(file, "255")?;
        file.write_all(&img_data)?;
    } else {
        // PNG format using image crate
        let img_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_raw(img_w, img_h, img_data)
                .expect("Failed to create image buffer");
        img_buffer.save(output_path)?;
    }

    println!("Saved image to: {}", output_path.display());
    println!("  Image size: {}x{}", img_w, img_h);

    println!("\n=== Generation Complete ===");

    Ok(())
}
