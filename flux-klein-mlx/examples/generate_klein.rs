//! Full image generation with FLUX.2-klein-4B
//!
//! FLUX.2-klein-4B uses:
//! - Qwen3-4B text encoder (36 layers, 2560 hidden dim)
//! - 5 double + 20 single transformer blocks
//! - 4 denoising steps
//! - ~13GB VRAM
//!
//! Run with: cargo run --example generate_klein --release -- "a cat"
//!
//! Note: This requires downloading the model weights from HuggingFace:
//!   black-forest-labs/FLUX.2-klein-4B

use flux_klein_mlx::autoencoder::{AutoEncoderConfig, Decoder};
use flux_klein_mlx::klein_model::{FluxKlein, FluxKleinParams};
use flux_klein_mlx::klein_quantized::QuantizedFluxKlein;
use flux_klein_mlx::qwen3_encoder::{Qwen3Config, Qwen3TextEncoder, sanitize_qwen3_weights};
use flux_klein_mlx::{load_safetensors, sanitize_vae_weights};
use flux_klein_mlx::weights::sanitize_klein_model_weights;
use hf_hub::api::sync::ApiBuilder;
use mlx_rs::module::ModuleParameters;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;
use std::collections::HashMap;
use tokenizers::Tokenizer;

/// Enum to hold either f32 or quantized FLUX model
enum FluxModel {
    F32(FluxKlein),
    Quantized(QuantizedFluxKlein),
}

impl FluxModel {
    fn forward_with_rope(
        &mut self,
        img: &Array,
        txt: &Array,
        timesteps: &Array,
        rope_cos: &Array,
        rope_sin: &Array,
    ) -> Result<Array, mlx_rs::error::Exception> {
        match self {
            FluxModel::F32(m) => m.forward_with_rope(img, txt, timesteps, rope_cos, rope_sin),
            FluxModel::Quantized(m) => m.forward_with_rope(img, txt, timesteps, rope_cos, rope_sin),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    // Check for --quantize flag
    let use_quantize = args.iter().any(|a| a == "--quantize" || a == "-q");

    // Check for --steps N flag
    let num_steps: i32 = args.iter()
        .position(|a| a == "--steps")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);  // Default 4 steps for FLUX.2-klein

    // Get prompt (filter out flags and their values)
    let mut skip_next = false;
    let prompt_parts: Vec<&str> = args[1..]
        .iter()
        .filter(|a| {
            if skip_next {
                skip_next = false;
                return false;
            }
            if *a == "--steps" || *a == "--output" || *a == "--prompt" {
                skip_next = true;
                return false;
            }
            !a.starts_with('-')
        })
        .map(|s| s.as_str())
        .collect();
    let prompt = if prompt_parts.is_empty() {
        "a beautiful sunset over the ocean".to_string()
    } else {
        prompt_parts.join(" ")
    };

    println!("=== FLUX.2-klein-4B Image Generation ===");
    if use_quantize {
        println!("Mode: INT8 Quantized");
    }
    println!("Steps: {}", num_steps);
    println!("Prompt: \"{}\"\n", prompt);

    // =========================================================================
    // Step 1: Download models from HuggingFace
    // =========================================================================
    println!("Step 1: Downloading models from HuggingFace...");

    let token = std::env::var("HF_TOKEN").ok().or_else(|| {
        let home = std::env::var("HOME").ok()?;
        let token_path = std::path::PathBuf::from(home).join(".cache/huggingface/token");
        std::fs::read_to_string(token_path)
            .ok()
            .map(|s| s.trim().to_string())
    });

    let api = if let Some(ref token) = token {
        ApiBuilder::new().with_token(Some(token.clone())).build()?
    } else {
        ApiBuilder::new().build()?
    };

    // FLUX.2-klein-4B (transformer + text encoder bundled)
    let flux_repo = api.model("black-forest-labs/FLUX.2-klein-4B".to_string());

    // Try to get the model index to see what files are available
    println!("  Checking available files...");

    // The model likely has these files in diffusers format:
    // - transformer/diffusion_pytorch_model.safetensors
    // - text_encoder/model.safetensors (Qwen3)
    // - vae/diffusion_pytorch_model.safetensors
    // - tokenizer/tokenizer.json

    // For now, let's try to get the files
    let transformer_path = match flux_repo.get("transformer/diffusion_pytorch_model.safetensors") {
        Ok(path) => {
            println!("  Transformer: downloaded");
            Some(path)
        }
        Err(e) => {
            println!("  Note: Could not download transformer: {}", e);
            println!("  Trying alternative paths...");
            // Try flux.safetensors
            flux_repo.get("flux.safetensors").ok()
        }
    };

    // Text encoder is split into two files
    let text_encoder_path1 = match flux_repo.get("text_encoder/model-00001-of-00002.safetensors") {
        Ok(path) => {
            println!("  Text encoder part 1: downloaded");
            Some(path)
        }
        Err(e) => {
            println!("  Note: Could not download text encoder part 1: {}", e);
            None
        }
    };
    let text_encoder_path2 = match flux_repo.get("text_encoder/model-00002-of-00002.safetensors") {
        Ok(path) => {
            println!("  Text encoder part 2: downloaded");
            Some(path)
        }
        Err(e) => {
            println!("  Note: Could not download text encoder part 2: {}", e);
            None
        }
    };

    let vae_path = match flux_repo.get("vae/diffusion_pytorch_model.safetensors") {
        Ok(path) => {
            println!("  VAE: downloaded");
            Some(path)
        }
        Err(e) => {
            println!("  Note: Could not download VAE: {}", e);
            // Try ae.safetensors
            flux_repo.get("ae.safetensors").ok()
        }
    };

    let tokenizer_path = match flux_repo.get("tokenizer/tokenizer.json") {
        Ok(path) => {
            println!("  Tokenizer: downloaded");
            Some(path)
        }
        Err(e) => {
            println!("  Note: Could not download tokenizer: {}", e);
            None
        }
    };

    // Check if we have all required files
    if transformer_path.is_none() {
        println!("\nError: Could not find transformer weights.");
        println!("The model format may be different than expected.");
        println!("\nTrying to list available files...");

        // List what files are available
        if let Ok(files) = flux_repo.info() {
            println!("Model info: {:?}", files.sha);
        }

        return Err("Missing transformer weights".into());
    }

    // =========================================================================
    // Step 2: Load Qwen3 text encoder
    // =========================================================================
    println!("\nStep 2: Loading Qwen3 text encoder...");

    // Qwen3-4B configuration (from FLUX.2-klein-4B text_encoder/config.json)
    let qwen3_config = Qwen3Config {
        hidden_size: 2560,
        num_hidden_layers: 36,
        intermediate_size: 9728,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        rms_norm_eps: 1e-6,
        vocab_size: 151936,
        max_position_embeddings: 40960,
        rope_theta: 1000000.0,
        head_dim: 128,  // 128, not 80
    };

    let mut qwen3 = Qwen3TextEncoder::new(qwen3_config.clone())?;
    println!("  Model created: {} layers", qwen3_config.num_hidden_layers);

    if text_encoder_path1.is_some() && text_encoder_path2.is_some() {
        let start = std::time::Instant::now();

        // Load both parts and combine
        let mut all_weights = HashMap::new();

        if let Some(ref path) = text_encoder_path1 {
            let weights = load_safetensors(path)?;
            all_weights.extend(weights);
        }
        if let Some(ref path) = text_encoder_path2 {
            let weights = load_safetensors(path)?;
            all_weights.extend(weights);
        }

        let weights = sanitize_qwen3_weights(all_weights);

        // Convert bf16 weights to f32 (Qwen3 weights are stored in bf16)
        let weights: HashMap<String, Array> = weights
            .into_iter()
            .map(|(k, v)| {
                let v32 = v.as_type::<f32>().unwrap_or(v);
                (k, v32)
            })
            .collect();

        println!("  Loaded {} weights in {:?}", weights.len(), start.elapsed());

        let weights_rc: HashMap<std::rc::Rc<str>, Array> = weights
            .into_iter()
            .map(|(k, v)| (std::rc::Rc::from(k.as_str()), v))
            .collect();
        qwen3.update_flattened(weights_rc);
        println!("  Text encoder ready");
    } else {
        println!("  Warning: Using random weights (text encoder files not found)");
    }

    // Load tokenizer
    let tokenizer = if let Some(ref path) = tokenizer_path {
        Some(Tokenizer::from_file(path).map_err(|e| format!("Tokenizer error: {}", e))?)
    } else {
        println!("  Warning: No tokenizer found, using dummy tokens");
        None
    };

    // Apply Qwen3 chat template to prompt
    // Format: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
    let chat_prompt = format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        prompt
    );
    println!("  Chat template applied");

    // =========================================================================
    // Step 3: Load FLUX transformer
    // =========================================================================
    println!("\nStep 3: Loading FLUX.2-klein transformer...");

    // FLUX.2-klein-4B configuration (new architecture)
    let params = FluxKleinParams::default();
    let mut flux = FluxKlein::new(params.clone())?;
    println!("  Model created: {} double + {} single blocks", params.depth, params.depth_single);

    if let Some(ref path) = transformer_path {
        let start = std::time::Instant::now();
        let raw_weights = load_safetensors(path)?;
        let weights = sanitize_klein_model_weights(raw_weights);
        println!("  Loaded {} weights in {:?}", weights.len(), start.elapsed());

        // Convert bf16 weights to f32
        let weights: HashMap<String, Array> = weights
            .into_iter()
            .map(|(k, v)| {
                let v32 = v.as_type::<f32>().unwrap_or(v);
                (k, v32)
            })
            .collect();

        let weights_rc: HashMap<std::rc::Rc<str>, Array> = weights
            .into_iter()
            .map(|(k, v)| (std::rc::Rc::from(k.as_str()), v))
            .collect();

        flux.update_flattened(weights_rc);
        println!("  Transformer ready");
    } else {
        println!("  Warning: Using random weights (no transformer found)");
    }

    // Optionally quantize the model to INT8
    let mut flux: FluxModel = if use_quantize {
        println!("\nStep 3b: Quantizing transformer to INT8...");
        let start = std::time::Instant::now();
        let quantized = QuantizedFluxKlein::from_unquantized(flux, 64, 8)?;
        println!("  Quantization complete in {:?}", start.elapsed());
        FluxModel::Quantized(quantized)
    } else {
        FluxModel::F32(flux)
    };

    // =========================================================================
    // Step 4: Load VAE decoder
    // =========================================================================
    println!("\nStep 4: Loading VAE decoder...");

    // FLUX.2 uses 32 latent channels, not 16
    let vae_config = AutoEncoderConfig::flux2();
    println!("  VAE z_channels: {}", vae_config.z_channels);
    let mut vae = Decoder::new(vae_config.clone())?;

    if let Some(ref path) = vae_path {
        let start = std::time::Instant::now();
        let weights = load_safetensors(path)?;
        let weights = sanitize_vae_weights(weights);
        println!("  Loaded {} weights in {:?}", weights.len(), start.elapsed());

        let weights_rc: HashMap<std::rc::Rc<str>, Array> = weights
            .into_iter()
            .map(|(k, v)| (std::rc::Rc::from(k.as_str()), v))
            .collect();
        vae.update_flattened(weights_rc);
        println!("  VAE decoder ready");
    } else {
        println!("  Warning: Using random weights (no VAE found)");
    }

    // =========================================================================
    // Step 5: Encode text prompt
    // =========================================================================
    println!("\nStep 5: Encoding text prompt...");

    let batch_size = 1i32;
    let max_seq_len = 512i32;

    // Tokenize using chat template
    let (input_ids, attention_mask) = if let Some(ref tok) = tokenizer {
        let encoding = tok.encode(chat_prompt.as_str(), true).map_err(|e| format!("Encode error: {}", e))?;
        let ids: Vec<i32> = encoding.get_ids().iter().map(|&x| x as i32).collect();
        let num_tokens = ids.len().min(max_seq_len as usize);

        // Pad tokens with 151643 (<|endoftext|>) - Qwen3's pad token
        let mut padded = vec![151643i32; max_seq_len as usize];
        padded[..num_tokens].copy_from_slice(&ids[..num_tokens]);

        // Create attention mask: 1 for real tokens, 0 for padding
        let mut mask = vec![0i32; max_seq_len as usize];
        for i in 0..num_tokens {
            mask[i] = 1;
        }

        let ids_arr = Array::from_slice(&padded, &[batch_size, max_seq_len]);
        let mask_arr = Array::from_slice(&mask, &[batch_size, max_seq_len]);
        (ids_arr, Some(mask_arr))
    } else {
        // Dummy tokens with full attention
        let ids_arr = Array::from_slice(&vec![1i32; max_seq_len as usize], &[batch_size, max_seq_len]);
        let mask_arr = Array::from_slice(&vec![1i32; max_seq_len as usize], &[batch_size, max_seq_len]);
        (ids_arr, Some(mask_arr))
    };

    let start = std::time::Instant::now();
    let txt_embed = qwen3.encode(&input_ids, attention_mask.as_ref())?;
    let txt_embed = txt_embed.as_dtype(mlx_rs::Dtype::Float32)?;
    txt_embed.eval()?;
    println!("  Text embeddings: {:?} (took {:?})", txt_embed.shape(), start.elapsed());

    // =========================================================================
    // Step 6: Setup generation parameters
    // =========================================================================
    println!("\nStep 6: Setting up generation...");

    let img_height = 512i32;
    let img_width = 512i32;
    let latent_height = img_height / 8;  // 64 - VAE latent dimension
    let latent_width = img_width / 8;    // 64

    // FLUX uses 2x2 patchify on the latent, so the transformer sequence length is (h/2)*(w/2)
    let patch_size = 2i32;
    let patch_h = latent_height / patch_size;  // 32
    let patch_w = latent_width / patch_size;   // 32
    let img_seq_len = patch_h * patch_w;       // 1024 patches (not 4096!)
    let in_channels = params.in_channels;      // 128 = 32 VAE channels * 2*2 patch

    println!("  Image size: {}x{}", img_width, img_height);
    println!("  VAE latent size: {}x{}", latent_width, latent_height);
    println!("  Patch grid: {}x{} = {} patches", patch_w, patch_h, img_seq_len);
    println!("  Patch channels: {}", in_channels);

    // =========================================================================
    // Step 7: Denoising loop
    // =========================================================================
    println!("\nStep 7: Running denoising ({} steps)...", num_steps);

    // Use flux.c's non-linear schedule with SNR shift
    let timesteps = flux_official_schedule(img_seq_len, num_steps);
    println!("  Using SNR-shifted schedule: {:?}", timesteps);

    // Start with random noise (sigma=1 for rectified flow)
    let mut latent = mlx_rs::random::normal::<f32>(
        &[batch_size, img_seq_len, in_channels],
        None,
        None,
        None,
    )?;

    // Create position IDs and compute RoPE ONCE before the loop
    let txt_ids = create_txt_ids(batch_size, max_seq_len)?;
    let img_ids = create_img_ids(batch_size, patch_h, patch_w)?;
    let (rope_cos, rope_sin) = FluxKlein::compute_rope(&txt_ids, &img_ids)?;

    // Iterate through timestep pairs (t_curr, t_next)
    for step in 0..num_steps as usize {
        let t_curr = timesteps[step];
        let t_next = timesteps[step + 1];
        let t_arr = Array::from_slice(&[t_curr * 1000.0], &[batch_size]);

        let start = std::time::Instant::now();

        let v_pred = flux.forward_with_rope(&latent, &txt_embed, &t_arr, &rope_cos, &rope_sin)?;

        // Euler step: z_next = z_curr + (t_next - t_curr) * v
        let dt = t_next - t_curr;
        let scaled_v = mlx_rs::ops::multiply(&v_pred, &Array::from_slice(&[dt], &[1]))?;
        latent = mlx_rs::ops::add(&latent, &scaled_v)?;
        latent.eval()?;  // Single eval per step to get timing

        println!("  Step {}/{}: t={:.3}->{:.3}, took {:?}", step + 1, num_steps, t_curr, t_next, start.elapsed());
    }

    // =========================================================================
    // Step 8: Decode latents to image
    // =========================================================================
    println!("\nStep 8: Decoding latents to image...");

    // FLUX.2 transformer outputs [batch, patch_h*patch_w, 128] where 128 = 2x2 patch × 32 channels
    // We need to unpack this to [batch, latent_h, latent_w, 32] for the VAE
    // latent_h = patch_h * 2, latent_w = patch_w * 2
    // First reshape from [batch, seq, 128] to [batch, patch_h, patch_w, 2, 2, 32]
    // Then permute to [batch, patch_h, 2, patch_w, 2, 32] and flatten to [batch, latent_h, latent_w, 32]
    let z_channels = vae_config.z_channels;  // 32

    // Unpack latent: [batch, seq, 128] -> [batch, H, W, 32] for VAE
    let latent = latent.reshape(&[batch_size, patch_h, patch_w, z_channels, patch_size, patch_size])?;
    let latent = latent.transpose_axes(&[0, 1, 4, 2, 5, 3])?;
    let vae_height = patch_h * patch_size;
    let vae_width = patch_w * patch_size;
    let latent_for_vae = latent.reshape(&[batch_size, vae_height, vae_width, z_channels])?;

    let start = std::time::Instant::now();
    let image = vae.forward(&latent_for_vae)?;
    image.eval()?;
    println!("  Decoded in {:?}", start.elapsed());

    // =========================================================================
    // Step 9: Save image
    // =========================================================================
    println!("\nStep 9: Saving image...");

    // VAE output is in [-1, 1] range, convert to [0, 255]
    let image = mlx_rs::ops::add(&image, &Array::from_slice(&[1.0f32], &[1]))?;
    let image = mlx_rs::ops::multiply(&image, &Array::from_slice(&[127.5f32], &[1]))?;
    let image = mlx_rs::ops::maximum(&image, &Array::from_slice(&[0.0f32], &[1]))?;
    let image = mlx_rs::ops::minimum(&image, &Array::from_slice(&[255.0f32], &[1]))?;
    image.eval()?;

    let shape = image.shape();
    let height = shape[1] as usize;
    let width = shape[2] as usize;

    // Flatten and convert to u8
    let image_flat = image.reshape(&[-1])?;
    let image_data: Vec<f32> = image_flat.as_slice().to_vec();
    let rgb_bytes: Vec<u8> = image_data.iter().map(|&v| v.round() as u8).collect();

    let output_path = "output_klein.ppm";
    let ppm_data = format!("P6\n{} {}\n255\n", width, height);
    std::fs::write(
        output_path,
        ppm_data
            .as_bytes()
            .iter()
            .chain(rgb_bytes.iter())
            .copied()
            .collect::<Vec<u8>>(),
    )?;
    println!("  Saved to: {}", output_path);

    println!("\n=== Generation Complete ===");

    Ok(())
}

/// Create image position IDs for 4D RoPE (FLUX.2-klein uses axes_dims_rope=[32,32,32,32])
///
/// Position IDs format: [batch, seq, 4] where:
/// - dim 0: T position (always 0 for images)
/// - dim 1: H1 position (y divided by some factor)
/// - dim 2: H2 position (y mod some factor)
/// - dim 3: W position (x coordinate)
fn create_img_ids(batch: i32, h: i32, w: i32) -> Result<Array, mlx_rs::error::Exception> {
    let mut ids = Vec::with_capacity((batch * h * w * 4) as usize);

    for _ in 0..batch {
        for y in 0..h {
            for x in 0..w {
                ids.push(0.0f32);      // T position = 0 for images
                ids.push(y as f32);    // H position
                ids.push(x as f32);    // W position
                ids.push(0.0f32);      // Extra dim = 0
            }
        }
    }

    Ok(Array::from_slice(&ids, &[batch, h * w, 4]))
}

/// Create text position IDs for 4D RoPE (FLUX.2-klein uses axes_dims_rope=[32,32,32,32])
///
/// Position IDs format: [batch, seq, 4] where (matching flux.c):
/// - dim 0: T position = 0 (always 0 for text)
/// - dim 1: H position = 0 (always 0 for text)
/// - dim 2: W position = 0 (always 0 for text)
/// - dim 3: L position = sequence index
fn create_txt_ids(batch: i32, seq_len: i32) -> Result<Array, mlx_rs::error::Exception> {
    let mut ids = Vec::with_capacity((batch * seq_len * 4) as usize);

    for _ in 0..batch {
        for s in 0..seq_len {
            ids.push(0.0f32);      // T position = 0 for text
            ids.push(0.0f32);      // H position = 0 for text
            ids.push(0.0f32);      // W position = 0 for text
            ids.push(s as f32);    // L position = sequence index
        }
    }

    Ok(Array::from_slice(&ids, &[batch, seq_len, 4]))
}

/// Compute empirical mu for SNR shift based on image sequence length and number of steps
/// From flux.c flux_sample.c
fn compute_empirical_mu(image_seq_len: i32, num_steps: i32) -> f32 {
    const A1: f32 = 8.73809524e-05;
    const B1: f32 = 1.89833333;
    const A2: f32 = 0.00016927;
    const B2: f32 = 0.45666666;

    if image_seq_len > 4300 {
        return A2 * (image_seq_len as f32) + B2;
    }

    let m_200 = A2 * (image_seq_len as f32) + B2;
    let m_10 = A1 * (image_seq_len as f32) + B1;
    let a = (m_200 - m_10) / 190.0;
    let b = m_200 - 200.0 * a;
    a * (num_steps as f32) + b
}

/// Apply generalized time SNR shift
/// From flux.c flux_sample.c
fn generalized_time_snr_shift(t: f32, mu: f32, sigma: f32) -> f32 {
    if t <= 0.0 {
        return 0.0;
    }
    if t >= 1.0 {
        return 1.0;
    }
    mu.exp() / (mu.exp() + (1.0 / t - 1.0).powf(sigma))
}

/// Generate the official FLUX timestep schedule with SNR shift
/// This matches flux.c's flux_official_schedule function
fn flux_official_schedule(image_seq_len: i32, num_steps: i32) -> Vec<f32> {
    let mu = compute_empirical_mu(image_seq_len, num_steps);
    let sigma = 1.0f32;

    let mut timesteps = Vec::with_capacity((num_steps + 1) as usize);
    for i in 0..=num_steps {
        let t_linear = 1.0 - (i as f32) / (num_steps as f32);
        let t_shifted = generalized_time_snr_shift(t_linear, mu, sigma);
        timesteps.push(t_shifted);
    }
    timesteps
}
