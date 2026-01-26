//! Z-Image-Turbo Image Generation
//!
//! Z-Image-Turbo is a 6B parameter Single-Stream DiT optimized for Apple Silicon.
//!
//! Architecture:
//! - Qwen3-4B text encoder (same as FLUX.2-klein, different extraction)
//! - Noise Refiner + Context Refiner + Joint transformer blocks
//! - 9 denoising steps (Turbo distilled)
//! - 3-axis RoPE (32, 48, 48)
//!
//! Run with: cargo run --example generate_zimage --release -- "a cat"
//!
//! Note: This requires downloading the model weights from HuggingFace:
//!   huggingface-cli download uqer1244/MLX-z-image --local-dir ./models/zimage-turbo-mlx

use flux_klein_mlx::autoencoder::{AutoEncoderConfig, Decoder};
use flux_klein_mlx::qwen3_encoder::{Qwen3Config, Qwen3TextEncoder, sanitize_qwen3_weights};
use flux_klein_mlx::{load_safetensors, sanitize_vae_weights};
use zimage_mlx::{
    ZImageTransformer, ZImageConfig, create_coordinate_grid, sanitize_zimage_weights, sanitize_mlx_weights,
    QuantizedQwen3TextEncoder, sanitize_quantized_qwen3_weights,
};
use hf_hub::api::sync::ApiBuilder;
use mlx_rs::module::ModuleParameters;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;
use std::collections::HashMap;
use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    // Check for --quantize flag
    let use_quantize = args.iter().any(|a| a == "--quantize" || a == "-q");

    // Get prompt (filter out flags)
    let prompt_parts: Vec<&str> = args[1..]
        .iter()
        .filter(|a| !a.starts_with('-'))
        .map(|s| s.as_str())
        .collect();
    let prompt = if prompt_parts.is_empty() {
        "a beautiful sunset over the ocean".to_string()
    } else {
        prompt_parts.join(" ")
    };

    println!("=== Z-Image-Turbo Image Generation ===");
    if use_quantize {
        println!("Mode: 4-bit Quantized");
    }
    println!("Prompt: \"{}\"\n", prompt);

    // =========================================================================
    // Step 1: Download/locate models from HuggingFace
    // =========================================================================
    println!("Step 1: Locating models...");

    let api = ApiBuilder::new().build()?;

    // Z-Image-Turbo MLX quantized version
    let zimage_repo = api.model("uqer1244/MLX-z-image".to_string());

    // Check for model files
    // The MLX version has these files:
    // - transformer/config.json
    // - transformer/model.safetensors.index.json (sharded)
    // - transformer/model-00001-of-*.safetensors
    // - text_encoder/config.json
    // - text_encoder/model.safetensors (4-bit quantized)
    // - vae/diffusion_pytorch_model.safetensors
    // - tokenizer/tokenizer.json

    let transformer_config = match zimage_repo.get("transformer/config.json") {
        Ok(path) => {
            println!("  Transformer config: found");
            Some(path)
        }
        Err(e) => {
            println!("  Warning: Could not find transformer config: {}", e);
            None
        }
    };

    let text_encoder_path = match zimage_repo.get("text_encoder/model.safetensors") {
        Ok(path) => {
            println!("  Text encoder (4-bit): found");
            Some(path)
        }
        Err(e) => {
            println!("  Warning: Could not find text encoder: {}", e);
            None
        }
    };

    let vae_path = match zimage_repo.get("vae/diffusion_pytorch_model.safetensors") {
        Ok(path) => {
            println!("  VAE: found");
            Some(path)
        }
        Err(e) => {
            println!("  Warning: Could not find VAE: {}", e);
            None
        }
    };

    let tokenizer_path = match zimage_repo.get("tokenizer/tokenizer.json") {
        Ok(path) => {
            println!("  Tokenizer: found");
            Some(path)
        }
        Err(e) => {
            println!("  Warning: Could not find tokenizer: {}", e);
            None
        }
    };

    // =========================================================================
    // Step 2: Load Qwen3 text encoder
    // =========================================================================
    println!("\nStep 2: Loading Qwen3 text encoder...");

    // Qwen3-4B configuration (identical to FLUX.2-klein)
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
        head_dim: 128,
    };

    // Use enum to handle both quantized and non-quantized text encoders
    enum TextEncoder {
        Quantized(QuantizedQwen3TextEncoder),
        NonQuantized(Qwen3TextEncoder),
    }

    let mut text_encoder: Option<TextEncoder> = None;
    let mut use_dummy_text = true;

    // Try to load text encoder weights from PyTorch model (non-quantized)
    let pytorch_te_path = std::path::Path::new("models/zimage-turbo-pytorch/text_encoder");
    if pytorch_te_path.exists() {
        let shard1 = pytorch_te_path.join("model-00001-of-00003.safetensors");
        if shard1.exists() {
            println!("  Loading non-quantized weights from PyTorch model...");
            let start = std::time::Instant::now();

            let mut qwen3 = Qwen3TextEncoder::new(qwen3_config.clone())?;

            // Load all shards
            let mut all_weights = HashMap::new();
            for i in 1..=3 {
                let shard_path = pytorch_te_path.join(format!("model-0000{}-of-00003.safetensors", i));
                if shard_path.exists() {
                    let shard_weights = load_safetensors(&shard_path)?;
                    println!("    Loaded shard {}: {} weights", i, shard_weights.len());
                    all_weights.extend(shard_weights);
                }
            }

            // Sanitize weight names
            let weights = sanitize_qwen3_weights(all_weights);
            println!("  Sanitized {} weights in {:?}", weights.len(), start.elapsed());

            // Convert bf16 to f32 if needed
            let weights: HashMap<String, Array> = weights
                .into_iter()
                .map(|(k, v)| {
                    let v32 = v.as_type::<f32>().unwrap_or(v);
                    (k, v32)
                })
                .collect();

            // Load into model
            let weights_rc: HashMap<std::rc::Rc<str>, Array> = weights
                .into_iter()
                .map(|(k, v)| (std::rc::Rc::from(k.as_str()), v))
                .collect();
            qwen3.update_flattened(weights_rc);
            text_encoder = Some(TextEncoder::NonQuantized(qwen3));
            use_dummy_text = false;
            println!("  Non-quantized text encoder weights loaded");
        }
    }

    // Fall back to MLX quantized weights if PyTorch not present
    if use_dummy_text {
        if let Some(ref path) = text_encoder_path {
            println!("  Loading 4-bit quantized weights from MLX model...");
            let start = std::time::Instant::now();

            // Load quantized weights
            let te_weights = load_safetensors(path)?;
            println!("    Loaded {} weight tensors", te_weights.len());

            // Sanitize and create quantized encoder
            let weights = sanitize_quantized_qwen3_weights(te_weights);

            let mut qwen3_quantized = QuantizedQwen3TextEncoder::new(qwen3_config.clone())?;

            // Load weights
            let weights_rc: HashMap<std::rc::Rc<str>, Array> = weights
                .into_iter()
                .map(|(k, v)| (std::rc::Rc::from(k.as_str()), v))
                .collect();
            qwen3_quantized.update_flattened(weights_rc);

            text_encoder = Some(TextEncoder::Quantized(qwen3_quantized));
            use_dummy_text = false;
            println!("  Quantized text encoder loaded in {:?}", start.elapsed());
        } else {
            println!("  Warning: No text encoder found, using dummy embeddings");
        }
    }

    println!("  Model created: {} layers", qwen3_config.num_hidden_layers);

    // Load tokenizer
    let tokenizer = if let Some(ref path) = tokenizer_path {
        Some(Tokenizer::from_file(path).map_err(|e| format!("Tokenizer error: {}", e))?)
    } else {
        println!("  Warning: No tokenizer found");
        None
    };

    // Apply Qwen3 chat template (same as FLUX.2-klein)
    let chat_prompt = format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        prompt
    );
    println!("  Chat template applied");

    // =========================================================================
    // Step 3: Load Z-Image transformer
    // =========================================================================
    println!("\nStep 3: Loading Z-Image transformer...");

    // Parse config if available
    let zimage_config = if let Some(ref config_path) = transformer_config {
        let config_str = std::fs::read_to_string(config_path)?;
        println!("  Config loaded: {} bytes", config_str.len());

        let json: serde_json::Value = serde_json::from_str(&config_str)?;
        ZImageConfig {
            dim: json.get("dim").and_then(|v| v.as_i64()).unwrap_or(3840) as i32,
            n_heads: json.get("n_heads").or(json.get("nheads")).and_then(|v| v.as_i64()).unwrap_or(30) as i32,
            n_kv_heads: json.get("n_kv_heads").and_then(|v| v.as_i64()).unwrap_or(30) as i32,
            n_layers: json.get("n_layers").and_then(|v| v.as_i64()).unwrap_or(30) as i32,
            n_refiner_layers: json.get("n_refiner_layers").and_then(|v| v.as_i64()).unwrap_or(2) as i32,
            in_channels: json.get("in_channels").and_then(|v| v.as_i64()).unwrap_or(16) as i32,
            cap_feat_dim: json.get("cap_feat_dim").and_then(|v| v.as_i64()).unwrap_or(2560) as i32,
            axes_dims: [32, 48, 48],  // From config
            rope_theta: json.get("rope_theta").and_then(|v| v.as_f64()).unwrap_or(256.0) as f32,
            t_scale: json.get("t_scale").and_then(|v| v.as_f64()).unwrap_or(1000.0) as f32,
            norm_eps: json.get("norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-5) as f32,
            patch_size: 2,
        }
    } else {
        ZImageConfig::default()
    };

    println!("  dim: {}", zimage_config.dim);
    println!("  n_heads: {}", zimage_config.n_heads);
    println!("  n_layers: {} (joint blocks)", zimage_config.n_layers);
    println!("  n_refiner_layers: {}", zimage_config.n_refiner_layers);
    println!("  in_channels: {}", zimage_config.in_channels);
    println!("  cap_feat_dim: {}", zimage_config.cap_feat_dim);

    // Create transformer model
    let start = std::time::Instant::now();
    let mut transformer = ZImageTransformer::new(zimage_config.clone())?;
    println!("  Model created in {:?}", start.elapsed());

    // Try to load transformer weights - first try dequantized MLX, then PyTorch
    let dequant_mlx_path = std::path::Path::new("/tmp/zimage_dequantized.safetensors");
    let pytorch_transformer_path = std::path::Path::new("models/zimage-turbo-pytorch/transformer");

    let weights_opt = if dequant_mlx_path.exists() {
        // Use dequantized MLX weights (known to work with Python implementation)
        println!("  Loading dequantized MLX weights...");
        let start = std::time::Instant::now();
        let raw_weights = load_safetensors(dequant_mlx_path)?;
        println!("  Loaded {} raw weights", raw_weights.len());
        let weights = sanitize_mlx_weights(raw_weights);
        println!("  Sanitized to {} weights in {:?}", weights.len(), start.elapsed());
        Some(weights)
    } else if pytorch_transformer_path.exists() {
        let shard1 = pytorch_transformer_path.join("diffusion_pytorch_model-00001-of-00003.safetensors");
        if shard1.exists() {
            println!("  Loading weights from PyTorch model...");
            let start = std::time::Instant::now();

            let mut all_weights = HashMap::new();
            for i in 1..=3 {
                let shard_path = pytorch_transformer_path.join(format!("diffusion_pytorch_model-0000{}-of-00003.safetensors", i));
                if shard_path.exists() {
                    let shard_weights = load_safetensors(&shard_path)?;
                    println!("    Loaded shard {}: {} weights", i, shard_weights.len());
                    all_weights.extend(shard_weights);
                }
            }

            let weights = sanitize_zimage_weights(all_weights);
            println!("  Sanitized {} weights in {:?}", weights.len(), start.elapsed());
            Some(weights)
        } else {
            None
        }
    } else {
        None
    };

    if let Some(weights) = weights_opt {
        // Convert bf16 to f32 if needed
        let weights: HashMap<String, Array> = weights
            .into_iter()
            .map(|(k, v)| {
                let v32 = v.as_type::<f32>().unwrap_or(v);
                (k, v32)
            })
            .collect();

        // Check weights before loading
        let te_before = transformer.t_embedder.linear1.weight.sum(None)?.item::<f32>();
        let layer_before = transformer.layers[0].attention.to_q.weight.sum(None)?.item::<f32>();
        println!("  Before: t_embedder.linear1={:.2}, layers.0.attention.to_q={:.2}", te_before, layer_before);

        // Load into model
        let weights_rc: HashMap<std::rc::Rc<str>, Array> = weights
            .into_iter()
            .map(|(k, v)| (std::rc::Rc::from(k.as_str()), v))
            .collect();
        transformer.update_flattened(weights_rc);

        // Verify weights loaded
        let te_after = transformer.t_embedder.linear1.weight.sum(None)?.item::<f32>();
        let layer_after = transformer.layers[0].attention.to_q.weight.sum(None)?.item::<f32>();
        println!("  After: t_embedder.linear1={:.2}, layers.0.attention.to_q={:.2}", te_after, layer_after);
        println!("  Weights changed: t_embedder={}, layers.0.to_q={}", te_before != te_after, layer_before != layer_after);
    } else {
        println!("  Note: Using random weights (no transformer weights found)");
    }

    // =========================================================================
    // Step 4: Load VAE decoder
    // =========================================================================
    println!("\nStep 4: Loading VAE decoder...");

    // Z-Image VAE has latent_channels=16 (different from FLUX's 32)
    let vae_config = AutoEncoderConfig {
        resolution: 1024,
        in_channels: 3,
        ch: 128,
        out_ch: 3,
        ch_mult: vec![1, 2, 4, 4],  // [128, 256, 512, 512]
        num_res_blocks: 2,
        z_channels: 16,  // Z-Image uses 16 latent channels
        scale_factor: 0.3611,
        shift_factor: 0.1159,
    };
    println!("  VAE z_channels: {}", vae_config.z_channels);
    let mut vae = Decoder::new(vae_config.clone())?;

    // Try PyTorch VAE first, then fall back to MLX
    let pytorch_vae_path = std::path::Path::new("models/zimage-turbo-pytorch/vae/diffusion_pytorch_model.safetensors");
    let vae_weights_path = if pytorch_vae_path.exists() {
        Some(pytorch_vae_path.to_path_buf())
    } else {
        vae_path.map(|p| p.clone())
    };

    if let Some(ref path) = vae_weights_path {
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
    // Step 5: Text encoding (Z-Image style - layer 34 only)
    // =========================================================================
    println!("\nStep 5: Encoding text prompt...");

    let batch_size = 1i32;
    let max_seq_len = 512i32;

    // Tokenize
    let (input_ids, attention_mask) = if let Some(ref tok) = tokenizer {
        let encoding = tok.encode(chat_prompt.as_str(), true).map_err(|e| format!("Encode error: {}", e))?;
        let ids: Vec<i32> = encoding.get_ids().iter().map(|&x| x as i32).collect();
        let num_tokens = ids.len().min(max_seq_len as usize);

        // Pad with 151643 (<|endoftext|>)
        let mut padded = vec![151643i32; max_seq_len as usize];
        padded[..num_tokens].copy_from_slice(&ids[..num_tokens]);

        // Attention mask
        let mut mask = vec![0i32; max_seq_len as usize];
        for i in 0..num_tokens {
            mask[i] = 1;
        }

        let ids_arr = Array::from_slice(&padded, &[batch_size, max_seq_len]);
        let mask_arr = Array::from_slice(&mask, &[batch_size, max_seq_len]);
        (ids_arr, Some(mask_arr))
    } else {
        let ids_arr = Array::from_slice(&vec![1i32; max_seq_len as usize], &[batch_size, max_seq_len]);
        let mask_arr = Array::from_slice(&vec![1i32; max_seq_len as usize], &[batch_size, max_seq_len]);
        (ids_arr, Some(mask_arr))
    };

    // Z-Image uses second-to-last layer (layer 34) for 2560-dim embeddings
    // Try to load Python reference text embeddings if available for exact comparison
    let ref_embed_path = std::path::Path::new("/tmp/ref_text_embed.bin");
    let txt_embed = if ref_embed_path.exists() {
        let bytes = std::fs::read(ref_embed_path)?;
        let floats: Vec<f32> = bytes.chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let embed = Array::from_slice(&floats, &[batch_size, max_seq_len, 2560]);
        println!("  Loaded text embeddings from Python reference: {:?}, sum: {:.4}", embed.shape(), embed.sum(None)?.item::<f32>());
        embed
    } else if use_dummy_text {
        // Use random embeddings for testing the architecture
        let embed = mlx_rs::random::normal::<f32>(
            &[batch_size, max_seq_len, 2560],  // cap_feat_dim = 2560
            None, None, None,
        )?;
        println!("  Using dummy text embeddings: {:?}", embed.shape());
        embed
    } else if let Some(ref mut encoder) = text_encoder {
        let start = std::time::Instant::now();
        let txt_embed = match encoder {
            TextEncoder::Quantized(ref mut q) => q.encode_zimage(&input_ids, attention_mask.as_ref())?,
            TextEncoder::NonQuantized(ref mut q) => q.encode_zimage(&input_ids, attention_mask.as_ref())?,
        };
        let txt_embed = txt_embed.as_dtype(mlx_rs::Dtype::Float32)?;
        mlx_rs::transforms::eval([&txt_embed])?;
        println!("  Text embeddings from Rust encoder: {:?} (took {:?})", txt_embed.shape(), start.elapsed());

        // Pad caption to multiple of 32 (as done in MLX_z-image)
        let cap_len = txt_embed.dim(1) as i32;
        let pad_to = ((cap_len + 31) / 32) * 32;
        if pad_to > cap_len {
            // Repeat last token to pad
            let last_token = txt_embed.index((.., (cap_len - 1)..));
            let padding = Array::repeat_axis::<f32>(last_token, pad_to - cap_len, 1)?;
            mlx_rs::ops::concatenate_axis(&[&txt_embed, &padding], 1)?
        } else {
            txt_embed
        }
    } else {
        // Fallback to dummy embeddings
        let embed = mlx_rs::random::normal::<f32>(
            &[batch_size, max_seq_len, 2560],
            None, None, None,
        )?;
        println!("  Using dummy text embeddings (fallback): {:?}", embed.shape());
        embed
    };
    println!("  Final text embeddings: {:?}", txt_embed.shape());

    // =========================================================================
    // Step 6: Generation parameters
    // =========================================================================
    println!("\nStep 6: Setting up generation...");

    let img_height = 512i32;  // Smaller for testing
    let img_width = 512i32;
    let latent_height = img_height / 8;
    let latent_width = img_width / 8;

    // Z-Image uses 2x2 patchify
    let patch_size = zimage_config.patch_size;
    let h_tok = latent_height / patch_size;
    let w_tok = latent_width / patch_size;
    let img_seq_len = h_tok * w_tok;
    let in_channels = zimage_config.in_channels;
    let patch_channels = in_channels * patch_size * patch_size;

    println!("  Image size: {}x{}", img_width, img_height);
    println!("  VAE latent size: {}x{}", latent_width, latent_height);
    println!("  Token grid: {}x{} = {} tokens", w_tok, h_tok, img_seq_len);
    println!("  Patch channels: {} ({}x{}x{})", patch_channels, in_channels, patch_size, patch_size);

    let num_steps = 9;  // Z-Image-Turbo was trained for exactly 9 steps
    println!("  Denoising steps: {}", num_steps);

    // Z-Image's dynamic shift calculation
    let mu = calculate_shift(img_seq_len, 256, 4096, 0.5, 1.15);
    println!("  Dynamic shift mu: {:.4}", mu);

    // =========================================================================
    // Step 7: Create position encodings
    // =========================================================================
    println!("\nStep 7: Creating position encodings...");

    let cap_len = txt_embed.dim(1) as i32;

    // Image positions: original version that produced blurry-but-recognizable images
    // Using (1, h_tok, w_tok) with start (cap_len + 1, 0, 0)
    let img_pos = create_coordinate_grid((1, h_tok, w_tok), (cap_len + 1, 0, 0))?;
    let img_pos = img_pos.reshape(&[1, img_seq_len, 3])?;

    // Caption positions: (cap_len, 1, 1) grid starting at (1, 0, 0)
    let cap_pos = create_coordinate_grid((cap_len, 1, 1), (1, 0, 0))?;
    let cap_pos = cap_pos.reshape(&[1, cap_len, 3])?;

    println!("  Image positions: {:?}", img_pos.shape());
    println!("  Caption positions: {:?}", cap_pos.shape());

    // Pre-compute RoPE frequencies
    let (cos, sin) = transformer.compute_rope(&img_pos, &cap_pos)?;
    println!("  RoPE cos: {:?}, sum: {:.4}", cos.shape(), cos.sum(None)?.item::<f32>());
    println!("  RoPE sin: {:?}, sum: {:.4}", sin.shape(), sin.sum(None)?.item::<f32>());

    // =========================================================================
    // Step 8: Test forward pass (optional - only if reference files exist)
    // =========================================================================
    let ref_x_path = std::path::Path::new("/tmp/ref_x.bin");
    let ref_cap_path = std::path::Path::new("/tmp/ref_cap.bin");

    if ref_x_path.exists() && ref_cap_path.exists() {
        println!("\nStep 8: Testing forward pass (comparing with Python reference)...");

        let test_latents = {
            let bytes = std::fs::read(ref_x_path)?;
            let floats: Vec<f32> = bytes.chunks(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Array::from_slice(&floats, &[batch_size, img_seq_len, patch_channels])
        };
        println!("  Test latents: {:?}, sum: {:.4}", test_latents.shape(), test_latents.sum(None)?.item::<f32>());

        let test_cap = {
            let bytes = std::fs::read(ref_cap_path)?;
            let floats: Vec<f32> = bytes.chunks(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Array::from_slice(&floats, &[batch_size, cap_len, 2560])
        };
        println!("  Test cap_feats: {:?}, sum: {:.4}", test_cap.shape(), test_cap.sum(None)?.item::<f32>());

        let t_test = Array::from_slice(&[0.0f32], &[1]);

        let start = std::time::Instant::now();
        let test_output = transformer.forward_with_rope(
            &test_latents,
            &t_test,
            &test_cap,
            &img_pos,
            &cap_pos,
            &cos,
            &sin,
            None,
            None,
        )?;
        test_output.eval()?;
        println!("  Forward pass took: {:?}", start.elapsed());
        println!("  Test output: {:?}", test_output.shape());
        println!("  Test output sum: {:.4}, range: [{:.4}, {:.4}]",
            test_output.sum(None)?.item::<f32>(),
            test_output.min(None)?.item::<f32>(),
            test_output.max(None)?.item::<f32>());
        println!("  Python reference: sum=-5079.4058, range=[-5.7455, 5.8879]");
    } else {
        println!("\nStep 8: Skipping test forward pass (no reference files)");
    }

    // =========================================================================
    // Step 9: Denoising loop (placeholder)
    // =========================================================================
    println!("\nStep 9: Denoising loop (placeholder)...");
    println!("  Note: Using random weights, output will be noise");

    // Flow matching Euler scheduler
    // sigma=1 is noise, sigma=0 is clean data
    // t = 1 - sigma, so t=0 is noise, t=1 is data
    let sigmas = generate_sigmas(num_steps, mu);
    println!("  Timesteps (t=1 noise -> t=0 clean): {:?}", sigmas);

    // Load exact same initial latents as Python (np.random.seed(42))
    let mut latents = {
        let path = std::path::Path::new("/tmp/ref_initial_latents.bin");
        if path.exists() {
            let bytes = std::fs::read(path)?;
            let floats: Vec<f32> = bytes.chunks(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let arr = Array::from_slice(&floats, &[batch_size, in_channels, latent_height, latent_width]);
            println!("  Loaded initial latents from Python reference");
            arr
        } else {
            mlx_rs::random::seed(42)?;
            println!("  Using MLX random seed 42 (may differ from Python)");
            mlx_rs::random::normal::<f32>(
                &[batch_size, in_channels, latent_height, latent_width],
                None, None, None,
            )?
        }
    };
    println!("  Initial latents: {:?}, sum: {:.4}", latents.shape(), latents.sum(None)?.item::<f32>());

    // Helper function to patchify: [1, C, H, W] -> [1, seq, C*4]
    // Matches reference: x.reshape(C, 1, 1, H_tok, 2, W_tok, 2).transpose(1, 2, 3, 5, 4, 6, 0).reshape(1, -1, C * 4)
    fn patchify(x: &Array, h_tok: i32, w_tok: i32, in_channels: i32) -> Result<Array, mlx_rs::error::Exception> {
        // [1, C, H, W] -> [C, 1, 1, H_tok, 2, W_tok, 2]
        let x = x.reshape(&[in_channels, 1, 1, h_tok, 2, w_tok, 2])?;
        // transpose to [1, 1, H_tok, W_tok, 2, 2, C]
        let x = x.transpose_axes(&[1, 2, 3, 5, 4, 6, 0])?;
        // reshape to [1, seq, C*4]
        x.reshape(&[1, h_tok * w_tok, in_channels * 4])
    }

    // Helper function to unpatchify: [1, seq, C*4] -> [1, C, H, W]
    // Matches reference: out.reshape(1, 1, H_tok, W_tok, 2, 2, C).transpose(6, 0, 1, 2, 4, 3, 5).reshape(1, C, H, W)
    fn unpatchify(x: &Array, h_tok: i32, w_tok: i32, in_channels: i32) -> Result<Array, mlx_rs::error::Exception> {
        // [1, seq, C*4] -> [1, 1, H_tok, W_tok, 2, 2, C]
        let x = x.reshape(&[1, 1, h_tok, w_tok, 2, 2, in_channels])?;
        // transpose to [C, 1, 1, H_tok, 2, W_tok, 2]
        let x = x.transpose_axes(&[6, 0, 1, 2, 4, 3, 5])?;
        // reshape to [1, C, H, W]
        x.reshape(&[1, in_channels, h_tok * 2, w_tok * 2])
    }

    for step in 0..num_steps as usize {
        let sigma_curr = sigmas[step];
        let sigma_next = sigmas[step + 1];
        // Reference passes t_input = 1.0 - t_curr, where t goes from 1 to 0
        // So model receives values from 0 (noise) to 1 (clean)
        let t_model = 1.0 - sigma_curr;
        let t = Array::from_slice(&[t_model], &[1]);

        let start = std::time::Instant::now();

        // Patchify latents for model input
        let latents_patched = patchify(&latents, h_tok, w_tok, in_channels)?;

        let model_out = transformer.forward_with_rope(
            &latents_patched,
            &t,
            &txt_embed,
            &img_pos,
            &cap_pos,
            &cos,
            &sin,
            None,
            None,
        )?;

        // Unpatchify and negate the model output (reference does: noise_pred = -out)
        let noise_pred = unpatchify(&model_out, h_tok, w_tok, in_channels)?;
        let noise_pred = mlx_rs::ops::negative(&noise_pred)?;

        // Scheduler step: prev_sample = sample + dt * noise_pred
        let dt = sigma_next - sigma_curr;  // Negative
        let scaled_noise = mlx_rs::ops::multiply(&noise_pred, &Array::from_slice(&[dt], &[1]))?;
        latents = mlx_rs::ops::add(&latents, &scaled_noise)?;
        latents.eval()?;

        let range = (latents.min(None)?.item::<f32>(), latents.max(None)?.item::<f32>());
        println!("  Step {}/{}: t={:.3}, latents=[{:.4}, {:.4}], took {:?}",
            step + 1, num_steps, t_model, range.0, range.1, start.elapsed());
    }

    // Print final latent sum for comparison with Python
    println!("  Final latents sum: {:.4}", latents.sum(None)?.item::<f32>());
    println!("  Python reference: -8182.2139");

    // =========================================================================
    // Step 10: VAE Decoding
    // =========================================================================
    println!("\nStep 10: VAE Decoding...");

    // Check latent statistics before VAE
    let lat_mean = latents.mean(None)?.item::<f32>();
    let lat_min = latents.min(None)?.item::<f32>();
    let lat_max = latents.max(None)?.item::<f32>();
    println!("  Latent stats: mean={:.3}, range=[{:.3}, {:.3}]", lat_mean, lat_min, lat_max);

    // Latents are already in [B, C, H, W] format from the denoising loop
    // VAE expects [B, H, W, C] format
    let latents = latents.transpose_axes(&[0, 2, 3, 1])?;
    println!("  VAE input: {:?}", latents.shape());
    println!("  VAE input stats: sum={:.4}, range=[{:.4}, {:.4}]",
        latents.sum(None)?.item::<f32>(),
        latents.min(None)?.item::<f32>(),
        latents.max(None)?.item::<f32>());
    // NOTE: scaling (z / scale_factor + shift_factor) is applied inside vae.forward()

    let start = std::time::Instant::now();
    let image = vae.forward(&latents)?;
    image.eval()?;
    println!("  VAE decoding took: {:?}", start.elapsed());
    println!("  VAE raw output: sum={:.4}, range=[{:.4}, {:.4}]",
        image.sum(None)?.item::<f32>(),
        image.min(None)?.item::<f32>(),
        image.max(None)?.item::<f32>());

    // Convert to RGB: VAE outputs [-1, 1], scale to [0, 1], then to [0, 255]
    // Reference: image = (image / 2 + 0.5).clamp(0, 1)
    let image = mlx_rs::ops::divide(&image, &mlx_rs::array!(2.0f32))?;
    let image = mlx_rs::ops::add(&image, &mlx_rs::array!(0.5f32))?;
    let image = mlx_rs::ops::clip(&image, (0.0f32, 1.0f32))?;
    let image = mlx_rs::ops::multiply(&image, &mlx_rs::array!(255.0f32))?;
    let image = image.as_dtype(mlx_rs::Dtype::Uint8)?;
    image.eval()?;

    // Save image as PNG
    let shape = image.shape();
    let (h, w, c) = (shape[1] as u32, shape[2] as u32, shape[3]);
    println!("  Final image: {}x{}x{}", w, h, c);

    let pixels: Vec<u8> = image.as_slice::<u8>().to_vec();
    let output_path = "output_zimage.png";
    let img = image::RgbImage::from_raw(w, h, pixels)
        .ok_or("Failed to create image buffer")?;
    img.save(output_path)?;
    println!("  Saved to: {}", output_path);

    // =========================================================================
    // Step 11: Summary
    // =========================================================================
    println!("\n=== Implementation Complete ===");
    println!("  [x] Text encoder (Qwen3) - encode_zimage() method");
    println!("  [x] Transformer - noise/context refiner + joint blocks");
    println!("  [x] VAE decoder - full pipeline");
    println!("  [x] Weight loading - PyTorch diffusers format");
    println!("\nGenerated image saved to: {}", output_path);

    Ok(())
}

/// Generate sigmas for flow matching (sigma=1-t, goes from 1 to 0)
/// In flow matching: sigma=1 is pure noise, sigma=0 is clean data
fn generate_sigmas(num_steps: i32, mu: f32) -> Vec<f32> {
    let mut sigmas = Vec::with_capacity((num_steps + 1) as usize);
    for i in 0..=num_steps {
        // Linear schedule from 1 (noise) to 0 (clean)
        let sigma_linear = 1.0 - (i as f32) / (num_steps as f32);
        // Apply time shift: exp(mu) / (exp(mu) + (1/sigma - 1))
        let sigma_shifted = if sigma_linear <= 0.0 {
            0.0
        } else if sigma_linear >= 1.0 {
            1.0
        } else {
            mu.exp() / (mu.exp() + (1.0 / sigma_linear - 1.0))
        };
        sigmas.push(sigma_shifted);
    }
    sigmas
}

/// Calculate dynamic time shift for Z-Image
/// Matches MLX_z-image's calculate_shift function
fn calculate_shift(
    image_seq_len: i32,
    base_seq_len: i32,
    max_seq_len: i32,
    base_shift: f32,
    max_shift: f32,
) -> f32 {
    let m = (max_shift - base_shift) / ((max_seq_len - base_seq_len) as f32);
    let b = base_shift - m * (base_seq_len as f32);
    (image_seq_len as f32) * m + b
}

