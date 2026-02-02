//! Full precision Qwen-Image generation example
//!
//! Tests loading full precision weights from Qwen/Qwen-Image
//!
//! Usage:
//!     cargo run --release --example generate_full_precision

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use clap::Parser;
use mlx_rs::Array;
use safetensors::SafeTensors;

use qwen_image_mlx::qwen_full_precision::{QwenFullConfig, QwenFullTransformer, load_full_precision_weights};

#[derive(Parser, Debug)]
#[command(name = "generate_full_precision")]
#[command(about = "Test full precision Qwen-Image transformer")]
struct Args {
    /// Model path (HuggingFace cache or local)
    #[arg(long)]
    model_path: Option<String>,
}

fn find_model_path() -> Result<String, Box<dyn std::error::Error>> {
    let home = std::env::var("HOME")?;
    let cache_path = format!(
        "{}/.cache/huggingface/hub/models--Qwen--Qwen-Image/snapshots",
        home
    );

    if Path::new(&cache_path).exists() {
        for entry in std::fs::read_dir(&cache_path)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                let transformer_path = entry.path().join("transformer");
                if transformer_path.exists() {
                    return Ok(entry.path().to_string_lossy().to_string());
                }
            }
        }
    }

    Err("Model not found. Please download with:\n  huggingface-cli download Qwen/Qwen-Image".into())
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("=== Full Precision Qwen-Image Transformer Test ===");

    // Find model path
    let model_path = match &args.model_path {
        Some(p) => p.clone(),
        None => find_model_path()?,
    };
    println!("Using model: {}", model_path);

    // Load transformer weights
    println!("\nLoading transformer weights...");
    let transformer_dir = Path::new(&model_path).join("transformer");
    let start = Instant::now();
    let weights = load_safetensors_weights(&transformer_dir)?;
    println!("  Loaded {} weights in {:.2}s", weights.len(), start.elapsed().as_secs_f32());

    // Print some weight info
    println!("\nWeight summary:");
    let mut weight_names: Vec<_> = weights.keys().collect();
    weight_names.sort();
    for name in weight_names.iter().take(10) {
        let w = &weights[*name];
        mlx_rs::transforms::eval([w]).ok();
        println!("  {}: {:?}, range: [{:.4}, {:.4}]",
            name, w.shape(),
            w.min(None).unwrap().item::<f32>(),
            w.max(None).unwrap().item::<f32>());
    }
    println!("  ... and {} more weights", weights.len() - 10);

    // Create and initialize transformer
    println!("\nInitializing transformer...");
    let config = QwenFullConfig::default();
    println!("  Config: {:?}", config);

    let start = Instant::now();
    let mut transformer = QwenFullTransformer::new(config)?;
    println!("  Created in {:.2}s", start.elapsed().as_secs_f32());

    let start = Instant::now();
    load_full_precision_weights(&mut transformer, weights)?;
    println!("  Weights loaded in {:.2}s", start.elapsed().as_secs_f32());

    // Test forward pass with dummy inputs
    println!("\nTesting forward pass...");
    let batch = 1;
    let latent_h = 32i32;
    let latent_w = 32i32;
    let img_seq = (latent_h * latent_w) as i32;  // 1024 patches
    let txt_seq = 77;

    // Generate random inputs (small values to avoid numerical issues)
    let seed = 42u64;
    let key = mlx_rs::random::key(seed)?;
    let img = mlx_rs::random::normal::<f32>(&[batch, img_seq, 64], None, None, Some(&key))?;
    let key2 = mlx_rs::random::key(seed + 1)?;
    let txt = mlx_rs::random::normal::<f32>(&[batch, txt_seq, 3584], None, None, Some(&key2))?;
    let timestep = Array::from_f32(0.5).reshape(&[batch])?;

    // Generate RoPE embeddings (Qwen style with complex values)
    let theta = 10000.0f32;
    let axes_dim = [16i32, 56i32, 56i32];  // frame, height, width

    fn compute_freqs(dim: i32, theta: f32) -> Vec<f32> {
        (0..dim/2).map(|i| {
            let scale = (i as f32 * 2.0) / dim as f32;
            1.0 / theta.powf(scale)
        }).collect()
    }

    let frame_freqs = compute_freqs(axes_dim[0], theta);
    let height_freqs = compute_freqs(axes_dim[1], theta);
    let width_freqs = compute_freqs(axes_dim[2], theta);

    // Build RoPE for each patch position
    let frame = 1;
    let half_height = (latent_h / 2) as usize;
    let half_width = (latent_w / 2) as usize;

    let mut img_cos_data: Vec<f32> = Vec::with_capacity((img_seq * 64) as usize);
    let mut img_sin_data: Vec<f32> = Vec::with_capacity((img_seq * 64) as usize);

    for _f in 0..frame {
        for h in 0..latent_h as usize {
            for w in 0..latent_w as usize {
                // Frame: use positive index 0
                for &freq in &frame_freqs {
                    img_cos_data.push((0.0 * freq).cos());
                    img_sin_data.push((0.0 * freq).sin());
                }

                // Height: centered positions
                let h_pos = if h < half_height {
                    -(((latent_h as usize - half_height) - h) as i32) as f32
                } else {
                    (h - half_height) as f32
                };
                for &freq in &height_freqs {
                    img_cos_data.push((h_pos * freq).cos());
                    img_sin_data.push((h_pos * freq).sin());
                }

                // Width: centered positions
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
    }

    let img_cos = Array::from_slice(&img_cos_data, &[img_seq, 64]);
    let img_sin = Array::from_slice(&img_sin_data, &[img_seq, 64]);

    // Text RoPE: positions start after max_vid_index
    let max_vid_index = half_height.max(half_width) as i32;
    let mut txt_cos_data: Vec<f32> = Vec::with_capacity((txt_seq * 64) as usize);
    let mut txt_sin_data: Vec<f32> = Vec::with_capacity((txt_seq * 64) as usize);

    for i in 0..txt_seq as usize {
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

    let txt_cos = Array::from_slice(&txt_cos_data, &[txt_seq, 64]);
    let txt_sin = Array::from_slice(&txt_sin_data, &[txt_seq, 64]);

    println!("  RoPE generated: img {:?}, txt {:?}", img_cos.shape(), txt_cos.shape());

    let start = Instant::now();
    let output = transformer.forward(&img, &txt, &timestep,
        Some((&img_cos, &img_sin)),
        Some((&txt_cos, &txt_sin)))?;
    mlx_rs::transforms::eval([&output]).ok();
    println!("  Forward pass completed in {:.2}s", start.elapsed().as_secs_f32());
    println!("  Output shape: {:?}", output.shape());
    println!("  Output range: [{:.4}, {:.4}]",
        output.min(None).unwrap().item::<f32>(),
        output.max(None).unwrap().item::<f32>());

    println!("\n=== Test Complete ===");
    println!("Full precision transformer is working!");
    println!("\nNote: Full image generation pipeline integration is still needed:");
    println!("  - RoPE embeddings");
    println!("  - Scheduler integration");
    println!("  - VAE decoding");

    Ok(())
}
