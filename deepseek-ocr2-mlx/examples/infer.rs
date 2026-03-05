//! DeepSeek-OCR-2 inference example.
//!
//! Usage:
//!   cargo run --example infer --release -- \
//!     --model-dir ./models/DeepSeek-OCR-2 \
//!     --image test.jpg \
//!     --prompt "<image>\n<|grounding|>Convert the document to markdown."

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

use mlx_rs::module::Module;

use deepseek_ocr2_mlx::{
    load_model, load_tokenizer, tokenize_prompt, find_best_crop_ratio,
    Generate, GenerateState,
};

#[derive(Parser)]
#[command(name = "deepseek-ocr2-infer")]
struct Args {
    /// Path to the model directory
    #[arg(long, default_value = "./models/DeepSeek-OCR-2")]
    model_dir: PathBuf,

    /// Path to image file
    #[arg(long)]
    image: Option<PathBuf>,

    /// Prompt (use <image> as placeholder for the image)
    #[arg(long, default_value = "<image>\nDescribe this image.")]
    prompt: String,

    /// Base resolution for global view
    #[arg(long, default_value_t = 1024)]
    base_size: u32,

    /// Crop resolution
    #[arg(long, default_value_t = 768)]
    image_size: u32,

    /// Enable dynamic cropping
    #[arg(long, default_value_t = true)]
    crop_mode: bool,

    /// Max tokens to generate
    #[arg(long, default_value_t = 8192)]
    max_tokens: usize,

    /// Temperature (0 = greedy)
    #[arg(long, default_value_t = 0.0)]
    temp: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Load model and tokenizer
    let tokenizer = load_tokenizer(&args.model_dir)?;
    let mut model = load_model(&args.model_dir)?;

    let has_image = args.image.is_some() && args.prompt.contains("<image>");

    // Process image if provided
    let (crop_images, global_image, crop_ratio) = if let Some(ref image_path) = args.image {
        eprintln!("Loading image: {:?}", image_path);
        let img = image::open(image_path)?.to_rgb8();
        let (w, h) = (img.width(), img.height());
        eprintln!("Image size: {}x{}", w, h);

        // Determine crop ratio
        let crop_ratio = if args.crop_mode && (w > args.image_size || h > args.image_size) {
            find_best_crop_ratio(w, h, 2, 6)
        } else {
            (1, 1)
        };
        eprintln!("Crop ratio: {:?}", crop_ratio);

        // Create global view (pad to base_size x base_size)
        let base = args.base_size;
        let mut global = image::RgbImage::new(base, base);
        // Fill with gray (0.5 * 255 = 128)
        for pixel in global.pixels_mut() {
            *pixel = image::Rgb([128, 128, 128]);
        }
        // Resize and center
        let scale = (base as f32 / w as f32).min(base as f32 / h as f32);
        let new_w = (w as f32 * scale) as u32;
        let new_h = (h as f32 * scale) as u32;
        let resized = image::imageops::resize(&img, new_w, new_h, image::imageops::FilterType::Lanczos3);
        let x_offset = (base - new_w) / 2;
        let y_offset = (base - new_h) / 2;
        image::imageops::overlay(&mut global, &resized, x_offset as i64, y_offset as i64);

        // Normalize: (pixel / 255 - 0.5) / 0.5 = pixel / 127.5 - 1.0
        let global_data: Vec<f32> = global
            .pixels()
            .flat_map(|p| p.0.iter().map(|&v| v as f32 / 127.5 - 1.0))
            .collect();
        let global_arr = mlx_rs::Array::from_slice(
            &global_data,
            &[1, base as i32, base as i32, 3],
        );

        // Create crop patches if needed
        let crop_arr = if crop_ratio.0 > 1 || crop_ratio.1 > 1 {
            let is = args.image_size;
            let target_w = is * crop_ratio.0 as u32;
            let target_h = is * crop_ratio.1 as u32;
            let resized_full = image::imageops::resize(&img, target_w, target_h, image::imageops::FilterType::Lanczos3);

            let n_crops = crop_ratio.0 * crop_ratio.1;
            let mut crop_data: Vec<f32> = Vec::with_capacity((n_crops as usize) * (is * is * 3) as usize);

            for cy in 0..crop_ratio.1 {
                for cx in 0..crop_ratio.0 {
                    let x0 = cx as u32 * is;
                    let y0 = cy as u32 * is;
                    for y in y0..y0 + is {
                        for x in x0..x0 + is {
                            let pixel = resized_full.get_pixel(x, y);
                            for &v in pixel.0.iter() {
                                crop_data.push(v as f32 / 127.5 - 1.0);
                            }
                        }
                    }
                }
            }

            Some(mlx_rs::Array::from_slice(
                &crop_data,
                &[n_crops, is as i32, is as i32, 3],
            ))
        } else {
            None
        };

        (crop_arr, Some(global_arr), crop_ratio)
    } else {
        (None, None, (1i32, 1i32))
    };

    // Tokenize prompt
    let (token_ids, seq_mask) = tokenize_prompt(
        &tokenizer,
        &args.prompt,
        has_image,
        args.base_size as i32,
        args.image_size as i32,
        crop_ratio,
    )?;

    eprintln!(
        "Input tokens: {}, image tokens: {}",
        token_ids.len(),
        seq_mask.iter().filter(|&&x| x).count()
    );

    let input_ids = mlx_rs::Array::from_iter(
        token_ids.iter().copied(),
        &[1, token_ids.len() as i32],
    );
    // Encode image and prepare inputs
    let embeds = if has_image {
        let global_img = global_image.as_ref().unwrap();
        eprintln!("Encoding image through vision pipeline...");
        let visual_features = model.encode_image(
            crop_images.as_ref(),
            global_img,
        )?;
        eprintln!("Visual features: {:?}", visual_features.shape());

        let seq_mask_bool = mlx_rs::Array::from_iter(
            seq_mask.iter().map(|&b| b),
            &[1, seq_mask.len() as i32],
        );
        model.prepare_inputs(&input_ids, &seq_mask_bool, &visual_features)?
    } else {
        model.embed_tokens.forward(&input_ids)?
    };

    // Generate
    eprintln!("Generating...");
    let mut cache = model.init_cache();
    let eos_id = model.config.eos_token_id;

    let mut gen = Generate {
        model: &mut model,
        cache: &mut cache,
        temp: args.temp,
        state: GenerateState::Prefill { embeds },
        eos_token_id: eos_id,
        repetition_penalty: 1.1,
        repetition_context_size: 512,
        generated_tokens: Vec::new(),
    };

    let mut all_tokens: Vec<u32> = Vec::new();
    let mut prev_text_len = 0;
    let gen_start = std::time::Instant::now();
    let mut first_token_time = None;

    for (i, token_result) in gen.by_ref().take(args.max_tokens).enumerate() {
        let token = token_result?;
        let token_id: i32 = token.item();

        if i == 0 {
            first_token_time = Some(gen_start.elapsed());
        }

        if token_id == eos_id {
            break;
        }

        all_tokens.push(token_id as u32);

        // Incremental decode
        let text = tokenizer
            .decode(&all_tokens, true)
            .unwrap_or_default();
        let new_text = &text[prev_text_len..];
        if !new_text.is_empty() {
            print!("{}", new_text);
            prev_text_len = text.len();
        }

        // Flush periodically
        if i % 10 == 0 {
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
    }

    let total_time = gen_start.elapsed();
    println!();
    eprintln!("Generated {} tokens", all_tokens.len());
    if let Some(ttft) = first_token_time {
        eprintln!("Time to first token: {:.2}s", ttft.as_secs_f64());
    }
    if all_tokens.len() > 1 {
        let decode_time = total_time.as_secs_f64() - first_token_time.map(|t| t.as_secs_f64()).unwrap_or(0.0);
        let decode_tokens = all_tokens.len() - 1;
        eprintln!("Decode: {:.1} tok/s ({} tokens in {:.2}s)", decode_tokens as f64 / decode_time, decode_tokens, decode_time);
    }
    eprintln!("Total: {:.2}s", total_time.as_secs_f64());

    Ok(())
}
