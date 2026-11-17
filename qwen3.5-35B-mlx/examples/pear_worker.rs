//! Persistent local Qwen3.5/3.8 multimodal worker; JSON lines in/out.
//! The Python HTTP adapter only tokenizes/preprocesses. Both towers run here.
use anyhow::{bail, Context, Result};
use mlx_rs::{module::Module, ops::indexing::IndexOp, Array};
use mlx_rs_core::memory::{self, MemoryGuard};
use qwen3_5_35b_mlx::{load_model, sample, vision::VisionModel, Model};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    io::{self, BufRead, Write},
    time::Instant,
};

#[derive(Deserialize)]
struct Request {
    tensors: String,
    max_tokens: usize,
    #[serde(default)]
    image_ranges: Vec<(i32, i32)>,
    eos_token_ids: Vec<u32>,
    debug_output: Option<String>,
}

fn run(model: &mut Model, vision: &mut VisionModel, r: Request) -> Result<Value> {
    if !(1..=4096).contains(&r.max_tokens) || r.image_ranges.len() > 15 {
        bail!("Token/image budget exceeded");
    }
    let start = Instant::now();
    let t = Array::load_safetensors(&r.tensors)?;
    let get =
        |key: &str| -> Result<&Array> { t.get(key).with_context(|| format!("Missing {key}")) };
    let ids = get("input_ids")?;
    if ids.ndim() != 2 || ids.shape()[0] != 1 || !(1..=8192).contains(&ids.shape()[1]) {
        bail!("Expected one prompt of at most 8192 tokens");
    }
    let len = ids.shape()[1];
    let cos = get("rotary_cos")?;
    let sin = get("rotary_sin")?;
    let rotary_dim = (model.args.text_config.head_dim as f32
        * model.args.text_config.rope_parameters.partial_rotary_factor) as i32;
    if cos.shape() != [1, 1, len + r.max_tokens as i32, rotary_dim] || sin.shape() != cos.shape() {
        bail!("Incorrect multimodal position tensor dimensions");
    }
    let embed = model.text_model.embed_tokens.forward(ids)?;
    let mut parts = Vec::new();
    let mut cursor = 0;
    let mut debug: HashMap<String, Array> = HashMap::new();
    for (i, &(offset, count)) in r.image_ranges.iter().enumerate() {
        if offset < cursor || count <= 0 || offset + count > len {
            bail!("Overlapping/out-of-range image tokens");
        }
        let image_ids = ids.index((.., offset..offset + count));
        image_ids.eval()?;
        if image_ids.as_slice::<u32>().iter().any(|&id| id != 248056) {
            bail!("Image embeddings must replace image placeholder tokens only");
        }
        let features = vision.forward(
            get(&format!("pixels_{i}"))?,
            get(&format!("pos_indices_{i}"))?,
            get(&format!("pos_weights_{i}"))?,
            get(&format!("vision_cos_{i}"))?,
            get(&format!("vision_sin_{i}"))?,
        )?;
        if features.shape() != [count, model.args.text_config.hidden_size] {
            bail!("Image token/feature count mismatch");
        }
        features.eval()?;
        if r.debug_output.is_some() {
            debug.insert(format!("features_{i}"), features.clone());
        }
        if offset > cursor {
            parts.push(embed.index((.., cursor..offset, ..)));
        }
        parts.push(features.as_dtype(embed.dtype())?.reshape(&[1, count, -1])?);
        cursor = offset + count;
    }
    if cursor < len {
        parts.push(embed.index((.., cursor..len, ..)));
    }
    let embeddings = mlx_rs::ops::concatenate_axis(&parts.iter().collect::<Vec<_>>(), 1)?;
    let mut cache = Vec::new();
    let logits = model.forward_embeddings(
        embeddings.as_dtype(mlx_rs::Dtype::Float32)?,
        &mut cache,
        Some((
            &cos.index((.., .., ..len, ..)),
            &sin.index((.., .., ..len, ..)),
        )),
    )?;
    let mut last = logits.index((.., -1, ..));
    if let Some(path) = &r.debug_output {
        debug.insert("logits".into(), last.clone());
        Array::save_safetensors(debug.iter(), None, path)?;
    }
    let mut generated = Vec::new();
    let mut ttft = 0.0;
    let mut finish = "length";
    let mut guard = MemoryGuard::default_guard();
    for step in 0..r.max_tokens {
        let token = sample(&last, 0.0)?;
        token.eval()?;
        let token_id = token.item::<u32>();
        if step == 0 {
            ttft = start.elapsed().as_secs_f64();
        }
        generated.push(token_id);
        if r.eos_token_ids.contains(&token_id) {
            finish = "stop";
            break;
        }
        if step + 1 == r.max_tokens {
            break;
        }
        let pos = len + step as i32;
        let h = model
            .text_model
            .embed_tokens
            .forward(&token.reshape(&[1, 1])?)?;
        last = model
            .forward_embeddings(
                h.as_dtype(mlx_rs::Dtype::Float32)?,
                &mut cache,
                Some((
                    &cos.index((.., .., pos..pos + 1, ..)),
                    &sin.index((.., .., pos..pos + 1, ..)),
                )),
            )?
            .index((.., -1, ..));
        guard.step();
    }
    Ok(
        json!({"token_ids": generated, "finish_reason": finish, "prompt_tokens": len,
        "image_count": r.image_ranges.len(), "ttft_s": ttft, "elapsed_s": start.elapsed().as_secs_f64(),
        "peak_bytes": memory::memory_snapshot().peak}),
    )
}

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .context("Usage: pear_worker MODEL_DIRECTORY")?;
    memory::set_cache_limit(2 * 1024 * 1024 * 1024);
    memory::set_memory_limit(90 * 1024 * 1024 * 1024);
    let mut model = load_model(&path)?;
    let mut vision = VisionModel::load(&path)?;
    println!("{}", json!({"ready": true, "backend": "ominix-mlx"}));
    io::stdout().flush()?;
    for line in io::stdin().lock().lines() {
        let result = serde_json::from_str::<Request>(&line?)
            .map_err(anyhow::Error::from)
            .and_then(|r| run(&mut model, &mut vision, r));
        let response = match result {
            Ok(value) => value,
            Err(error) => json!({"error": format!("{error:#}")}),
        };
        println!("{response}");
        io::stdout().flush()?;
        memory::clear_cache();
    }
    Ok(())
}
