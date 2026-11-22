//! Native MLX inference for PaddleOCR-VL-1.6 single-image text spotting.
mod decoder;
mod image_input;
mod spotting;
mod vision;
mod weights;

use anyhow::{ensure, Context, Result};
use decoder::Decoder;
use image::{DynamicImage, GenericImageView};
use mlx_rs::ops::concatenate_axis;
use serde::{Deserialize, Serialize};
pub use spotting::Region;
use std::{fs, path::Path};
use tokenizers::Tokenizer;

#[derive(Deserialize)]
pub(crate) struct VisionConfig {
    pub hidden_size: i32,
    pub num_attention_heads: i32,
    pub num_hidden_layers: usize,
    pub patch_size: i32,
    pub spatial_merge_size: i32,
    pub layer_norm_eps: f32,
    pub image_size: i32,
}
#[derive(Deserialize)]
struct RopeConfig {
    mrope_section: [usize; 3],
}
#[derive(Deserialize)]
struct Config {
    model_type: String,
    image_token_id: u32,
    num_hidden_layers: usize,
    head_dim: i32,
    num_attention_heads: i32,
    num_key_value_heads: i32,
    rms_norm_eps: f32,
    rope_theta: f32,
    rope_scaling: RopeConfig,
    vision_config: VisionConfig,
}

pub struct PaddleOcr {
    cfg: Config,
    tokenizer: Tokenizer,
    vision: vision::Vision,
    decoder: Decoder,
    prefix: Vec<u32>,
    suffix: Vec<u32>,
    eos: u32,
}

#[derive(Serialize)]
pub struct SpottingResult {
    pub width: u32,
    pub height: u32,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub finished: bool,
    pub regions: Vec<Region>,
    pub raw: String,
}

impl PaddleOcr {
    /// Load the official single-file BF16 checkpoint once; each call to spot creates fresh page state.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let cfg: Config = serde_json::from_slice(&fs::read(dir.join("config.json"))?)?;
        ensure!(
            cfg.model_type == "paddleocr_vl",
            "expected model_type paddleocr_vl"
        );
        ensure!(
            cfg.rope_scaling.mrope_section.iter().sum::<usize>() * 2 == cfg.head_dim as usize,
            "invalid mRoPE dimensions"
        );
        ensure!(
            cfg.head_dim > 0
                && cfg.num_key_value_heads > 0
                && cfg.num_attention_heads % cfg.num_key_value_heads == 0
                && cfg.num_hidden_layers > 0,
            "invalid decoder config"
        );
        let v = &cfg.vision_config;
        ensure!(
            v.patch_size == 14
                && v.spatial_merge_size == 2
                && v.hidden_size > 0
                && v.num_attention_heads > 0
                && v.hidden_size % v.num_attention_heads == 0
                && v.num_hidden_layers > 0
                && v.image_size >= v.patch_size,
            "unsupported vision config"
        );
        let tokenizer =
            Tokenizer::from_file(dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
        let eos = tokenizer.token_to_id("</s>").context("missing EOS token")?;
        let prompt="<|begin_of_sentence|>User: <|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>Spotting:\nAssistant:\n";
        let encoding = tokenizer
            .encode(prompt, false)
            .map_err(anyhow::Error::msg)?;
        let ids = encoding.get_ids();
        let index = ids
            .iter()
            .position(|t| *t == cfg.image_token_id)
            .context("image token missing in prompt")?;
        let prefix = ids[..index].to_vec();
        let suffix = ids[index + 1..].to_vec();
        let mut w = weights::Weights::load(dir)?;
        let vision = vision::Vision::load(&mut w, v)?;
        let decoder = Decoder::load(&mut w, &cfg)?;
        Ok(Self {
            cfg,
            tokenizer,
            vision,
            decoder,
            prefix,
            suffix,
            eos,
        })
    }

    /// Greedy spotting. A token-limited result exposes only complete polygons and sets finished=false.
    pub fn spot(&self, image: &DynamicImage, max_tokens: usize) -> Result<SpottingResult> {
        ensure!(max_tokens > 0, "max_tokens must be positive");
        let (width, height) = image.dimensions();
        let cfg = &self.cfg.vision_config;
        let input = image_input::prepare(image, cfg.patch_size, cfg.spatial_merge_size)?;
        let [h, w] = input.grid;
        let features = self.vision.encode(&input.pixels, h, w)?;
        let (positions, mut offset) = decoder::positions(
            self.prefix.len() as i32,
            h / cfg.spatial_merge_size,
            w / cfg.spatial_merge_size,
            self.suffix.len() as i32,
        );
        let embeds = concatenate_axis(
            &[
                self.decoder.embed(&self.prefix)?,
                features,
                self.decoder.embed(&self.suffix)?,
            ],
            0,
        )?;
        ensure!(
            embeds.shape()[0] as usize == positions.len(),
            "image features and prompt positions disagree"
        );
        let mut cache = self.decoder.cache();
        let mut token = self.decoder.prefill(embeds, &positions, &mut cache)?;
        let mut ids = Vec::new();
        let mut finished = false;
        for i in 0..max_tokens {
            let id = token.try_item::<u32>()?;
            if id == self.eos {
                finished = true;
                break;
            }
            ids.push(id);
            if i + 1 < max_tokens {
                token = self.decoder.step(&token, offset, &mut cache)?;
                offset += 1;
            }
        }
        let raw = self
            .tokenizer
            .decode(&ids, false)
            .map_err(anyhow::Error::msg)?;
        let regions = spotting::parse(&raw, width, height, !finished)?;
        Ok(SpottingResult {
            width,
            height,
            prompt_tokens: positions.len(),
            generated_tokens: ids.len() + usize::from(finished),
            finished,
            regions,
            raw,
        })
    }
}
