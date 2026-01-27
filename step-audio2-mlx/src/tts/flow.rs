//! CosyVoice2 Flow Matching Decoder for Step-Audio 2
//!
//! Weight-based implementation that directly uses loaded safetensors weights.
//!
//! Architecture:
//! - Codebook: 6561 × 512 embedding
//! - Encoder: input_proj (Linear+LN) + 6 conformer layers
//! - Flow encoder: up_embed + up_layer (conv) + 4 up_encoders + lookahead + norm + proj
//! - DiT decoder: t_embedder + in_proj + 16 blocks (AdaLN + QKnorm attn + conv + MLP) + final_layer
//! - Flow matching: 10-step rectified flow

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::{Array, error::Exception, ops, ops::indexing::IndexOp};

use crate::error::{Error, Result};

/// Flow decoder configuration
#[derive(Debug, Clone)]
pub struct FlowDecoderConfig {
    pub vocab_size: i32,
    pub hidden_dim: i32,
    pub mel_dim: i32,
    pub spk_embed_dim: i32,
    pub num_encoder_blocks: i32,
    pub num_up_blocks: i32,
    pub num_heads: i32,
    pub head_dim: i32,
    pub dit_depth: i32,
    pub num_steps: i32,
}

impl Default for FlowDecoderConfig {
    fn default() -> Self {
        Self {
            vocab_size: 6561,
            hidden_dim: 512,
            mel_dim: 80,
            spk_embed_dim: 192,
            num_encoder_blocks: 6,
            num_up_blocks: 4,
            num_heads: 8,
            head_dim: 64,
            dit_depth: 16,
            num_steps: 10,
        }
    }
}

// =========================================================================
// Helper functions
// =========================================================================

fn linear(x: &Array, w: &Array, b: Option<&Array>) -> std::result::Result<Array, Exception> {
    let out = ops::matmul(x, w.t())?;
    match b {
        Some(bias) => out.add(bias),
        None => Ok(out),
    }
}

fn layer_norm(x: &Array, w: &Array, b: &Array, eps: f32) -> std::result::Result<Array, Exception> {
    mlx_rs::fast::layer_norm(x, Some(w), Some(b), eps)
}

fn gelu(x: &Array) -> std::result::Result<Array, Exception> {
    mlx_rs::nn::gelu_approximate(x)
}

fn silu(x: &Array) -> std::result::Result<Array, Exception> {
    let sigmoid = ops::sigmoid(x)?;
    x.multiply(&sigmoid)
}

fn conv1d_same(x: &Array, w: &Array, b: Option<&Array>) -> std::result::Result<Array, Exception> {
    // Weights from PyTorch are [out_ch, in_ch, kernel], MLX expects [out_ch, kernel, in_ch]
    let w = w.transpose_axes(&[0, 2, 1])?;
    let kernel_size = w.shape()[1];
    let padding = kernel_size / 2;
    let out = ops::conv1d_device(x, &w, None, Some(padding), None, None, mlx_rs::StreamOrDevice::default())?;
    match b {
        Some(bias) => out.add(bias),
        None => Ok(out),
    }
}

fn attention(
    q: &Array, k: &Array, v: &Array,
    num_heads: i32, head_dim: i32,
) -> std::result::Result<Array, Exception> {
    let shape = q.shape();
    let batch = shape[0];
    let seq_len = shape[1];
    let scale = (head_dim as f32).powf(-0.5);

    let q = q.reshape(&[batch, seq_len, num_heads, head_dim])?
        .transpose_axes(&[0, 2, 1, 3])?;
    let k = k.reshape(&[batch, seq_len, num_heads, head_dim])?
        .transpose_axes(&[0, 2, 1, 3])?;
    let v = v.reshape(&[batch, seq_len, num_heads, head_dim])?
        .transpose_axes(&[0, 2, 1, 3])?;

    let attn = mlx_rs::fast::scaled_dot_product_attention(q, k, v, scale, None)?;
    attn.transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[batch, seq_len, num_heads * head_dim])
}

fn timestep_embedding(t: f32, dim: i32) -> Array {
    let half_dim = dim / 2;
    let mut embed = vec![0.0f32; dim as usize];
    for i in 0..half_dim {
        let freq = (-(i as f32) / half_dim as f32 * (10000.0f32).ln()).exp();
        let angle = t * freq;
        embed[i as usize] = angle.cos();
        embed[(i + half_dim) as usize] = angle.sin();
    }
    Array::from_slice(&embed, &[1, dim])
}

// =========================================================================
// Flow Decoder
// =========================================================================

/// CosyVoice2 Flow Decoder with direct weight access
pub struct FlowDecoder {
    pub weights: HashMap<String, Array>,
    pub config: FlowDecoderConfig,
    pub weights_loaded: bool,
}

impl FlowDecoder {
    pub fn new(config: FlowDecoderConfig) -> Result<Self> {
        Ok(Self {
            weights: HashMap::new(),
            config,
            weights_loaded: false,
        })
    }

    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let weights_path = model_dir.join("tts_mlx").join("flow.safetensors");

        if !weights_path.exists() {
            let alt_path = model_dir.join("flow.safetensors");
            if alt_path.exists() {
                return Self::load_from_file(&alt_path);
            }
            return Err(Error::ModelLoad(format!(
                "Flow weights not found at {:?}", weights_path
            )));
        }
        Self::load_from_file(&weights_path)
    }

    fn load_from_file(weights_path: &Path) -> Result<Self> {
        let config = FlowDecoderConfig::default();
        let weights = Array::load_safetensors(weights_path)?;
        println!("  Loaded {} flow weights", weights.len());
        Ok(Self { weights, config, weights_loaded: true })
    }

    fn w(&self, key: &str) -> &Array {
        self.weights.get(key).unwrap_or_else(|| panic!("Missing weight: {}", key))
    }

    // =====================================================================
    // Encoder: codebook + input_proj + 6 conformer layers
    // =====================================================================

    pub fn encode(&self, codes: &[i32]) -> Result<Array> {
        if codes.is_empty() {
            return Err(Error::Inference("Empty codes".into()));
        }

        let codes_array = Array::from_slice(codes, &[1, codes.len() as i32]);
        let codebook_w = self.w("codebook.embeddings.weight");
        let flat_codes = codes_array.reshape(&[-1])
            .map_err(|e| Error::Inference(format!("Reshape codes: {}", e)))?;
        let embeddings = codebook_w.take_axis(&flat_codes, 0)
            .map_err(|e| Error::Inference(format!("Codebook lookup: {}", e)))?;
        let embeddings = embeddings.reshape(&[1, codes.len() as i32, self.config.hidden_dim])
            .map_err(|e| Error::Inference(format!("Reshape embeddings: {}", e)))?;

        let h = linear(
            &embeddings,
            self.w("encoder.input_proj.out.0.weight"),
            Some(self.w("encoder.input_proj.out.0.bias")),
        ).map_err(|e| Error::Inference(format!("Input proj: {}", e)))?;

        let h = layer_norm(
            &h,
            self.w("encoder.input_proj.out.1.weight"),
            self.w("encoder.input_proj.out.1.bias"),
            1e-5,
        ).map_err(|e| Error::Inference(format!("Input proj norm: {}", e)))?;

        let mut h = h;
        for i in 0..self.config.num_encoder_blocks as usize {
            h = self.conformer_block(&h, &format!("encoder.layers.{}", i))
                .map_err(|e| Error::Inference(format!("Encoder block {}: {}", i, e)))?;
        }

        Ok(h)
    }

    fn conformer_block(&self, x: &Array, prefix: &str) -> std::result::Result<Array, Exception> {
        let h = layer_norm(x,
            self.w(&format!("{}.norm_mha.weight", prefix)),
            self.w(&format!("{}.norm_mha.bias", prefix)), 1e-5)?;

        let q = linear(&h, self.w(&format!("{}.self_attn.q_proj.weight", prefix)),
                        Some(self.w(&format!("{}.self_attn.q_proj.bias", prefix))))?;
        let k = linear(&h, self.w(&format!("{}.self_attn.k_proj.weight", prefix)),
                        Some(self.w(&format!("{}.self_attn.k_proj.bias", prefix))))?;
        let v = linear(&h, self.w(&format!("{}.self_attn.v_proj.weight", prefix)),
                        Some(self.w(&format!("{}.self_attn.v_proj.bias", prefix))))?;

        let attn_out = attention(&q, &k, &v, self.config.num_heads, self.config.head_dim)?;
        let attn_out = linear(&attn_out,
                              self.w(&format!("{}.self_attn.out_proj.weight", prefix)),
                              Some(self.w(&format!("{}.self_attn.out_proj.bias", prefix))))?;

        let x = x.add(&attn_out)?;

        let h = layer_norm(&x,
            self.w(&format!("{}.ffn_norm.weight", prefix)),
            self.w(&format!("{}.ffn_norm.bias", prefix)), 1e-5)?;
        let h = linear(&h, self.w(&format!("{}.ffn.up_proj.weight", prefix)),
                        Some(self.w(&format!("{}.ffn.up_proj.bias", prefix))))?;
        let h = gelu(&h)?;
        let h = linear(&h, self.w(&format!("{}.ffn.down_proj.weight", prefix)),
                        Some(self.w(&format!("{}.ffn.down_proj.bias", prefix))))?;

        x.add(&h)
    }

    // =====================================================================
    // Flow Encoder: upsample + conformer + lookahead + norm + proj
    // =====================================================================

    fn flow_encode(&self, h: &Array) -> std::result::Result<Array, Exception> {
        let h = linear(h,
            self.w("flow.encoder.up_embed.out.0.weight"),
            Some(self.w("flow.encoder.up_embed.out.0.bias")))?;
        let h = layer_norm(&h,
            self.w("flow.encoder.up_embed.out.1.weight"),
            self.w("flow.encoder.up_embed.out.1.bias"), 1e-5)?;

        // Upsample 2x
        let shape = h.shape();
        let (batch, seq_len, dim) = (shape[0], shape[1], shape[2]);
        let h_expanded = h.reshape(&[batch, seq_len, 1, dim])?;
        let h_tiled = ops::tile(&h_expanded, &[1, 1, 2, 1])?;
        let mut h = h_tiled.reshape(&[batch, seq_len * 2, dim])?;

        h = conv1d_same(&h,
            self.w("flow.encoder.up_layer.conv.weight"),
            Some(self.w("flow.encoder.up_layer.conv.bias")))?;

        for i in 0..self.config.num_up_blocks as usize {
            h = self.flow_conformer_block(&h, &format!("flow.encoder.up_encoders.{}", i))?;
        }

        // Lookahead convolutions
        let h = conv1d_same(&h,
            self.w("flow.encoder.pre_lookahead_layer.conv1.weight"),
            Some(self.w("flow.encoder.pre_lookahead_layer.conv1.bias")))?;
        let h = gelu(&h)?;
        let h = conv1d_same(&h,
            self.w("flow.encoder.pre_lookahead_layer.conv2.weight"),
            Some(self.w("flow.encoder.pre_lookahead_layer.conv2.bias")))?;
        let h = gelu(&h)?;

        let h = layer_norm(&h,
            self.w("flow.encoder.after_norm.weight"),
            self.w("flow.encoder.after_norm.bias"), 1e-5)?;

        linear(&h,
            self.w("flow.encoder_proj.weight"),
            Some(self.w("flow.encoder_proj.bias")))
    }

    fn flow_conformer_block(&self, x: &Array, prefix: &str) -> std::result::Result<Array, Exception> {
        let h = layer_norm(x,
            self.w(&format!("{}.norm_mha.weight", prefix)),
            self.w(&format!("{}.norm_mha.bias", prefix)), 1e-5)?;

        let q = linear(&h, self.w(&format!("{}.self_attn.linear_q.weight", prefix)),
                        Some(self.w(&format!("{}.self_attn.linear_q.bias", prefix))))?;
        let k = linear(&h, self.w(&format!("{}.self_attn.linear_k.weight", prefix)),
                        Some(self.w(&format!("{}.self_attn.linear_k.bias", prefix))))?;
        let v = linear(&h, self.w(&format!("{}.self_attn.linear_v.weight", prefix)),
                        Some(self.w(&format!("{}.self_attn.linear_v.bias", prefix))))?;

        let attn_out = attention(&q, &k, &v, self.config.num_heads, self.config.head_dim)?;
        let attn_out = linear(&attn_out,
                              self.w(&format!("{}.self_attn.linear_out.weight", prefix)),
                              Some(self.w(&format!("{}.self_attn.linear_out.bias", prefix))))?;

        let x = x.add(&attn_out)?;

        let h = layer_norm(&x,
            self.w(&format!("{}.norm_ff.weight", prefix)),
            self.w(&format!("{}.norm_ff.bias", prefix)), 1e-5)?;
        let h = linear(&h, self.w(&format!("{}.feed_forward.w_1.weight", prefix)),
                        Some(self.w(&format!("{}.feed_forward.w_1.bias", prefix))))?;
        let h = gelu(&h)?;
        let h = linear(&h, self.w(&format!("{}.feed_forward.w_2.weight", prefix)),
                        Some(self.w(&format!("{}.feed_forward.w_2.bias", prefix))))?;

        x.add(&h)
    }

    // =====================================================================
    // DiT Decoder
    // =====================================================================

    fn dit_forward(&self, x_mel: &Array, mu: &Array, t: f32) -> std::result::Result<Array, Exception> {
        let dim = self.config.hidden_dim;

        // Timestep embedding
        let t_emb = timestep_embedding(t * 1000.0, 256);
        let t_emb = linear(&t_emb,
            self.w("flow.decoder.estimator.t_embedder.mlp.0.weight"),
            Some(self.w("flow.decoder.estimator.t_embedder.mlp.0.bias")))?;
        let t_emb = silu(&t_emb)?;
        let t_emb = linear(&t_emb,
            self.w("flow.decoder.estimator.t_embedder.mlp.2.weight"),
            Some(self.w("flow.decoder.estimator.t_embedder.mlp.2.bias")))?;

        // Input: concat x_mel(80) + mu(80) + (x_mel - mu)(80) + spk(80) = 320
        let shape = x_mel.shape();
        let (batch, seq_len) = (shape[0], shape[1]);
        let diff = x_mel.subtract(mu)?;
        let spk = Array::zeros::<f32>(&[batch, seq_len, 80])?; // No speaker embedding for now
        let full_input = ops::concatenate_axis(&[x_mel, mu, &diff, &spk], 2)?;

        let mut h = linear(&full_input,
            self.w("flow.decoder.estimator.in_proj.weight"),
            Some(self.w("flow.decoder.estimator.in_proj.bias")))?;

        // 16 DiT blocks
        for i in 0..self.config.dit_depth as usize {
            h = self.dit_block(&h, &t_emb, i)?;
        }

        // Final layer
        let t_emb_expanded = t_emb.reshape(&[batch, 1, dim])?;
        let adaln = silu(&t_emb_expanded)?;
        let adaln = linear(&adaln,
            self.w("flow.decoder.estimator.final_layer.adaLN_modulation.1.weight"),
            Some(self.w("flow.decoder.estimator.final_layer.adaLN_modulation.1.bias")))?;

        let shift = adaln.index((.., .., 0..dim));
        let scale = adaln.index((.., .., dim..dim*2));

        let h_normed = mlx_rs::fast::layer_norm(&h, None::<&Array>, None::<&Array>, 1e-5)?;
        let one = Array::from_slice(&[1.0f32], &[]);
        let h = h_normed.multiply(&scale.add(&one)?)?.add(&shift)?;

        linear(&h,
            self.w("flow.decoder.estimator.final_layer.linear.weight"),
            Some(self.w("flow.decoder.estimator.final_layer.linear.bias")))
    }

    fn dit_block(&self, x: &Array, t_emb: &Array, block_idx: usize) -> std::result::Result<Array, Exception> {
        let prefix = format!("flow.decoder.estimator.blocks.{}", block_idx);
        let dim = self.config.hidden_dim;
        let shape = x.shape();
        let batch = shape[0];

        // AdaLN modulation
        let t_emb_expanded = t_emb.reshape(&[batch, 1, dim])?;
        let adaln_input = silu(&t_emb_expanded)?;
        let adaln = linear(&adaln_input,
            self.w(&format!("{}.adaLN_modulation.1.weight", prefix)),
            Some(self.w(&format!("{}.adaLN_modulation.1.bias", prefix))))?;

        // Split into 9 chunks of dim
        let shift_attn = adaln.index((.., .., 0..dim));
        let scale_attn = adaln.index((.., .., dim..dim*2));
        let gate_attn = adaln.index((.., .., dim*2..dim*3));
        let shift_conv = adaln.index((.., .., dim*3..dim*4));
        let scale_conv = adaln.index((.., .., dim*4..dim*5));
        let gate_conv = adaln.index((.., .., dim*5..dim*6));
        let shift_mlp = adaln.index((.., .., dim*6..dim*7));
        let scale_mlp = adaln.index((.., .., dim*7..dim*8));
        let gate_mlp = adaln.index((.., .., dim*8..dim*9));

        let one = Array::from_slice(&[1.0f32], &[]);

        // 1. Self-attention with AdaLN
        let h_normed = mlx_rs::fast::layer_norm(x, None::<&Array>, None::<&Array>, 1e-5)?;
        let h = h_normed.multiply(&scale_attn.add(&one)?)?.add(&shift_attn)?;

        let q = linear(&h, self.w(&format!("{}.attn.to_q.weight", prefix)),
                        Some(self.w(&format!("{}.attn.to_q.bias", prefix))))?;
        let k = linear(&h, self.w(&format!("{}.attn.to_k.weight", prefix)),
                        Some(self.w(&format!("{}.attn.to_k.bias", prefix))))?;
        let v = linear(&h, self.w(&format!("{}.attn.to_v.weight", prefix)),
                        Some(self.w(&format!("{}.attn.to_v.bias", prefix))))?;

        let q = self.per_head_norm(&q, &format!("{}.attn.q_norm", prefix))?;
        let k = self.per_head_norm(&k, &format!("{}.attn.k_norm", prefix))?;

        let attn_out = attention(&q, &k, &v, self.config.num_heads, self.config.head_dim)?;
        let attn_out = linear(&attn_out,
                              self.w(&format!("{}.attn.proj.weight", prefix)),
                              Some(self.w(&format!("{}.attn.proj.bias", prefix))))?;

        let x = x.add(&attn_out.multiply(&gate_attn)?)?;

        // 2. Conv block with AdaLN
        let h_normed = mlx_rs::fast::layer_norm(&x, None::<&Array>, None::<&Array>, 1e-5)?;
        let h = h_normed.multiply(&scale_conv.add(&one)?)?.add(&shift_conv)?;

        // conv.block: [0]=SiLU, [1]=Conv1d, [2]=SiLU, [3]=LayerNorm, [4]=Dropout, [5]=SiLU, [6]=Conv1d
        let h = silu(&h)?;
        let h = conv1d_same(&h,
            self.w(&format!("{}.conv.block.1.weight", prefix)),
            Some(self.w(&format!("{}.conv.block.1.bias", prefix))))?;
        let h = layer_norm(&h,
            self.w(&format!("{}.conv.block.3.weight", prefix)),
            self.w(&format!("{}.conv.block.3.bias", prefix)), 1e-5)?;
        let h = silu(&h)?;
        let h = conv1d_same(&h,
            self.w(&format!("{}.conv.block.6.weight", prefix)),
            Some(self.w(&format!("{}.conv.block.6.bias", prefix))))?;

        let x = x.add(&h.multiply(&gate_conv)?)?;

        // 3. MLP with AdaLN
        let h_normed = mlx_rs::fast::layer_norm(&x, None::<&Array>, None::<&Array>, 1e-5)?;
        let h = h_normed.multiply(&scale_mlp.add(&one)?)?.add(&shift_mlp)?;

        let h = linear(&h, self.w(&format!("{}.mlp.fc1.weight", prefix)),
                        Some(self.w(&format!("{}.mlp.fc1.bias", prefix))))?;
        let h = gelu(&h)?;
        let h = linear(&h, self.w(&format!("{}.mlp.fc2.weight", prefix)),
                        Some(self.w(&format!("{}.mlp.fc2.bias", prefix))))?;

        x.add(&h.multiply(&gate_mlp)?)
    }

    fn per_head_norm(&self, x: &Array, prefix: &str) -> std::result::Result<Array, Exception> {
        let shape = x.shape();
        let (batch, seq_len) = (shape[0], shape[1]);
        let h = x.reshape(&[batch * seq_len, self.config.num_heads, self.config.head_dim])?;
        let w = self.w(&format!("{}.weight", prefix));
        let b = self.w(&format!("{}.bias", prefix));
        let h = mlx_rs::fast::layer_norm(&h, Some(w), Some(b), 1e-5)?;
        h.reshape(&[batch, seq_len, self.config.hidden_dim])
    }

    // =====================================================================
    // Flow matching generation
    // =====================================================================

    pub fn generate(&self, codes: &[i32]) -> Result<Array> {
        let encoder_out = self.encode(codes)?;

        let mu = self.flow_encode(&encoder_out)
            .map_err(|e| Error::Inference(format!("Flow encode: {}", e)))?;

        let shape = mu.shape();
        let (batch, mel_len, mel_dim) = (shape[0], shape[1], shape[2]);

        let mut x = mlx_rs::random::normal::<f32>(&[batch, mel_len, mel_dim], None, None, None)
            .map_err(|e| Error::Inference(format!("Sample prior: {}", e)))?;

        let num_steps = self.config.num_steps;
        let schedule: Vec<f32> = (0..=num_steps)
            .map(|i| 1.0 - i as f32 / num_steps as f32)
            .collect();

        for i in 0..num_steps as usize {
            let t = schedule[i];
            let t_next = schedule[i + 1];

            let v = self.dit_forward(&x, &mu, t)
                .map_err(|e| Error::Inference(format!("DiT step {}: {}", i, e)))?;

            let dt = Array::from_slice(&[t - t_next], &[]);
            let dx = v.multiply(&dt)
                .map_err(|e| Error::Inference(format!("Flow step: {}", e)))?;
            x = x.subtract(&dx)
                .map_err(|e| Error::Inference(format!("Flow update: {}", e)))?;
        }

        x.transpose_axes(&[0, 2, 1])
            .map_err(|e| Error::Inference(format!("Final transpose: {}", e)))
    }
}

/// Empty Codebook struct kept for backward compatibility
pub struct Codebook;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flow_decoder_config() {
        let config = FlowDecoderConfig::default();
        assert_eq!(config.vocab_size, 6561);
        assert_eq!(config.hidden_dim, 512);
        assert_eq!(config.mel_dim, 80);
    }

    #[test]
    fn test_timestep_embedding() {
        let emb = timestep_embedding(0.5, 256);
        assert_eq!(emb.shape(), &[1, 256]);
    }
}
