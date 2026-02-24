//! Qwen3-ASR Audio Encoder (AuT).
//!
//! Conv2d frontend + Transformer encoder with windowed attention.
//! Processes mel spectrograms in chunks of 100 frames.

use crate::error::Result;
use mlx_rs::builder::Builder;
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::Module;
use mlx_rs::nn;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Array;

/// Audio encoder configuration.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct AudioEncoderConfig {
    #[serde(default = "default_num_mel_bins")]
    pub num_mel_bins: i32,
    #[serde(default = "default_encoder_layers")]
    pub encoder_layers: i32,
    #[serde(default = "default_encoder_attention_heads")]
    pub encoder_attention_heads: i32,
    #[serde(default = "default_encoder_ffn_dim")]
    pub encoder_ffn_dim: i32,
    #[serde(default = "default_d_model")]
    pub d_model: i32,
    #[serde(default = "default_max_source_positions")]
    pub max_source_positions: i32,
    #[serde(default = "default_n_window")]
    pub n_window: i32,
    #[serde(default = "default_output_dim")]
    pub output_dim: i32,
    #[serde(default = "default_n_window_infer")]
    pub n_window_infer: i32,
    #[serde(default = "default_conv_chunksize")]
    pub conv_chunksize: i32,
    #[serde(default = "default_downsample_hidden_size")]
    pub downsample_hidden_size: i32,
    #[serde(default)]
    pub scale_embedding: bool,
}

fn default_num_mel_bins() -> i32 { 128 }
fn default_encoder_layers() -> i32 { 24 }
fn default_encoder_attention_heads() -> i32 { 16 }
fn default_encoder_ffn_dim() -> i32 { 4096 }
fn default_d_model() -> i32 { 1024 }
fn default_max_source_positions() -> i32 { 1500 }
fn default_n_window() -> i32 { 50 }
fn default_output_dim() -> i32 { 2048 }
fn default_n_window_infer() -> i32 { 800 }
fn default_conv_chunksize() -> i32 { 500 }
fn default_downsample_hidden_size() -> i32 { 480 }

impl Default for AudioEncoderConfig {
    fn default() -> Self {
        Self {
            num_mel_bins: 128,
            encoder_layers: 24,
            encoder_attention_heads: 16,
            encoder_ffn_dim: 4096,
            d_model: 1024,
            max_source_positions: 1500,
            n_window: 50,
            output_dim: 2048,
            n_window_infer: 800,
            conv_chunksize: 500,
            downsample_hidden_size: 480,
            scale_embedding: false,
        }
    }
}

/// Compute output length after conv layers for a given input length.
/// Accounts for 100-frame chunk processing (13 tokens per chunk).
fn get_feat_extract_output_lengths(input_length: i32) -> i32 {
    let leave = input_length % 100;
    let feat_len = (leave - 1) / 2 + 1;
    ((feat_len - 1) / 2 + 1 - 1) / 2 + 1 + (input_length / 100) * 13
}

/// Pre-computed sinusoidal position embeddings.
#[derive(Debug, Clone, ModuleParameters)]
pub struct SinusoidalPositionEmbedding {
    embedding: Array,
}

impl SinusoidalPositionEmbedding {
    pub fn new(length: i32, channels: i32) -> Result<Self> {
        let half = channels / 2;
        let log_timescale = (10000.0f32).ln() / (half - 1) as f32;

        let mut data = vec![0.0f32; length as usize * channels as usize];

        for pos in 0..length as usize {
            for i in 0..half as usize {
                let scaled = pos as f32 * (-log_timescale * i as f32).exp();
                data[pos * channels as usize + i] = scaled.sin();
                data[pos * channels as usize + half as usize + i] = scaled.cos();
            }
        }

        let embedding = Array::from_slice(&data, &[length, channels]);
        Ok(Self { embedding })
    }

    pub fn forward(&self, seq_len: i32) -> std::result::Result<Array, mlx_rs::error::Exception> {
        Ok(self.embedding.index(..seq_len))
    }
}

/// Multi-head attention for audio encoder (with bias).
#[derive(Debug, Clone, ModuleParameters)]
pub struct AudioAttention {
    #[param]
    pub q_proj: nn::Linear,
    #[param]
    pub k_proj: nn::Linear,
    #[param]
    pub v_proj: nn::Linear,
    #[param]
    pub out_proj: nn::Linear,

    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl AudioAttention {
    pub fn new(config: &AudioEncoderConfig) -> Result<Self> {
        let embed_dim = config.d_model;
        let num_heads = config.encoder_attention_heads;
        let head_dim = embed_dim / num_heads;

        Ok(Self {
            q_proj: nn::LinearBuilder::new(embed_dim, embed_dim).bias(true).build()?,
            k_proj: nn::LinearBuilder::new(embed_dim, embed_dim).bias(true).build()?,
            v_proj: nn::LinearBuilder::new(embed_dim, embed_dim).bias(true).build()?,
            out_proj: nn::LinearBuilder::new(embed_dim, embed_dim).bias(true).build()?,
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    pub fn forward_attn(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
    ) -> std::result::Result<Array, mlx_rs::error::Exception> {
        let shape = x.shape();
        let (bsz, seq_len) = (shape[0], shape[1]);

        let q = self.q_proj.forward(x)?.multiply(&Array::from(self.scale))?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape(&[bsz, seq_len, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = k.reshape(&[bsz, seq_len, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = v.reshape(&[bsz, seq_len, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // scale=1.0 because we already scaled q
        let attn_out = match mask {
            Some(m) => mlx_rs::fast::scaled_dot_product_attention(
                q, k, v, 1.0,
                mlx_rs::fast::ScaledDotProductAttentionMask::Array(m),
            )?,
            None => mlx_rs::fast::scaled_dot_product_attention(
                q, k, v, 1.0,
                None::<mlx_rs::fast::ScaledDotProductAttentionMask>,
            )?,
        };

        let attn_out = attn_out
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[bsz, seq_len, self.num_heads * self.head_dim])?;

        self.out_proj.forward(&attn_out)
    }
}

/// Single audio encoder transformer layer (pre-norm).
#[derive(Debug, Clone, ModuleParameters)]
pub struct AudioEncoderLayer {
    #[param]
    pub self_attn: AudioAttention,
    #[param]
    pub self_attn_layer_norm: nn::LayerNorm,
    #[param]
    pub fc1: nn::Linear,
    #[param]
    pub fc2: nn::Linear,
    #[param]
    pub final_layer_norm: nn::LayerNorm,
}

impl AudioEncoderLayer {
    pub fn new(config: &AudioEncoderConfig) -> Result<Self> {
        let embed_dim = config.d_model;
        Ok(Self {
            self_attn: AudioAttention::new(config)?,
            self_attn_layer_norm: nn::LayerNormBuilder::new(embed_dim).build()?,
            fc1: nn::LinearBuilder::new(embed_dim, config.encoder_ffn_dim).bias(true).build()?,
            fc2: nn::LinearBuilder::new(config.encoder_ffn_dim, embed_dim).bias(true).build()?,
            final_layer_norm: nn::LayerNormBuilder::new(embed_dim).build()?,
        })
    }

    pub fn forward_layer(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
    ) -> std::result::Result<Array, mlx_rs::error::Exception> {
        let residual = x.clone();
        let h = self.self_attn_layer_norm.forward(x)?;
        let h = self.self_attn.forward_attn(&h, mask)?;
        let h = residual.add(&h)?;

        let residual = h.clone();
        let h = self.final_layer_norm.forward(&h)?;
        let h = nn::gelu(&self.fc1.forward(&h)?)?;
        let h = self.fc2.forward(&h)?;
        residual.add(&h)
    }
}

/// Qwen3-ASR Audio Encoder.
#[derive(Debug, Clone, ModuleParameters)]
pub struct AudioEncoder {
    #[param]
    pub conv2d1: nn::Conv2d,
    #[param]
    pub conv2d2: nn::Conv2d,
    #[param]
    pub conv2d3: nn::Conv2d,
    #[param]
    pub conv_out: nn::Linear,
    #[param]
    pub layers: Vec<AudioEncoderLayer>,
    #[param]
    pub ln_post: nn::LayerNorm,
    #[param]
    pub proj1: nn::Linear,
    #[param]
    pub proj2: nn::Linear,

    positional_embedding: SinusoidalPositionEmbedding,
    config: AudioEncoderConfig,
}

impl AudioEncoder {
    pub fn new(config: AudioEncoderConfig) -> Result<Self> {
        let embed_dim = config.d_model;
        let ds = config.downsample_hidden_size;

        // 3x Conv2d with stride 2, kernel 3, padding 1
        let conv2d1 = nn::Conv2dBuilder::new(1, ds, (3, 3))
            .stride((2, 2)).padding((1, 1)).build()?;
        let conv2d2 = nn::Conv2dBuilder::new(ds, ds, (3, 3))
            .stride((2, 2)).padding((1, 1)).build()?;
        let conv2d3 = nn::Conv2dBuilder::new(ds, ds, (3, 3))
            .stride((2, 2)).padding((1, 1)).build()?;

        // Frequency after 3x stride-2 convolutions
        let freq_after_conv = ((((config.num_mel_bins + 1) / 2) + 1) / 2 + 1) / 2;
        let conv_out = nn::LinearBuilder::new(ds * freq_after_conv, embed_dim)
            .bias(false).build()?;

        let positional_embedding = SinusoidalPositionEmbedding::new(
            config.max_source_positions,
            embed_dim,
        )?;

        let layers: Result<Vec<_>> = (0..config.encoder_layers)
            .map(|_| AudioEncoderLayer::new(&config))
            .collect();

        let ln_post = nn::LayerNormBuilder::new(embed_dim).build()?;
        let proj1 = nn::LinearBuilder::new(embed_dim, embed_dim).bias(true).build()?;
        let proj2 = nn::LinearBuilder::new(embed_dim, config.output_dim).bias(true).build()?;

        Ok(Self {
            conv2d1,
            conv2d2,
            conv2d3,
            conv_out,
            layers: layers?,
            ln_post,
            proj1,
            proj2,
            positional_embedding,
            config,
        })
    }

    /// Encode mel spectrogram to audio features.
    ///
    /// Input: mel spectrogram [n_mels, n_frames]
    /// Output: audio features [num_audio_tokens, output_dim]
    pub fn forward_encoder(
        &mut self,
        input_features: &Array,
    ) -> std::result::Result<Array, mlx_rs::error::Exception> {
        let feature_len = input_features.shape()[1];
        let chunk_size = self.config.n_window * 2;  // 100 frames per chunk

        // Split mel into chunks of 100 frames
        let num_chunks = (feature_len + chunk_size - 1) / chunk_size;

        let mut chunks = Vec::new();
        let mut chunk_lengths = Vec::new();

        for j in 0..num_chunks {
            let start = j * chunk_size;
            let clen = if j == num_chunks - 1 {
                let remainder = feature_len % chunk_size;
                if remainder == 0 { chunk_size } else { remainder }
            } else {
                chunk_size
            };

            let chunk = input_features.index((.., start..start + clen));
            chunks.push(chunk);
            chunk_lengths.push(clen);
        }

        // Pad chunks to max length
        let max_chunk_len = chunk_lengths.iter().copied().max().unwrap_or(chunk_size);
        let mut padded_chunks = Vec::new();
        for (i, chunk) in chunks.iter().enumerate() {
            let clen = chunk_lengths[i];
            if clen < max_chunk_len {
                let pad_width = max_chunk_len - clen;
                // Pad along second axis (time)
                let padded = mlx_rs::ops::pad(
                    chunk,
                    &[(0i32, 0i32), (0, pad_width)],
                    None::<Array>,
                    None::<mlx_rs::ops::PadMode>,
                )?;
                padded_chunks.push(padded);
            } else {
                padded_chunks.push(chunk.clone());
            }
        }

        // Stack to [num_chunks, n_mels, max_chunk_len]
        let padded_feature = mlx_rs::ops::stack_axis(&padded_chunks, 0)?;

        // Compute per-chunk output lengths after CNN
        let feature_lens_after_cnn: Vec<i32> = chunk_lengths
            .iter()
            .map(|&l| get_feat_extract_output_lengths(l))
            .collect();
        let max_len_after_cnn = *feature_lens_after_cnn.iter().max().unwrap();

        // Conv2d expects [B, H, W, C] (NHWC), input is [B, n_mels, time]
        // Reshape to [B, n_mels, time, 1]
        let b = padded_feature.shape()[0];
        let h = padded_feature.shape()[1];
        let w = padded_feature.shape()[2];
        let x = padded_feature.reshape(&[b, h, w, 1])?;

        let x = nn::gelu(&self.conv2d1.forward(&x)?)?;
        let x = nn::gelu(&self.conv2d2.forward(&x)?)?;
        let x = nn::gelu(&self.conv2d3.forward(&x)?)?;

        // x is [B, freq, time, channels]
        let shape = x.shape();
        let (b, _f, t, c) = (shape[0], shape[1], shape[2], shape[3]);
        // Transpose to [B, time, channels, freq] then reshape to [B, time, channels*freq]
        let f = shape[1];
        let x = x.transpose_axes(&[0, 2, 3, 1])?
            .reshape(&[b, t, c * f])?;

        // Project to d_model
        let x = self.conv_out.forward(&x)?;

        // Add sinusoidal position embeddings
        let seq_len = x.shape()[1];
        let pos_emb = self.positional_embedding.forward(seq_len)?;
        let x = x.add(&pos_emb)?;

        // Extract valid tokens (remove padding), concatenate all chunks
        let mut hidden_list = Vec::new();
        for i in 0..num_chunks as usize {
            let valid_len = feature_lens_after_cnn[i];
            let valid = x.index((i as i32, ..valid_len, ..));
            hidden_list.push(valid);
        }

        let hidden_states = mlx_rs::ops::concatenate_axis(&hidden_list, 0)?;

        // Compute total output length and build windowed attention mask
        let total_output_len = get_feat_extract_output_lengths(feature_len);
        let window_aftercnn = max_len_after_cnn * (self.config.n_window_infer / (self.config.n_window * 2));

        // Build block boundaries for windowed attention
        let mut boundaries = vec![0i32];
        let num_full_windows = total_output_len / window_aftercnn;
        for w in 1..=num_full_windows {
            boundaries.push(w * window_aftercnn);
        }
        let remainder = total_output_len % window_aftercnn;
        if remainder != 0 {
            boundaries.push(total_output_len);
        }

        // Create block attention mask
        let seq_len = hidden_states.shape()[0];
        let attention_mask = self.create_block_attention_mask(seq_len, &boundaries)?;

        // Add batch and head dims: [1, 1, seq_len, seq_len]
        let attention_mask = attention_mask.reshape(&[1, 1, seq_len, seq_len])?;

        // Add batch dim to hidden_states: [1, seq_len, d_model]
        let mut hidden_states = hidden_states.reshape(&[1, seq_len, self.config.d_model])?;

        // Run through transformer layers
        for layer in &mut self.layers {
            hidden_states = layer.forward_layer(&hidden_states, Some(&attention_mask))?;
        }

        // Remove batch dim
        let hidden_states = hidden_states.index((0, .., ..));

        // Post layer norm + projection
        let hidden_states = self.ln_post.forward(&hidden_states)?;
        let hidden_states = nn::gelu(&self.proj1.forward(&hidden_states)?)?;
        let hidden_states = self.proj2.forward(&hidden_states)?;

        Ok(hidden_states)
    }

    /// Create block attention mask for windowed attention.
    fn create_block_attention_mask(
        &self,
        seq_len: i32,
        boundaries: &[i32],
    ) -> std::result::Result<Array, mlx_rs::error::Exception> {
        let n = seq_len as usize;
        let mut mask_data = vec![-1e9f32; n * n];

        for w in 0..boundaries.len().saturating_sub(1) {
            let start = boundaries[w] as usize;
            let end = boundaries[w + 1] as usize;
            for i in start..end.min(n) {
                for j in start..end.min(n) {
                    mask_data[i * n + j] = 0.0;
                }
            }
        }

        Ok(Array::from_slice(&mask_data, &[seq_len, seq_len]))
    }
}
