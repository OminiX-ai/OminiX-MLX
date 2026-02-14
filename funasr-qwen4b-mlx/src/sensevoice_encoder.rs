//! SenseVoice encoder for FunASR-Qwen4B.
//!
//! This encoder uses SAN-M (Self-Attention with Memory) attention,
//! which combines multi-head self-attention with FSMN memory blocks.

use crate::error::{Error, Result};
use mlx_rs::builder::Builder;
use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::{Module, Param};
use mlx_rs::nn;
use mlx_rs::transforms::eval;
use mlx_rs::Array;
use std::collections::HashMap;
use std::path::Path;

/// SenseVoice encoder configuration.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct SenseVoiceEncoderConfig {
    /// Output dimension of encoder
    #[serde(default = "default_output_size")]
    pub output_size: i32,

    /// Number of attention heads
    #[serde(default = "default_attention_heads")]
    pub attention_heads: i32,

    /// FFN intermediate dimension
    #[serde(default = "default_linear_units")]
    pub linear_units: i32,

    /// Number of main encoder blocks
    #[serde(default = "default_num_blocks")]
    pub num_blocks: i32,

    /// Number of temporal-parallel encoder blocks
    #[serde(default = "default_tp_blocks")]
    pub tp_blocks: i32,

    /// FSMN kernel size
    #[serde(default = "default_kernel_size")]
    pub kernel_size: i32,

    /// Dropout rate
    #[serde(default)]
    pub dropout_rate: f32,

    /// LFR input dimension (n_mels * lfr_m)
    #[serde(default = "default_lfr_dim")]
    pub lfr_dim: i32,
}

fn default_output_size() -> i32 { 512 }
fn default_attention_heads() -> i32 { 4 }
fn default_linear_units() -> i32 { 2048 }
fn default_num_blocks() -> i32 { 50 }
fn default_tp_blocks() -> i32 { 20 }
fn default_kernel_size() -> i32 { 11 }
fn default_lfr_dim() -> i32 { 560 }

impl Default for SenseVoiceEncoderConfig {
    fn default() -> Self {
        Self {
            output_size: 512,
            attention_heads: 4,
            linear_units: 2048,
            num_blocks: 50,
            tp_blocks: 20,
            kernel_size: 11,
            dropout_rate: 0.0,
            lfr_dim: 560,
        }
    }
}

/// FSMN (Feedforward Sequential Memory Network) block.
/// Implements depthwise 1D convolution for sequential memory.
///
/// The official FunASR implementation uses symmetric padding with sanm_shfit=0:
/// - left_padding = (kernel_size - 1) // 2 = 5 for kernel_size=11
/// - right_padding = kernel_size - 1 - left_padding = 5
///
/// The FSMN block applies: output = conv(pad(x)) + x (residual connection inside)
#[derive(Debug, Clone, ModuleParameters)]
pub struct FSMNBlock {
    /// Depthwise convolution weight: [dim, 1, kernel_size]
    #[param]
    pub weight: Param<Array>,

    pub kernel_size: i32,
    pub dim: i32,
    pub left_padding: i32,
    pub right_padding: i32,
}

impl FSMNBlock {
    pub fn new(dim: i32, kernel_size: i32) -> Result<Self> {
        Self::new_with_shift(dim, kernel_size, 0)
    }

    pub fn new_with_shift(dim: i32, kernel_size: i32, sanm_shift: i32) -> Result<Self> {
        // Initialize weight: [dim, 1, kernel_size]
        let weight = mlx_rs::ops::zeros::<f32>(&[dim, 1, kernel_size])?;

        // Calculate symmetric padding as in official FunASR
        let mut left_padding = (kernel_size - 1) / 2;
        if sanm_shift > 0 {
            left_padding += sanm_shift;
        }
        let right_padding = kernel_size - 1 - left_padding;

        Ok(Self {
            weight: Param::new(weight),
            kernel_size,
            dim,
            left_padding,
            right_padding,
        })
    }

    /// Apply FSMN memory block using depthwise conv1d.
    /// Input: [batch, seq_len, dim]
    /// Output: [batch, seq_len, dim] = conv(pad(x)) + x
    ///
    /// This follows the official FunASR forward_fsmn implementation:
    /// 1. Transpose input to (B, D, T)
    /// 2. Apply symmetric padding
    /// 3. Apply depthwise conv1d
    /// 4. Transpose back to (B, T, D)
    /// 5. Add residual: output = conv_out + input
    pub fn forward(&self, x: &Array) -> std::result::Result<Array, mlx_rs::error::Exception> {
        let shape = x.shape();
        let batch = shape[0];
        let _seq_len = shape[1];
        let dim = shape[2];

        // Transpose to (B, D, T) for conv1d
        let x_t = x.transpose_axes(&[0, 2, 1])?;

        // Apply symmetric padding: (left_padding, right_padding) on the time axis
        // x_t shape is [batch, dim, seq_len]
        let left_pad = mlx_rs::ops::zeros_dtype(&[batch, dim, self.left_padding], x.dtype())?;
        let right_pad = mlx_rs::ops::zeros_dtype(&[batch, dim, self.right_padding], x.dtype())?;
        let x_padded = mlx_rs::ops::concatenate_axis(&[&left_pad, &x_t, &right_pad], 2)?;

        // Transpose back to (B, T_padded, D) for MLX conv1d
        let x_padded = x_padded.transpose_axes(&[0, 2, 1])?;

        // Reshape weight from [dim, 1, kernel_size] to [dim, kernel_size, 1] for MLX conv1d
        let weight_reshaped = self.weight.as_ref().transpose_axes(&[0, 2, 1])?;

        // Apply depthwise conv1d with groups=dim
        let conv_out = mlx_rs::ops::conv1d(&x_padded, &weight_reshaped, 1, 0, 1, self.dim)?;

        // Add residual connection: conv_out + x
        // This is the key difference from my previous implementation
        conv_out.add(x)
    }
}

/// SAN-M (Self-Attention with Memory) attention.
/// Combines multi-head self-attention with FSMN memory blocks.
///
/// The official FunASR implementation flow:
/// 1. QKV projection: q, k, v_h, v = forward_qkv(x)
/// 2. FSMN memory: fsmn_memory = forward_fsmn(v)  # FSMN includes residual internally
/// 3. Attention: att_outs = forward_attention(v_h, scores, mask)
/// 4. Combine: return att_outs + fsmn_memory
///
/// Key insight: FSMN output is added to attention output, NOT to V before attention!
#[derive(Debug, Clone, ModuleParameters)]
pub struct SANMAttention {
    /// Fused QKV projection
    #[param]
    pub linear_q_k_v: nn::Linear,

    /// FSMN memory block
    #[param]
    pub fsmn: FSMNBlock,

    /// Output projection
    #[param]
    pub linear_out: nn::Linear,

    pub n_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
}

impl SANMAttention {
    pub fn new(dim: i32, n_heads: i32, input_dim: Option<i32>, kernel_size: i32) -> Result<Self> {
        let input_dim = input_dim.unwrap_or(dim);
        let head_dim = dim / n_heads;

        Ok(Self {
            linear_q_k_v: nn::LinearBuilder::new(input_dim, dim * 3)
                .bias(true)
                .build()?,
            fsmn: FSMNBlock::new(dim, kernel_size)?,
            linear_out: nn::LinearBuilder::new(dim, dim)
                .bias(true)
                .build()?,
            n_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    pub fn forward(&mut self, x: &Array, _mask: Option<&Array>) -> std::result::Result<Array, mlx_rs::error::Exception> {
        let shape = x.shape();
        let (batch, seq_len, _) = (shape[0], shape[1], shape[2]);

        // QKV projection
        let qkv = self.linear_q_k_v.forward(x)?;
        let qkv_parts = mlx_rs::ops::split(&qkv, 3, -1)?;
        let (q, k, v) = (&qkv_parts[0], &qkv_parts[1], &qkv_parts[2]);

        // Apply FSMN to raw V (before reshaping for attention)
        // FSMN internally adds residual: fsmn_memory = conv(v) + v
        let fsmn_memory = self.fsmn.forward(v)?;

        // Reshape Q, K, V for multi-head attention
        // Note: We use v (not fsmn_memory) for attention, following official implementation
        let q = q.reshape(&[batch, seq_len, self.n_heads, self.head_dim])?;
        let k = k.reshape(&[batch, seq_len, self.n_heads, self.head_dim])?;
        let v_h = v.reshape(&[batch, seq_len, self.n_heads, self.head_dim])?;

        // Transpose to [batch, heads, seq, head_dim]
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v_h = v_h.transpose_axes(&[0, 2, 1, 3])?;

        // Use optimized SDPA (scaled dot-product attention)
        let attn_out = mlx_rs::fast::scaled_dot_product_attention(
            &q, &k, &v_h, self.scale, None::<mlx_rs::fast::ScaledDotProductAttentionMask>
        )?;

        // Reshape back to [batch, seq, dim]
        let attn_out = attn_out
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch, seq_len, self.n_heads * self.head_dim])?;

        // Output projection
        let att_outs = self.linear_out.forward(&attn_out)?;

        // CRITICAL: Add FSMN memory to attention output (not to V before attention!)
        att_outs.add(&fsmn_memory)
    }
}

/// Sinusoidal Position Encoder.
/// Adds sinusoidal position encoding to input features.
/// This follows the FunASR SinusoidalPositionEncoder implementation.
#[derive(Debug, Clone)]
pub struct SinusoidalPositionEncoder;

impl SinusoidalPositionEncoder {
    /// Encode positions using sinusoidal functions.
    /// positions: [1, timesteps] - position indices starting from 1
    /// depth: feature dimension
    pub fn encode(positions: &Array, depth: i32, dtype: mlx_rs::Dtype) -> std::result::Result<Array, mlx_rs::error::Exception> {
        // log_timescale_increment = log(10000) / (depth/2 - 1)
        let half_depth = depth / 2;
        let log_10000 = 10000.0_f32.ln();
        let log_timescale_increment = log_10000 / (half_depth as f32 - 1.0);

        // inv_timescales = exp(arange(depth/2) * -log_timescale_increment)
        let indices = mlx_rs::ops::arange::<_, f32>(0, half_depth, 1)?;
        let inv_timescales = mlx_rs::ops::exp(
            &indices.multiply(&Array::from(-log_timescale_increment))?
        )?;

        // positions shape: [1, timesteps]
        // inv_timescales shape: [half_depth]
        // scaled_time = positions[:, :, None] * inv_timescales[None, None, :]
        // Result shape: [1, timesteps, half_depth]
        let positions_expanded = positions.expand_dims(-1)?; // [1, timesteps, 1]
        let inv_timescales_expanded = inv_timescales.reshape(&[1, 1, half_depth])?; // [1, 1, half_depth]
        let scaled_time = positions_expanded.multiply(&inv_timescales_expanded)?;

        // encoding = cat([sin(scaled_time), cos(scaled_time)], dim=2)
        let sin_enc = mlx_rs::ops::sin(&scaled_time)?;
        let cos_enc = mlx_rs::ops::cos(&scaled_time)?;
        let encoding = mlx_rs::ops::concatenate_axis(&[&sin_enc, &cos_enc], 2)?;

        // Cast to target dtype
        encoding.as_dtype(dtype)
    }

    /// Apply position encoding to input.
    /// x: [batch, timesteps, input_dim]
    pub fn forward(x: &Array) -> std::result::Result<Array, mlx_rs::error::Exception> {
        let shape = x.shape();
        let timesteps = shape[1];
        let input_dim = shape[2];

        // positions = arange(1, timesteps + 1)[None, :]
        // Note: positions start from 1, not 0!
        let positions = mlx_rs::ops::arange::<_, i32>(1, timesteps + 1, 1)?;
        let positions = positions.reshape(&[1, timesteps])?;
        let positions = positions.as_dtype(x.dtype())?;

        // Get position encoding
        let pos_encoding = Self::encode(&positions, input_dim, x.dtype())?;

        // Add to input
        x.add(&pos_encoding)
    }
}

/// Feed-forward network for encoder.
#[derive(Debug, Clone, ModuleParameters)]
pub struct EncoderFFN {
    #[param]
    pub w_1: nn::Linear,
    #[param]
    pub w_2: nn::Linear,
}

impl EncoderFFN {
    pub fn new(dim: i32, hidden_dim: i32) -> Result<Self> {
        Ok(Self {
            w_1: nn::LinearBuilder::new(dim, hidden_dim).bias(true).build()?,
            w_2: nn::LinearBuilder::new(hidden_dim, dim).bias(true).build()?,
        })
    }
}

impl Module<&Array> for EncoderFFN {
    type Output = Array;
    type Error = mlx_rs::error::Exception;

    fn training_mode(&mut self, _mode: bool) {}

    fn forward(&mut self, x: &Array) -> std::result::Result<Array, Self::Error> {
        let h = self.w_1.forward(x)?;
        let h = nn::relu(&h)?;
        self.w_2.forward(&h)
    }
}

/// SenseVoice encoder layer.
#[derive(Debug, Clone, ModuleParameters)]
pub struct SenseVoiceEncoderLayer {
    #[param]
    pub self_attn: SANMAttention,
    #[param]
    pub feed_forward: EncoderFFN,
    #[param]
    pub norm1: nn::LayerNorm,
    #[param]
    pub norm2: nn::LayerNorm,

    /// Whether this is the first layer with different input dim
    pub is_first_layer: bool,
}

impl SenseVoiceEncoderLayer {
    pub fn new(dim: i32, n_heads: i32, ffn_dim: i32, kernel_size: i32, input_dim: Option<i32>) -> Result<Self> {
        let norm_dim = input_dim.unwrap_or(dim);
        let is_first = input_dim.is_some() && input_dim.unwrap() != dim;

        Ok(Self {
            self_attn: SANMAttention::new(dim, n_heads, input_dim, kernel_size)?,
            feed_forward: EncoderFFN::new(dim, ffn_dim)?,
            norm1: nn::LayerNormBuilder::new(norm_dim).build()?,
            norm2: nn::LayerNormBuilder::new(dim).build()?,
            is_first_layer: is_first,
        })
    }

    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> std::result::Result<Array, mlx_rs::error::Exception> {
        // Pre-norm architecture
        let h = self.norm1.forward(x)?;
        let h = self.self_attn.forward(&h, mask)?;

        // Skip connection only if dimensions match
        let h = if self.is_first_layer {
            h
        } else {
            x.add(&h)?
        };

        let residual = h.clone();
        let h = self.norm2.forward(&h)?;
        let h = self.feed_forward.forward(&h)?;
        residual.add(&h)
    }
}

/// SenseVoice encoder.
#[derive(Debug, Clone, ModuleParameters)]
pub struct SenseVoiceEncoder {
    /// Initial encoder layer (handles LFR input dimension)
    #[param]
    pub encoders0: Vec<SenseVoiceEncoderLayer>,

    /// Main encoder layers
    #[param]
    pub encoders: Vec<SenseVoiceEncoderLayer>,

    /// Temporal-parallel encoder layers
    #[param]
    pub tp_encoders: Vec<SenseVoiceEncoderLayer>,

    /// Final normalization after main encoders
    #[param]
    pub after_norm: nn::LayerNorm,

    /// Final normalization after tp_encoders
    #[param]
    pub tp_norm: nn::LayerNorm,

    pub config: SenseVoiceEncoderConfig,
}

impl SenseVoiceEncoder {
    pub fn new(config: SenseVoiceEncoderConfig) -> Result<Self> {
        let dim = config.output_size;
        let n_heads = config.attention_heads;
        let ffn_dim = config.linear_units;
        let kernel_size = config.kernel_size;

        // encoders0: 1 layer with input_dim=lfr_dim (560)
        let encoders0 = vec![
            SenseVoiceEncoderLayer::new(dim, n_heads, ffn_dim, kernel_size, Some(config.lfr_dim))?
        ];

        // encoders: num_blocks - 1 layers
        let num_main_layers = (config.num_blocks - 1) as usize;
        let encoders: Result<Vec<_>> = (0..num_main_layers)
            .map(|_| SenseVoiceEncoderLayer::new(dim, n_heads, ffn_dim, kernel_size, None))
            .collect();

        // tp_encoders: tp_blocks layers
        let num_tp_layers = config.tp_blocks as usize;
        let tp_encoders: Result<Vec<_>> = (0..num_tp_layers)
            .map(|_| SenseVoiceEncoderLayer::new(dim, n_heads, ffn_dim, kernel_size, None))
            .collect();

        Ok(Self {
            encoders0,
            encoders: encoders?,
            tp_encoders: tp_encoders?,
            after_norm: nn::LayerNormBuilder::new(dim).build()?,
            tp_norm: nn::LayerNormBuilder::new(dim).build()?,
            config,
        })
    }

    /// Get output dimension.
    pub fn output_dim(&self) -> i32 {
        self.config.output_size
    }

    /// Forward pass.
    pub fn forward(&mut self, x: &Array) -> std::result::Result<Array, mlx_rs::error::Exception> {
        // Apply sinusoidal position encoding BEFORE encoder layers
        // This follows the official FunASR implementation with input_layer="pe"
        let mut h = SinusoidalPositionEncoder::forward(x)?;

        // encoders0 (1 layer, handles 560->512 dimension change)
        for layer in &mut self.encoders0 {
            h = layer.forward(&h, None)?;
        }

        // Main encoders (49 layers)
        for layer in &mut self.encoders {
            h = layer.forward(&h, None)?;
        }

        // After main encoder normalization
        h = self.after_norm.forward(&h)?;

        // Temporal-parallel encoders (20 layers)
        for layer in &mut self.tp_encoders {
            h = layer.forward(&h, None)?;
        }

        // Final normalization
        self.tp_norm.forward(&h)
    }

    /// Load weights from safetensors file.
    ///
    /// Expected weight key format (from funasr-nano model.safetensors):
    /// - encoder.encoders0.0.attn.qkv.weight/bias
    /// - encoder.encoders0.0.attn.fsmn.weight
    /// - encoder.encoders0.0.attn.out.weight/bias
    /// - encoder.encoders0.0.ffn.w1.weight/bias
    /// - encoder.encoders0.0.ffn.w2.weight/bias
    /// - encoder.encoders0.0.norm1.weight/bias
    /// - encoder.encoders0.0.norm2.weight/bias
    /// - encoder.encoders.{i}.* (same structure)
    /// - encoder.tp_encoders.{i}.* (same structure)
    /// - encoder.after_norm.weight/bias
    /// - encoder.tp_norm.weight/bias
    pub fn load_weights(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(Error::ModelLoad(format!(
                "SenseVoice weights not found: {}",
                path.display()
            )));
        }

        eprintln!("Loading SenseVoice encoder weights from {}...", path.display());

        // Load safetensors
        let weights = Array::load_safetensors(path)
            .map_err(|e| Error::ModelLoad(format!("Failed to load safetensors: {:?}", e)))?;

        // Convert to HashMap for easier lookup
        let weights: HashMap<String, Array> = weights.into_iter().collect();

        let mut loaded_count = 0;

        // Load encoders0 (first layer, handles 560->512)
        for (i, layer) in self.encoders0.iter_mut().enumerate() {
            let prefix = format!("encoder.encoders0.{}", i);
            Self::load_encoder_layer(&weights, &prefix, layer, &mut loaded_count)?;
        }

        // Load main encoders (49 layers)
        for (i, layer) in self.encoders.iter_mut().enumerate() {
            let prefix = format!("encoder.encoders.{}", i);
            Self::load_encoder_layer(&weights, &prefix, layer, &mut loaded_count)?;
        }

        // Load tp_encoders (20 layers)
        for (i, layer) in self.tp_encoders.iter_mut().enumerate() {
            let prefix = format!("encoder.tp_encoders.{}", i);
            Self::load_encoder_layer(&weights, &prefix, layer, &mut loaded_count)?;
        }

        // Load after_norm
        if let Some(w) = weights.get("encoder.after_norm.weight") {
            *self.after_norm.weight = Some(Self::to_f32(w));
            loaded_count += 1;
        }
        if let Some(w) = weights.get("encoder.after_norm.bias") {
            *self.after_norm.bias = Some(Self::to_f32(w));
            loaded_count += 1;
        }

        // Load tp_norm
        if let Some(w) = weights.get("encoder.tp_norm.weight") {
            *self.tp_norm.weight = Some(Self::to_f32(w));
            loaded_count += 1;
        }
        if let Some(w) = weights.get("encoder.tp_norm.bias") {
            *self.tp_norm.bias = Some(Self::to_f32(w));
            loaded_count += 1;
        }

        eprintln!("Loaded {} SenseVoice encoder parameters", loaded_count);

        // Evaluate all loaded weights
        let all_weights: Vec<&Array> = weights.values().collect();
        eval(all_weights.iter().copied())?;

        Ok(())
    }

    /// Convert weight to float32 for numerical precision
    fn to_f32(w: &Array) -> Array {
        if w.dtype() == mlx_rs::Dtype::Float32 {
            w.clone()
        } else {
            w.as_dtype(mlx_rs::Dtype::Float32).unwrap_or_else(|_| w.clone())
        }
    }

    /// Load weights for a single encoder layer.
    fn load_encoder_layer(
        weights: &HashMap<String, Array>,
        prefix: &str,
        layer: &mut SenseVoiceEncoderLayer,
        loaded_count: &mut usize,
    ) -> Result<()> {
        // Attention QKV
        if let Some(w) = weights.get(&format!("{}.attn.qkv.weight", prefix)) {
            *layer.self_attn.linear_q_k_v.weight = Self::to_f32(w);
            *loaded_count += 1;
        }
        if let Some(w) = weights.get(&format!("{}.attn.qkv.bias", prefix)) {
            *layer.self_attn.linear_q_k_v.bias = Some(Self::to_f32(w));
            *loaded_count += 1;
        }

        // FSMN weight
        if let Some(w) = weights.get(&format!("{}.attn.fsmn.weight", prefix)) {
            *layer.self_attn.fsmn.weight = Self::to_f32(w);
            *loaded_count += 1;
        }

        // Attention output projection
        if let Some(w) = weights.get(&format!("{}.attn.out.weight", prefix)) {
            *layer.self_attn.linear_out.weight = Self::to_f32(w);
            *loaded_count += 1;
        }
        if let Some(w) = weights.get(&format!("{}.attn.out.bias", prefix)) {
            *layer.self_attn.linear_out.bias = Some(Self::to_f32(w));
            *loaded_count += 1;
        }

        // FFN w1
        if let Some(w) = weights.get(&format!("{}.ffn.w1.weight", prefix)) {
            *layer.feed_forward.w_1.weight = Self::to_f32(w);
            *loaded_count += 1;
        }
        if let Some(w) = weights.get(&format!("{}.ffn.w1.bias", prefix)) {
            *layer.feed_forward.w_1.bias = Some(Self::to_f32(w));
            *loaded_count += 1;
        }

        // FFN w2
        if let Some(w) = weights.get(&format!("{}.ffn.w2.weight", prefix)) {
            *layer.feed_forward.w_2.weight = Self::to_f32(w);
            *loaded_count += 1;
        }
        if let Some(w) = weights.get(&format!("{}.ffn.w2.bias", prefix)) {
            *layer.feed_forward.w_2.bias = Some(Self::to_f32(w));
            *loaded_count += 1;
        }

        // Norm1
        if let Some(w) = weights.get(&format!("{}.norm1.weight", prefix)) {
            *layer.norm1.weight = Some(Self::to_f32(w));
            *loaded_count += 1;
        }
        if let Some(w) = weights.get(&format!("{}.norm1.bias", prefix)) {
            *layer.norm1.bias = Some(Self::to_f32(w));
            *loaded_count += 1;
        }

        // Norm2
        if let Some(w) = weights.get(&format!("{}.norm2.weight", prefix)) {
            *layer.norm2.weight = Some(Self::to_f32(w));
            *loaded_count += 1;
        }
        if let Some(w) = weights.get(&format!("{}.norm2.bias", prefix)) {
            *layer.norm2.bias = Some(Self::to_f32(w));
            *loaded_count += 1;
        }

        Ok(())
    }
}
