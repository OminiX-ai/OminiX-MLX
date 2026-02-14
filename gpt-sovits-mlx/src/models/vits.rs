//! VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)
//!
//! This implements the SynthesizerTrn model from GPT-SoVITS for vocoding.
//!
//! Key components:
//! - ResidualVectorQuantizer: Decodes semantic codes to continuous representations
//! - TextEncoder (enc_p): Combines SSL features with text features via MRTE
//! - ResidualCouplingBlock (flow): Normalizing flow for latent transformation
//! - Generator (dec): HiFiGAN-style decoder for audio synthesis
//! - MelStyleEncoder (ref_enc): Extracts style embedding from reference mel

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::{
    array,
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::{
        concatenate_axis, exp, expand_dims, indexing::IndexOp, matmul, maximum, minimum,
        softmax_axis, split, sqrt, swap_axes, tanh, zeros_like,
    },
    random,
    Array,
};

/// Linear interpolation along the last axis (1D)
///
/// Equivalent to PyTorch's F.interpolate(x, size=new_size, mode="linear")
///
/// Args:
/// - x: Input tensor [batch, channels, seq_len]
/// - new_size: Target sequence length
///
/// Returns tensor [batch, channels, new_size]
fn interpolate_linear(x: &Array, new_size: i32) -> Result<Array, Exception> {
    let shape = x.shape();
    let src_len = shape[2] as i32;

    if new_size == src_len {
        return Ok(x.clone());
    }

    if new_size <= 0 {
        return Err(Exception::from("new_size must be positive"));
    }

    // For each output position, compute the weighted average of two neighboring input positions
    // scale = (src_len - 1) / (new_size - 1) for endpoint-aligned interpolation
    // But PyTorch's "linear" mode uses: scale = src_len / new_size (area-based)
    let scale = src_len as f32 / new_size as f32;

    // Build indices and weights for linear interpolation
    let mut left_indices = Vec::with_capacity(new_size as usize);
    let mut right_indices = Vec::with_capacity(new_size as usize);
    let mut weights = Vec::with_capacity(new_size as usize);

    for i in 0..new_size {
        // Source position (align to center of output bin)
        let src_pos = (i as f32 + 0.5) * scale - 0.5;
        let src_pos = src_pos.max(0.0).min((src_len - 1) as f32);

        let left = src_pos.floor() as i32;
        let right = (left + 1).min(src_len - 1);
        let weight = src_pos - left as f32;

        left_indices.push(left);
        right_indices.push(right);
        weights.push(weight);
    }

    // Gather left and right values
    // x shape: [batch, channels, seq_len]
    // We need to index along the last axis

    // Convert to arrays for indexing
    let left_idx = Array::from_slice(&left_indices, &[new_size]);
    let right_idx = Array::from_slice(&right_indices, &[new_size]);
    let weights_arr = Array::from_slice(&weights, &[1, 1, new_size]);

    // Use take along last axis
    // Transpose to [seq_len, batch, channels]
    let x_t = x.transpose_axes(&[2, 0, 1])?; // [batch, channels, seq_len] -> [seq_len, batch, channels]

    let left_vals = x_t.index((&left_idx, .., ..));   // [new_size, batch, channels]
    let right_vals = x_t.index((&right_idx, .., ..)); // [new_size, batch, channels]

    // Transpose back to [batch, channels, new_size]
    let left_vals = left_vals.transpose_axes(&[1, 2, 0])?;  // [new_size, batch, channels] -> [batch, channels, new_size]
    let right_vals = right_vals.transpose_axes(&[1, 2, 0])?;

    // Linear interpolation: left * (1 - weight) + right * weight
    let one_minus_w = array!(1.0f32).subtract(&weights_arr)?;
    let result = left_vals.multiply(&one_minus_w)?.add(&right_vals.multiply(&weights_arr)?)?;

    Ok(result)
}

use crate::error::Error;
use crate::nn::{WeightNormConv1d, WeightNormConvTranspose1d};

/// Configuration for VITS model
#[derive(Debug, Clone)]
pub struct VITSConfig {
    /// Hidden channels (192 in GPT-SoVITS)
    pub hidden_channels: i32,
    /// SSL feature dimension (768 from CNHubert)
    pub ssl_dim: i32,
    /// Number of attention heads
    pub n_heads: i32,
    /// Number of encoder layers
    pub n_layers: i32,
    /// Filter channels in FFN
    pub filter_channels: i32,
    /// Kernel size in encoder
    pub kernel_size: i32,
    /// Number of flow layers
    pub n_flows: i32,
    /// Gin channels (style conditioning)
    pub gin_channels: i32,
    /// Text vocabulary size
    pub vocab_size: i32,
    /// Codebook size
    pub codebook_size: i32,
    /// Codebook dimension
    pub codebook_dim: i32,
    /// Upsample rates
    pub upsample_rates: Vec<i32>,
    /// Upsample kernel sizes
    pub upsample_kernel_sizes: Vec<i32>,
    /// Upsample initial channel
    pub upsample_initial_channel: i32,
    /// ResBlock kernel sizes
    pub resblock_kernel_sizes: Vec<i32>,
    /// ResBlock dilation sizes
    pub resblock_dilation_sizes: Vec<Vec<i32>>,
    /// Semantic frame rate: "25hz" (2x downsample in ssl_proj) or "50hz" (no downsample)
    /// Models trained with s1bert25hz-* use "25hz", older models use "50hz"
    pub semantic_frame_rate: String,
    /// Segment size for training (in frames, not samples)
    /// Default 32 frames = 20480 samples at hop_length=640
    pub segment_size: i32,
}

impl Default for VITSConfig {
    fn default() -> Self {
        Self {
            hidden_channels: 192,
            ssl_dim: 768,
            n_heads: 2,
            n_layers: 6,
            filter_channels: 768,
            kernel_size: 3,
            n_flows: 4,
            gin_channels: 512,
            vocab_size: 732,
            codebook_size: 1024,
            codebook_dim: 768,
            upsample_rates: vec![10, 8, 2, 2, 2],
            upsample_kernel_sizes: vec![16, 16, 8, 2, 2],
            upsample_initial_channel: 512,
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            // Default to "50hz" for backwards compatibility with older models (luoxiang, mayun)
            // Set to "25hz" for models trained with s1bert25hz-* (doubao-mixed)
            semantic_frame_rate: "50hz".to_string(),
            // segment_size = 20480 samples / hop_length = 20480 / 640 = 32 frames
            segment_size: 32,
        }
    }
}

/// Randomly slice segments from a tensor for training.
/// Python: commons.rand_slice_segments(x, x_lengths, segment_size)
///
/// # Arguments
/// * `x` - Input tensor [batch, channels, time]
/// * `x_lengths` - Optional lengths per batch item (if None, uses full length)
/// * `segment_size` - Size of segment to slice
///
/// # Returns
/// * `(sliced, ids_slice)` - Sliced tensor and start indices
pub fn rand_slice_segments(
    x: &Array,
    x_lengths: Option<&Array>,
    segment_size: i32,
) -> Result<(Array, Array), Exception> {
    use mlx_rs::transforms::eval;

    let batch = x.dim(0) as i32;
    let time = x.dim(2) as i32;

    // Get max valid start index per batch item
    let ids_str_max = if let Some(lengths) = x_lengths {
        // lengths - segment_size + 1, but clamp to >= 1
        let seg_arr = Array::from_int(segment_size - 1);
        let max_idx = lengths.subtract(&seg_arr)?;
        // Clamp to at least 1 to avoid negative indices
        maximum(&max_idx, &Array::from_int(1))?
    } else {
        // Use full time length: create array filled with (time - segment_size + 1)
        let max_val = time - segment_size + 1;
        Array::from_iter((0..batch).map(|_| max_val), &[batch])
    };

    // Random start indices: (rand([batch]) * max_idx).to_int()
    // random::uniform needs: uniform::<LowType, HighType>(low, high, shape, key)
    let rand_vals = random::uniform::<f32, f32>(0.0, 1.0, &[batch], None)?;
    let ids_str = rand_vals.multiply(&ids_str_max.as_type::<f32>()?)?;
    let ids_str = ids_str.as_type::<i32>()?;

    // Need to evaluate to get actual values for indexing
    eval([&ids_str])?;

    // Slice each batch item - need to gather slices
    // For simplicity, we'll use a loop since MLX doesn't have built-in slice_segments
    let mut slices = Vec::with_capacity(batch as usize);
    for b in 0..batch {
        let start_idx: i32 = ids_str.index(b).item();
        let end_idx = start_idx + segment_size;
        let slice = x.index((b, .., start_idx..end_idx));
        slices.push(slice);
    }

    // Stack slices back to [batch, channels, segment_size]
    let slices_refs: Vec<&Array> = slices.iter().collect();
    let sliced = mlx_rs::ops::stack_axis(&slices_refs, 0)?;

    Ok((sliced, ids_str))
}

// ============================================================================
// Residual Vector Quantizer
// ============================================================================

/// RVQ Codebook for decoding semantic codes
#[derive(Debug, Clone, ModuleParameters)]
pub struct RVQCodebook {
    #[param]
    pub embed: Param<Array>,
    pub codebook_size: i32,
    pub codebook_dim: i32,
}

impl RVQCodebook {
    pub fn new(codebook_size: i32, codebook_dim: i32) -> Result<Self, Exception> {
        let embed = Array::zeros::<f32>(&[codebook_size, codebook_dim])?;
        Ok(Self {
            embed: Param::new(embed),
            codebook_size,
            codebook_dim,
        })
    }

    /// Decode indices to embeddings
    /// Input: codes [n_q, batch, seq] or [batch, n_q, seq]
    /// Output: quantized [batch, dim, seq]
    pub fn decode(&self, codes: &Array) -> Result<Array, Exception> {
        use mlx_rs::transforms::eval;

        // codes shape: [1, 1, seq] from GPT-SoVITS typically
        let shape = codes.shape();

        // Flatten to get indices
        let indices = codes.flatten(None, None)?;
        let indices = indices.as_type::<i32>()?;

        // Gather embeddings using take_axis (embedding lookup)
        // embed: [codebook_size, codebook_dim], indices: [seq]
        // result: [seq, codebook_dim]
        let quantized = self.embed.take_axis(&indices, 0)?;
        eval([&quantized])?; // Force evaluation to materialize

        // Reshape: if input was [1, 1, seq], output should be [1, dim, seq]
        if shape.len() == 3 {
            let seq_len = shape[2] as i32;
            // quantized is [seq, dim] - we need [1, dim, seq]
            // First add batch dim: [1, seq, dim]
            let batched = quantized.reshape(&[1, seq_len, self.codebook_dim])?;
            // Then transpose last two dims: [1, seq, dim] -> [1, dim, seq]
            // Use transpose_axes for explicit permutation
            batched.transpose_axes(&[0, 2, 1])
        } else {
            Ok(quantized)
        }
    }

    /// Encode features to codebook indices (for few-shot mode)
    /// Input: features [batch, dim, seq]
    /// Output: codes [batch, 1, seq]
    ///
    /// Uses cosine similarity (L2-normalized dot product) to match
    /// Python GPT-SoVITS VectorQuantize(use_cosine_sim=True).
    pub fn encode(&self, features: &Array) -> Result<Array, Exception> {
        use mlx_rs::transforms::eval;
        use mlx_rs::ops::{sum_axis, maximum, indexing::argmax_axis_device};

        let shape = features.shape();
        let batch = shape[0] as i32;
        let dim = shape[1] as i32;
        let seq = shape[2] as i32;

        // Transpose features: [batch, dim, seq] -> [batch, seq, dim]
        let features_t = features.transpose_axes(&[0, 2, 1])?;
        // Reshape to [batch * seq, dim]
        let flat_features = features_t.reshape(&[batch * seq, dim])?;

        // L2-normalize features: feat / ||feat||
        let feat_norm = sum_axis(&flat_features.multiply(&flat_features)?, -1, true)?;
        let eps = array!(1e-8f32);
        let feat_norm = maximum(&feat_norm.sqrt()?, &eps)?;
        let flat_normed = flat_features.divide(&feat_norm)?;

        // L2-normalize codebook embeddings: embed / ||embed||
        let embed_norm = sum_axis(&self.embed.multiply(&self.embed)?, -1, true)?;
        let embed_norm = maximum(&embed_norm.sqrt()?, &eps)?;
        let embed_normed = self.embed.divide(&embed_norm)?;

        // Cosine similarity: [batch * seq, dim] @ [dim, codebook_size]
        let embed_t = embed_normed.transpose()?;
        let similarity = matmul(&flat_normed, &embed_t)?;

        eval([&similarity])?;

        // Find argmax (highest cosine similarity)
        let codes = argmax_axis_device(&similarity, -1, false, mlx_rs::StreamOrDevice::default())?;
        let codes = codes.as_type::<i32>()?;

        // Reshape to [batch, 1, seq]
        codes.reshape(&[batch, 1, seq])
    }

    /// Forward pass with commitment loss (for training).
    ///
    /// This performs VQ quantization with straight-through estimator:
    /// - Forward: returns quantized embeddings
    /// - Backward: gradients flow through input features (not through codebook lookup)
    ///
    /// Python equivalent:
    /// ```python
    /// quantized, codes, commit_loss, _ = self.quantizer(ssl, layers=[0])
    /// ```
    ///
    /// Input: features [batch, dim, seq] (NCL format)
    /// Returns: (quantized_st, commit_loss)
    ///   - quantized_st: Quantized features with straight-through gradients [batch, dim, seq]
    ///   - commit_loss: VQ commitment loss (scalar)
    pub fn forward_with_loss(&self, features: &Array) -> Result<(Array, Array), Exception> {
        use mlx_rs::transforms::eval;
        use mlx_rs::ops::{sum_axis, indexing::argmin_axis};
        use mlx_rs::stop_gradient;

        let shape = features.shape();
        let batch = shape[0] as i32;
        let dim = shape[1] as i32;
        let seq = shape[2] as i32;

        // Step 1: Find nearest codebook entries (same as encode)
        let features_t = features.transpose_axes(&[0, 2, 1])?;
        let flat_features = features_t.reshape(&[batch * seq, dim])?;

        // Compute L2 distances: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a . b
        let features_sq = sum_axis(&flat_features.multiply(&flat_features)?, -1, true)?;
        let embed_sq = sum_axis(&self.embed.multiply(&self.embed)?, -1, true)?;
        let embed_sq_t = embed_sq.transpose()?;
        let embed_t = self.embed.transpose()?;
        let dot = matmul(&flat_features, &embed_t)?;
        let distances = features_sq
            .add(&embed_sq_t)?
            .subtract(&dot.multiply(array!(2.0f32))?)?;

        // Find nearest codebook entry
        let codes = argmin_axis(&distances, -1, false)?;
        let codes = codes.as_type::<i32>()?;

        // Step 2: Look up quantized embeddings
        let quantized_flat = self.embed.take_axis(&codes, 0)?;
        // quantized_flat: [batch * seq, dim]

        // Reshape back to [batch, seq, dim] then transpose to [batch, dim, seq]
        let quantized_nlc = quantized_flat.reshape(&[batch, seq, dim])?;
        let quantized = quantized_nlc.transpose_axes(&[0, 2, 1])?;

        eval([&quantized])?;

        // Step 3: Compute commitment loss
        // commit_loss = ||z - quantized||^2
        // Python uses F.mse_loss, which is mean((z - quantized)^2)
        let diff = features.subtract(&quantized)?;
        let commit_loss = diff.square()?.mean(false)?;

        // Step 4: Straight-through estimator
        // quantized_st = z + stop_gradient(quantized - z)
        // Forward: returns quantized
        // Backward: gradients flow through z (as if identity)
        let quantized_st = features.add(&stop_gradient(&quantized.subtract(features)?)?)?;

        Ok((quantized_st, commit_loss))
    }
}

// ============================================================================
// Attention Layer (for transformer encoder)
// ============================================================================

/// Multi-head attention with relative positional encoding
#[derive(Debug, Clone, ModuleParameters)]
pub struct RelativeAttention {
    #[param]
    pub conv_q: nn::Conv1d,
    #[param]
    pub conv_k: nn::Conv1d,
    #[param]
    pub conv_v: nn::Conv1d,
    #[param]
    pub conv_o: nn::Conv1d,
    #[param]
    pub emb_rel_k: Param<Array>,
    #[param]
    pub emb_rel_v: Param<Array>,
    pub n_heads: i32,
    pub head_dim: i32,
    pub window_size: i32,
}

impl RelativeAttention {
    pub fn new(channels: i32, n_heads: i32) -> Result<Self, Exception> {
        Self::new_with_window(channels, n_heads, 4) // default window_size=4
    }

    pub fn new_with_window(channels: i32, n_heads: i32, window_size: i32) -> Result<Self, Exception> {
        let head_dim = channels / n_heads;

        let conv_q = nn::Conv1dBuilder::new(channels, channels, 1).build()?;
        let conv_k = nn::Conv1dBuilder::new(channels, channels, 1).build()?;
        let conv_v = nn::Conv1dBuilder::new(channels, channels, 1).build()?;
        let conv_o = nn::Conv1dBuilder::new(channels, channels, 1).build()?;

        // Relative position embeddings: [1, window*2+1, head_dim]
        let emb_size = window_size * 2 + 1;
        let emb_rel_k = Array::zeros::<f32>(&[1, emb_size, head_dim])?;
        let emb_rel_v = Array::zeros::<f32>(&[1, emb_size, head_dim])?;

        Ok(Self {
            conv_q,
            conv_k,
            conv_v,
            conv_o,
            emb_rel_k: Param::new(emb_rel_k),
            emb_rel_v: Param::new(emb_rel_v),
            n_heads,
            head_dim,
            window_size,
        })
    }

    /// Get relative embeddings for the given sequence length
    #[allow(dead_code)]
    fn get_relative_embeddings(&self, rel_emb: &Array, length: i32) -> Result<Array, Exception> {
        let _max_rel_pos = 2 * self.window_size + 1;
        let pad_length = (length - (self.window_size + 1)).max(0);
        let slice_start = ((self.window_size + 1) - length).max(0);
        let slice_end = slice_start + 2 * length - 1;

        let padded = if pad_length > 0 {
            // Pad along the sequence dimension (dim 1)
            // rel_emb shape: [1, max_rel_pos, head_dim]
            let widths: &[(i32, i32)] = &[(0, 0), (pad_length, pad_length), (0, 0)];
            mlx_rs::ops::pad(rel_emb, widths, None, None)?
        } else {
            rel_emb.clone()
        };

        // Slice: padded[:, slice_start:slice_end, :]
        Ok(padded.index((.., slice_start..slice_end, ..)))
    }

    /// Matmul with relative keys: x[b,h,l,d] @ y[1,m,d].T -> [b,h,l,m]
    #[allow(dead_code)]
    fn matmul_with_relative_keys(&self, x: &Array, y: &Array) -> Result<Array, Exception> {
        // y shape: [1, m, d] -> [1, 1, m, d] -> transpose to [1, 1, d, m]
        let y_exp = y.index((mlx_rs::ops::indexing::NewAxis, .., .., ..));
        let y_t = swap_axes(&y_exp, 2, 3)?;
        matmul(x, &y_t)
    }

    /// Matmul with relative values: x[b,h,l,m] @ y[1,m,d] -> [b,h,l,d]
    #[allow(dead_code)]
    fn matmul_with_relative_values(&self, x: &Array, y: &Array) -> Result<Array, Exception> {
        // y shape: [1, m, d] -> [1, 1, m, d]
        let y_exp = y.index((mlx_rs::ops::indexing::NewAxis, .., .., ..));
        matmul(x, &y_exp)
    }

    /// Convert relative position to absolute position
    /// x: [b, h, l, 2*l-1] -> [b, h, l, l]
    #[allow(dead_code)]
    fn relative_position_to_absolute_position(&self, x: &Array) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch = shape[0] as i32;
        let heads = shape[1] as i32;
        let length = shape[2] as i32;

        // Pad along last dim: [b, h, l, 2*l-1] -> [b, h, l, 2*l]
        let widths: &[(i32, i32)] = &[(0, 0), (0, 0), (0, 0), (0, 1)];
        let x_padded = mlx_rs::ops::pad(x, widths, None, None)?;

        // Reshape to [b, h, l * 2 * l]
        let x_flat = x_padded.reshape(&[batch, heads, length * 2 * length])?;

        // Pad: [b, h, l*2*l] -> [b, h, l*2*l + l - 1]
        let widths: &[(i32, i32)] = &[(0, 0), (0, 0), (0, length - 1)];
        let x_flat = mlx_rs::ops::pad(&x_flat, widths, None, None)?;

        // Reshape to [b, h, l+1, 2*l-1]
        let x_reshaped = x_flat.reshape(&[batch, heads, length + 1, 2 * length - 1])?;

        // Slice: [:, :, :length, length-1:]
        Ok(x_reshaped.index((.., .., ..length, (length - 1)..)))
    }

    /// Convert absolute position to relative position
    /// x: [b, h, l, l] -> [b, h, l, 2*l-1]
    #[allow(dead_code)]
    fn absolute_position_to_relative_position(&self, x: &Array) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch = shape[0] as i32;
        let heads = shape[1] as i32;
        let length = shape[2] as i32;

        // Pad along last dim: [b, h, l, l] -> [b, h, l, 2*l-1]
        let widths: &[(i32, i32)] = &[(0, 0), (0, 0), (0, 0), (0, length - 1)];
        let x_padded = mlx_rs::ops::pad(x, widths, None, None)?;

        // Reshape to [b, h, l^2 + l*(l-1)]
        let flat_size = length * length + length * (length - 1);
        let x_flat = x_padded.reshape(&[batch, heads, flat_size])?;

        // Pad at beginning: [b, h, flat_size] -> [b, h, flat_size + length]
        let widths: &[(i32, i32)] = &[(0, 0), (0, 0), (length, 0)];
        let x_flat = mlx_rs::ops::pad(&x_flat, widths, None, None)?;

        // Reshape to [b, h, l, 2*l]
        let x_reshaped = x_flat.reshape(&[batch, heads, length, 2 * length])?;

        // Slice: [:, :, :, 1:]
        Ok(x_reshaped.index((.., .., .., 1..)))
    }

    /// Forward pass (expects NCL input, returns NCL output)
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch = shape[0] as i32;
        let channels = shape[1] as i32;
        let seq_len = shape[2] as i32;

        // Convert NCL to NLC for Conv1d (mlx-rs expects NLC)
        let x_nlc = swap_axes(x, 1, 2)?;

        // Q, K, V projections (input/output in NLC)
        let q = self.conv_q.forward(&x_nlc)?;
        let k = self.conv_k.forward(&x_nlc)?;
        let v = self.conv_v.forward(&x_nlc)?;

        // Convert NLC to NCL: [batch, seq, channels] -> [batch, channels, seq]
        let q = swap_axes(&q, 1, 2)?;
        let k = swap_axes(&k, 1, 2)?;
        let v = swap_axes(&v, 1, 2)?;

        // Reshape for multi-head: [batch, channels, seq] -> [batch, heads, head_dim, seq]
        let q = q.reshape(&[batch, self.n_heads, self.head_dim, seq_len])?;
        let k = k.reshape(&[batch, self.n_heads, self.head_dim, seq_len])?;
        let v = v.reshape(&[batch, self.n_heads, self.head_dim, seq_len])?;

        // Transpose for attention: [batch, heads, seq, head_dim]
        let q = swap_axes(&q, 2, 3)?;
        let k = swap_axes(&k, 2, 3)?;
        let v = swap_axes(&v, 2, 3)?;

        // Attention scores: [batch, heads, seq, seq]
        let scale = (self.head_dim as f32).sqrt();
        let q_scaled = q.divide(array!(scale))?;
        let scores = matmul(&q_scaled, &swap_axes(&k, 2, 3)?)?;

        // TODO: Re-enable relative position encoding after verifying baseline
        // Add relative position bias for keys
        // let rel_emb_k = self.get_relative_embeddings(&self.emb_rel_k, seq_len)?;
        // let rel_logits = self.matmul_with_relative_keys(&q_scaled, &rel_emb_k)?;
        // let scores_local = self.relative_position_to_absolute_position(&rel_logits)?;
        // let scores = scores.add(&scores_local)?;

        // Apply attention mask if provided
        // mask shape: [batch, 1, seq, seq] - positions with 0 are masked out
        let scores = if let Some(m) = mask {
            // scores.masked_fill(mask == 0, -1e4)
            let neg_inf = array!(-1e4f32);
            let zero = array!(0.0f32);
            let mask_zero = m.eq(&zero)?;
            mlx_rs::ops::r#where(&mask_zero, &neg_inf, &scores)?
        } else {
            scores
        };

        // Softmax
        let attn = softmax_axis(&scores, -1, false)?;

        // Apply to values: [batch, heads, seq, head_dim]
        let out = matmul(&attn, &v)?;

        // TODO: Re-enable relative position encoding for values
        // Add relative position bias for values
        // let rel_weights = self.absolute_position_to_relative_position(&attn)?;
        // let rel_emb_v = self.get_relative_embeddings(&self.emb_rel_v, seq_len)?;
        // let rel_values = self.matmul_with_relative_values(&rel_weights, &rel_emb_v)?;
        // out = out.add(&rel_values)?;

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, channels, seq]
        let out = swap_axes(&out, 2, 3)?;
        let out = out.reshape(&[batch, channels, seq_len])?;

        // Convert NCL to NLC for output projection
        let out_nlc = swap_axes(&out, 1, 2)?;
        let out_nlc = self.conv_o.forward(&out_nlc)?;

        // Convert back to NCL
        swap_axes(&out_nlc, 1, 2)
    }

    /// Cross-attention: Q from x, K/V from c (both NCL format)
    /// attn_mask shape: [batch, 1, q_len, kv_len] - positions with 0 are masked out
    pub fn cross_forward(&mut self, x: &Array, c: &Array, attn_mask: Option<&Array>) -> Result<Array, Exception> {
        let x_shape = x.shape();
        let c_shape = c.shape();
        let batch = x_shape[0] as i32;
        let channels = x_shape[1] as i32;
        let q_len = x_shape[2] as i32;  // SSL sequence length
        let kv_len = c_shape[2] as i32;  // Text sequence length

        // Convert NCL to NLC for Conv1d
        let x_nlc = swap_axes(x, 1, 2)?;
        let c_nlc = swap_axes(c, 1, 2)?;

        // Q from x (query), K/V from c (key/value)
        let q = self.conv_q.forward(&x_nlc)?;
        let k = self.conv_k.forward(&c_nlc)?;
        let v = self.conv_v.forward(&c_nlc)?;

        // Convert NLC to NCL
        let q = swap_axes(&q, 1, 2)?;
        let k = swap_axes(&k, 1, 2)?;
        let v = swap_axes(&v, 1, 2)?;

        // Reshape for multi-head
        let q = q.reshape(&[batch, self.n_heads, self.head_dim, q_len])?;
        let k = k.reshape(&[batch, self.n_heads, self.head_dim, kv_len])?;
        let v = v.reshape(&[batch, self.n_heads, self.head_dim, kv_len])?;

        // Transpose: [batch, heads, seq, head_dim]
        let q = swap_axes(&q, 2, 3)?;
        let k = swap_axes(&k, 2, 3)?;
        let v = swap_axes(&v, 2, 3)?;

        // Cross-attention scores: [batch, heads, q_len, kv_len]
        let scale = (self.head_dim as f32).sqrt();
        let scores = matmul(&q, &swap_axes(&k, 2, 3)?)?;
        let scores = scores.divide(array!(scale))?;

        // Apply attention mask: scores.masked_fill(mask == 0, -1e4)
        let scores = if let Some(mask) = attn_mask {
            // mask shape: [batch, 1, q_len, kv_len]
            // Create large negative value where mask == 0
            let neg_inf = array!(-1e4f32);
            let zero = array!(0.0f32);
            // where(mask == 0, -1e4, scores)
            let mask_bool = mask.eq(&zero)?;
            mlx_rs::ops::r#where(&mask_bool, &neg_inf, &scores)?
        } else {
            scores
        };

        // Softmax
        let attn = softmax_axis(&scores, -1, false)?;

        // Apply to values: [batch, heads, q_len, head_dim]
        let out = matmul(&attn, &v)?;

        // Reshape back: [batch, heads, q_len, head_dim] -> [batch, channels, q_len]
        let out = swap_axes(&out, 2, 3)?;
        let out = out.reshape(&[batch, channels, q_len])?;

        // Convert NCL to NLC for output projection
        let out_nlc = swap_axes(&out, 1, 2)?;
        let out_nlc = self.conv_o.forward(&out_nlc)?;

        // Convert back to NCL
        swap_axes(&out_nlc, 1, 2)
    }
}

// ============================================================================
// FFN Layer (Feed-Forward Network)
// ============================================================================

/// Feed-forward network with Conv1d
#[derive(Debug, Clone, ModuleParameters)]
pub struct FFN {
    #[param]
    pub conv_1: nn::Conv1d,
    #[param]
    pub conv_2: nn::Conv1d,
    pub kernel_size: i32,
}

impl FFN {
    pub fn new(
        in_channels: i32,
        out_channels: i32,
        filter_channels: i32,
        kernel_size: i32,
    ) -> Result<Self, Exception> {
        let padding = (kernel_size - 1) / 2;
        let conv_1 = nn::Conv1dBuilder::new(in_channels, filter_channels, kernel_size)
            .padding(padding)
            .build()?;
        let conv_2 = nn::Conv1dBuilder::new(filter_channels, out_channels, kernel_size)
            .padding(padding)
            .build()?;

        Ok(Self {
            conv_1,
            conv_2,
            kernel_size,
        })
    }

    /// Forward pass (expects NCL input, returns NCL output)
    pub fn forward(&mut self, x: &Array, mask: &Array) -> Result<Array, Exception> {
        // Convert NCL to NLC for Conv1d
        let x_nlc = swap_axes(x, 1, 2)?;
        let mask_nlc = swap_axes(mask, 1, 2)?;

        let x = self.conv_1.forward(&x_nlc)?;
        let x = nn::relu(&x)?;
        let x = x.multiply(&mask_nlc)?;
        let x = self.conv_2.forward(&x)?;
        let x = x.multiply(&mask_nlc)?;

        // Convert back to NCL
        swap_axes(&x, 1, 2)
    }
}

// ============================================================================
// Transformer Encoder
// ============================================================================

/// Layer normalization for conv inputs (channels-first)
#[derive(Debug, Clone, ModuleParameters)]
pub struct ConvLayerNorm {
    #[param]
    pub gamma: Param<Array>,
    #[param]
    pub beta: Param<Array>,
    pub channels: i32,
    pub eps: f32,
}

impl ConvLayerNorm {
    pub fn new(channels: i32) -> Result<Self, Exception> {
        let gamma = Array::ones::<f32>(&[channels])?;
        let beta = Array::zeros::<f32>(&[channels])?;
        Ok(Self {
            gamma: Param::new(gamma),
            beta: Param::new(beta),
            channels,
            eps: 1e-5,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        // x: [batch, channels, seq]
        // Transpose to [batch, seq, channels], normalize, transpose back
        let x = swap_axes(x, 1, 2)?;

        // Manual layer norm along last dimension
        let mean = x.mean_axis(-1, true)?;
        let x_centered = x.subtract(&mean)?;
        let var = x_centered.square()?.mean_axis(-1, true)?;
        let x_norm = x_centered.divide(&sqrt(&var.add(array!(self.eps))?)?)?;

        // Apply scale and bias
        // gamma and beta are [channels], need [1, 1, channels] for broadcasting
        let gamma = self.gamma.reshape(&[1, 1, self.channels])?;
        let beta = self.beta.reshape(&[1, 1, self.channels])?;
        let out = x_norm.multiply(&gamma)?.add(&beta)?;

        // Transpose back
        swap_axes(&out, 1, 2)
    }
}

/// Transformer encoder layer
#[derive(Debug, Clone, ModuleParameters)]
pub struct EncoderLayer {
    #[param]
    pub attn: RelativeAttention,
    #[param]
    pub ffn: FFN,
    #[param]
    pub norm1: ConvLayerNorm,
    #[param]
    pub norm2: ConvLayerNorm,
}

impl EncoderLayer {
    pub fn new(
        channels: i32,
        n_heads: i32,
        filter_channels: i32,
        kernel_size: i32,
    ) -> Result<Self, Exception> {
        let attn = RelativeAttention::new(channels, n_heads)?;
        let ffn = FFN::new(channels, channels, filter_channels, kernel_size)?;
        let norm1 = ConvLayerNorm::new(channels)?;
        let norm2 = ConvLayerNorm::new(channels)?;

        Ok(Self {
            attn,
            ffn,
            norm1,
            norm2,
        })
    }

    /// Forward pass - POST-NORM version (matching Python GPT-SoVITS)
    /// Using norm(x + attn(x)) instead of x + attn(norm(x))
    pub fn forward(&mut self, x: &Array, mask: &Array) -> Result<Array, Exception> {
        // POST-NORM: x = norm1(x + attn(x))
        let attn_out = self.attn.forward(x, None)?;
        let x = self.norm1.forward(&x.add(&attn_out)?)?;

        // x = norm2(x + ffn(x))
        let ffn_out = self.ffn.forward(&x, mask)?;
        self.norm2.forward(&x.add(&ffn_out)?)
    }
}

/// Transformer encoder
#[derive(Debug, Clone, ModuleParameters)]
pub struct TransformerEncoder {
    #[param]
    pub layers: Vec<EncoderLayer>,
    pub n_layers: i32,
}

impl TransformerEncoder {
    pub fn new(
        channels: i32,
        n_heads: i32,
        filter_channels: i32,
        kernel_size: i32,
        n_layers: i32,
    ) -> Result<Self, Exception> {
        let mut layers = Vec::with_capacity(n_layers as usize);
        for _ in 0..n_layers {
            layers.push(EncoderLayer::new(
                channels,
                n_heads,
                filter_channels,
                kernel_size,
            )?);
        }
        Ok(Self { layers, n_layers })
    }

    /// Forward pass - simple version without explicit attention mask
    pub fn forward(&mut self, x: &Array, mask: &Array) -> Result<Array, Exception> {
        let mut h = x.clone();
        for layer in &mut self.layers {
            h = layer.forward(&h, mask)?;
        }
        Ok(h)
    }
}

// ============================================================================
// MRTE (Multi-Resolution Temporal Encoder) for cross-attention
// ============================================================================

/// Cross-attention for combining SSL and text features
#[derive(Debug, Clone, ModuleParameters)]
pub struct MRTECrossAttention {
    #[param]
    pub c_pre: nn::Conv1d,
    #[param]
    pub text_pre: nn::Conv1d,
    #[param]
    pub cross_attention: RelativeAttention,
    #[param]
    pub c_post: nn::Conv1d,
    pub channels: i32,
    pub hidden: i32,
}

impl MRTECrossAttention {
    pub fn new(channels: i32, hidden: i32, n_heads: i32) -> Result<Self, Exception> {
        let c_pre = nn::Conv1dBuilder::new(channels, hidden, 1).build()?;
        let text_pre = nn::Conv1dBuilder::new(channels, hidden, 1).build()?;
        let cross_attention = RelativeAttention::new(hidden, n_heads)?;
        let c_post = nn::Conv1dBuilder::new(hidden, channels, 1).build()?;

        Ok(Self {
            c_pre,
            text_pre,
            cross_attention,
            c_post,
            channels,
            hidden,
        })
    }

    /// Forward pass (expects NCL input, returns NCL output)
    /// Cross-attention: SSL features (query) attend to text features (key/value)
    ///
    /// Following actual GPT-SoVITS implementation:
    /// 1. Apply mask BEFORE c_pre/text_pre convolutions
    /// 2. Create attention mask from ssl_mask and text_mask
    /// 3. Apply mask BEFORE c_post convolution
    pub fn forward(
        &mut self,
        ssl_features: &Array,
        ssl_mask: &Array,
        text_features: &Array,
        text_mask: &Array,
        style: Option<&Array>,
    ) -> Result<Array, Exception> {
        // Create attention mask: text_mask.unsqueeze(2) * ssl_mask.unsqueeze(-1)
        // text_mask: [batch, 1, text_len] -> [batch, 1, 1, text_len]
        // ssl_mask: [batch, 1, ssl_len] -> [batch, 1, ssl_len, 1]
        // attn_mask: [batch, 1, ssl_len, text_len]
        let text_mask_4d = expand_dims(text_mask, 2)?;  // [batch, 1, 1, text_len]
        let ssl_mask_4d = expand_dims(ssl_mask, -1)?;   // [batch, 1, ssl_len, 1]
        let attn_mask = text_mask_4d.multiply(&ssl_mask_4d)?;  // [batch, 1, ssl_len, text_len]

        // Apply mask BEFORE c_pre (following actual GPT-SoVITS)
        let ssl_masked_input = ssl_features.multiply(ssl_mask)?;
        let text_masked_input = text_features.multiply(text_mask)?;

        // Convert NCL to NLC for Conv1d
        let ssl_nlc = swap_axes(&ssl_masked_input, 1, 2)?;
        let text_nlc = swap_axes(&text_masked_input, 1, 2)?;

        // Project features (NLC format for mlx-rs Conv1d)
        let ssl_proj = self.c_pre.forward(&ssl_nlc)?;
        let text_proj = self.text_pre.forward(&text_nlc)?;

        // Convert back to NCL for attention
        let ssl_ncl = swap_axes(&ssl_proj, 1, 2)?;  // [batch, hidden, ssl_seq]
        let text_ncl = swap_axes(&text_proj, 1, 2)?;  // [batch, hidden, text_seq]

        // Apply masks again for cross-attention input (following actual GPT-SoVITS)
        let ssl_masked = ssl_ncl.multiply(ssl_mask)?;
        let text_masked = text_ncl.multiply(text_mask)?;

        // Cross-attention: Q from SSL, K/V from text, with attention mask
        let attn_out = self.cross_attention.cross_forward(&ssl_masked, &text_masked, Some(&attn_mask))?;

        // Add residual from projected SSL
        let attn_out = attn_out.add(&ssl_masked)?;

        // Add style embedding if provided (ge=0 if None in Python)
        let attn_out = if let Some(ge) = style {
            attn_out.add(ge)?
        } else {
            attn_out
        };

        // Apply mask BEFORE c_post (following actual GPT-SoVITS)
        let attn_masked = attn_out.multiply(ssl_mask)?;

        // Convert NCL to NLC for output projection
        let attn_nlc = swap_axes(&attn_masked, 1, 2)?;
        let out = self.c_post.forward(&attn_nlc)?;
        // Convert back to NCL
        swap_axes(&out, 1, 2)
    }
}

// ============================================================================
// TextEncoder (enc_p)
// ============================================================================

/// TextEncoder: Combines SSL features with text phoneme features
#[derive(Debug, Clone, ModuleParameters)]
pub struct TextEncoder {
    #[param]
    pub ssl_proj: nn::Conv1d,
    #[param]
    pub encoder_ssl: TransformerEncoder,
    #[param]
    pub text_embedding: nn::Embedding,
    #[param]
    pub encoder_text: TransformerEncoder,
    #[param]
    pub mrte: MRTECrossAttention,
    #[param]
    pub encoder2: TransformerEncoder,
    #[param]
    pub proj: nn::Conv1d,
    pub out_channels: i32,
}

impl TextEncoder {
    pub fn new(config: &VITSConfig) -> Result<Self, Exception> {
        let ssl_proj = nn::Conv1dBuilder::new(config.ssl_dim, config.hidden_channels, 1).build()?;

        let encoder_ssl = TransformerEncoder::new(
            config.hidden_channels,
            config.n_heads,
            config.filter_channels,
            config.kernel_size,
            config.n_layers / 2,
        )?;

        let text_embedding = nn::Embedding::new(config.vocab_size, config.hidden_channels)?;

        let encoder_text = TransformerEncoder::new(
            config.hidden_channels,
            config.n_heads,
            config.filter_channels,
            config.kernel_size,
            config.n_layers,
        )?;

        let mrte = MRTECrossAttention::new(config.hidden_channels, config.gin_channels, 4)?;

        let encoder2 = TransformerEncoder::new(
            config.hidden_channels,
            config.n_heads,
            config.filter_channels,
            config.kernel_size,
            config.n_layers / 2,
        )?;

        // Output: mean and log_var (2 * hidden_channels)
        let proj =
            nn::Conv1dBuilder::new(config.hidden_channels, config.hidden_channels * 2, 1).build()?;

        Ok(Self {
            ssl_proj,
            encoder_ssl,
            text_embedding,
            encoder_text,
            mrte,
            encoder2,
            proj,
            out_channels: config.hidden_channels,
        })
    }

    /// Forward pass (matching actual GPT-SoVITS TextEncoder.forward)
    /// - quantized: [batch, ssl_dim, seq] from RVQ decode (NCL format)
    /// - text: [batch, text_seq] phoneme indices
    /// - style: [batch, gin_channels, 1] style embedding
    /// Returns: (encoded, mean, log_var, mask) all in NCL format
    pub fn forward(
        &mut self,
        quantized: &Array,
        text: &Array,
        style: Option<&Array>,
    ) -> Result<(Array, Array, Array, Array), Exception> {
        let batch = quantized.shape()[0] as i32;
        let seq_len = quantized.shape()[2] as i32;

        // Create masks
        // NCL format mask for convolutions and encoder
        let y_mask = Array::ones::<f32>(&[batch, 1, seq_len])?;

        // Step 1: ssl_proj with mask before AND after (matching Python)
        // Python: y = self.ssl_proj(y * y_mask) * y_mask
        let quantized_masked = quantized.multiply(&y_mask)?;  // mask BEFORE ssl_proj
        let quantized_nlc = swap_axes(&quantized_masked, 1, 2)?;
        let ssl = self.ssl_proj.forward(&quantized_nlc)?;
        let mask_nlc = swap_axes(&y_mask, 1, 2)?;
        let ssl = ssl.multiply(&mask_nlc)?;  // mask AFTER ssl_proj
        let ssl_ncl = swap_axes(&ssl, 1, 2)?;

        // Step 2: encoder_ssl with mask before (matching Python)
        // Python: y = self.encoder_ssl(y * y_mask, y_mask)
        let ssl_masked = ssl_ncl.multiply(&y_mask)?;  // mask BEFORE encoder_ssl
        let ssl_enc = self.encoder_ssl.forward(&ssl_masked, &y_mask)?;

        // Step 3: text embedding and encoder_text with mask before
        // Python: text = self.encoder_text(text * text_mask, text_mask)
        let text_seq_len = text.shape()[1] as i32;
        let text_mask = Array::ones::<f32>(&[batch, 1, text_seq_len])?;
        let text_embed = self.text_embedding.forward(text)?;
        // [batch, seq, channels] -> [batch, channels, seq]
        let text_embed = swap_axes(&text_embed, 1, 2)?;
        let text_masked = text_embed.multiply(&text_mask)?;  // mask BEFORE encoder_text
        let text_enc = self.encoder_text.forward(&text_masked, &text_mask)?;

        // Step 4: MRTE (already fixed to match actual GPT-SoVITS)
        let mrte_out = self.mrte.forward(&ssl_enc, &y_mask, &text_enc, &text_mask, style)?;

        // Step 5: encoder2 with mask before (matching Python)
        // Python: y = self.encoder2(y * y_mask, y_mask)
        let mrte_masked = mrte_out.multiply(&y_mask)?;  // mask BEFORE encoder2
        let encoded = self.encoder2.forward(&mrte_masked, &y_mask)?;

        // Step 6: output projection
        // Python: stats = self.proj(y) * y_mask
        let encoded_nlc = swap_axes(&encoded, 1, 2)?;
        let stats = self.proj.forward(&encoded_nlc)?;
        let stats = swap_axes(&stats, 1, 2)?;
        let stats = stats.multiply(&y_mask)?;

        // Split into mean and log_var
        let halves = split(&stats, 2, 1)?;
        let mean = halves[0].clone();
        let log_var = halves[1].clone();

        Ok((encoded, mean, log_var, y_mask))
    }

    /// Debug forward that returns all intermediate outputs
    pub fn forward_debug(
        &mut self,
        quantized: &Array,
        text: &Array,
        style: Option<&Array>,
    ) -> Result<Vec<(String, Array)>, Exception> {
        let mut outputs = Vec::new();

        let batch = quantized.shape()[0] as i32;
        let seq_len = quantized.shape()[2] as i32;

        // Create masks
        let mask_nlc = Array::ones::<f32>(&[batch, seq_len, 1])?;
        let mask_ncl = Array::ones::<f32>(&[batch, 1, seq_len])?;
        outputs.push(("step0_y_mask".to_string(), mask_ncl.clone()));

        // Step 1: ssl_proj
        let quantized_nlc = swap_axes(quantized, 1, 2)?;
        outputs.push(("step1_ssl_proj_input".to_string(), quantized.clone()));

        let ssl = self.ssl_proj.forward(&quantized_nlc)?;
        let ssl = ssl.multiply(&mask_nlc)?;
        let ssl_ncl = swap_axes(&ssl, 1, 2)?;
        outputs.push(("step1_ssl_proj_output".to_string(), ssl_ncl.clone()));

        // Step 2: encoder_ssl
        let mask_enc = Array::ones::<f32>(&[batch, 1, seq_len])?;
        outputs.push(("step2_encoder_ssl_input".to_string(), ssl_ncl.clone()));

        let ssl_ncl = self.encoder_ssl.forward(&ssl_ncl, &mask_enc)?;
        outputs.push(("step2_encoder_ssl_output".to_string(), ssl_ncl.clone()));

        // Step 3: text_embedding and encoder_text
        let text_seq_len = text.shape()[1] as i32;
        let text_mask = Array::ones::<f32>(&[batch, 1, text_seq_len])?;
        outputs.push(("step3_text_mask".to_string(), text_mask.clone()));

        let text_embed = self.text_embedding.forward(text)?;
        let text_embed = swap_axes(&text_embed, 1, 2)?;
        outputs.push(("step3_text_embed".to_string(), text_embed.clone()));

        let text_encoded = self.encoder_text.forward(&text_embed, &text_mask)?;
        outputs.push(("step3_text_encoded".to_string(), text_encoded.clone()));

        // Step 4: mrte
        let combined = self.mrte.forward(&ssl_ncl, &mask_enc, &text_encoded, &text_mask, style)?;
        outputs.push(("step4_mrte_output".to_string(), combined.clone()));

        // Step 5: encoder2
        let enc2_input = combined.multiply(&mask_enc)?;
        outputs.push(("step5_encoder2_input".to_string(), enc2_input.clone()));

        let encoded = self.encoder2.forward(&enc2_input, &mask_enc)?;
        outputs.push(("step5_encoder2_output".to_string(), encoded.clone()));

        // Step 6: proj
        let encoded_nlc = swap_axes(&encoded, 1, 2)?;
        let stats = self.proj.forward(&encoded_nlc)?;
        let stats = swap_axes(&stats, 1, 2)?;
        let stats = stats.multiply(&mask_ncl)?;
        outputs.push(("step6_proj_output".to_string(), stats.clone()));

        let halves = split(&stats, 2, 1)?;
        outputs.push(("step6_m_p".to_string(), halves[0].clone()));
        outputs.push(("step6_logs_p".to_string(), halves[1].clone()));

        Ok(outputs)
    }
}

// ============================================================================
// WN (WaveNet-style) encoder for flow
// ============================================================================

/// WaveNet-style network for flow coupling layers
#[derive(Debug, Clone, ModuleParameters)]
pub struct WNEncoder {
    #[param]
    pub in_layers: Vec<nn::Conv1d>,
    #[param]
    pub res_skip_layers: Vec<nn::Conv1d>,
    #[param]
    pub cond_layer: nn::Conv1d,
    pub n_layers: i32,
    pub hidden_channels: i32,
}

impl WNEncoder {
    pub fn new(
        hidden_channels: i32,
        kernel_size: i32,
        n_layers: i32,
        gin_channels: i32,
    ) -> Result<Self, Exception> {
        let padding = (kernel_size - 1) / 2;
        let mut in_layers = Vec::with_capacity(n_layers as usize);
        let mut res_skip_layers = Vec::with_capacity(n_layers as usize);

        for i in 0..n_layers {
            let dilation = 1; // Simplified: use dilation 1
            let in_layer = nn::Conv1dBuilder::new(hidden_channels, hidden_channels * 2, kernel_size)
                .padding(padding * dilation)
                .dilation(dilation)
                .build()?;
            in_layers.push(in_layer);

            // Last layer outputs hidden_channels, others output hidden_channels * 2
            let out_ch = if i < n_layers - 1 {
                hidden_channels * 2
            } else {
                hidden_channels
            };
            let res_skip = nn::Conv1dBuilder::new(hidden_channels, out_ch, 1).build()?;
            res_skip_layers.push(res_skip);
        }

        let cond_layer =
            nn::Conv1dBuilder::new(gin_channels, hidden_channels * 2 * n_layers, 1).build()?;

        Ok(Self {
            in_layers,
            res_skip_layers,
            cond_layer,
            n_layers,
            hidden_channels,
        })
    }

    /// Forward pass (expects NCL input, returns NCL output)
    pub fn forward(
        &mut self,
        x: &Array,
        mask: &Array,
        g: Option<&Array>,
    ) -> Result<Array, Exception> {
        let mut output = zeros_like(x)?;

        // Condition on style (NCL -> NLC -> NCL for conv)
        let g_cond = if let Some(style) = g {
            let style_nlc = swap_axes(style, 1, 2)?;
            let cond = self.cond_layer.forward(&style_nlc)?;
            Some(swap_axes(&cond, 1, 2)?) // Back to NCL
        } else {
            None
        };

        let _mask_nlc = swap_axes(mask, 1, 2)?;
        let mut h = x.clone();

        for (i, (in_layer, res_skip)) in self
            .in_layers
            .iter_mut()
            .zip(self.res_skip_layers.iter_mut())
            .enumerate()
        {
            // Convert to NLC for conv
            let h_nlc = swap_axes(&h, 1, 2)?;
            let h_in_nlc = in_layer.forward(&h_nlc)?;
            let h_in = swap_axes(&h_in_nlc, 1, 2)?; // Back to NCL

            // Add conditioning if available (both in NCL)
            let h_in = if let Some(ref g) = g_cond {
                let g_slice =
                    g.index((.., i as i32 * self.hidden_channels * 2..(i as i32 + 1) * self.hidden_channels * 2, ..));
                h_in.add(&g_slice)?
            } else {
                h_in
            };

            // Gated activation (NCL format, split on channel dim 1)
            let halves = split(&h_in, 2, 1)?;
            let h_tanh = tanh(&halves[0])?;
            let h_sigmoid = nn::sigmoid(&halves[1])?;
            let acts = h_tanh.multiply(&h_sigmoid)?; // NCL

            // Residual and skip connection (convert to NLC for conv)
            let acts_nlc = swap_axes(&acts, 1, 2)?;
            let res_skip_out_nlc = res_skip.forward(&acts_nlc)?;
            let res_skip_out = swap_axes(&res_skip_out_nlc, 1, 2)?; // Back to NCL

            if i < (self.n_layers - 1) as usize {
                let res_skip_halves = split(&res_skip_out, 2, 1)?;
                // Python: x = (x + res_acts) * x_mask
                h = h.add(&res_skip_halves[0])?.multiply(mask)?;
                output = output.add(&res_skip_halves[1])?;
            } else {
                output = output.add(&res_skip_out)?;
            }
        }

        output.multiply(mask)
    }
}

// ============================================================================
// ResidualCouplingLayer
// ============================================================================

/// Residual coupling layer for normalizing flow
#[derive(Debug, Clone, ModuleParameters)]
pub struct ResidualCouplingLayer {
    #[param]
    pub pre: nn::Conv1d,
    #[param]
    pub enc: WNEncoder,
    #[param]
    pub post: nn::Conv1d,
    pub half_channels: i32,
    pub mean_only: bool,
}

impl ResidualCouplingLayer {
    pub fn new(
        channels: i32,
        hidden_channels: i32,
        kernel_size: i32,
        n_layers: i32,
        gin_channels: i32,
        mean_only: bool,
    ) -> Result<Self, Exception> {
        let half_channels = channels / 2;

        let pre = nn::Conv1dBuilder::new(half_channels, hidden_channels, 1).build()?;

        let enc = WNEncoder::new(hidden_channels, kernel_size, n_layers, gin_channels)?;

        let post_out = if mean_only {
            half_channels
        } else {
            half_channels * 2
        };
        let post = nn::Conv1dBuilder::new(hidden_channels, post_out, 1).build()?;

        Ok(Self {
            pre,
            enc,
            post,
            half_channels,
            mean_only,
        })
    }

    /// Forward pass (expects NCL input, returns NCL output)
    pub fn forward(
        &mut self,
        x: &Array,
        mask: &Array,
        g: Option<&Array>,
        reverse: bool,
    ) -> Result<Array, Exception> {
        // Split input (NCL format)
        let x0 = x.index((.., ..self.half_channels, ..));
        let x1 = x.index((.., self.half_channels.., ..));

        // Convert NCL to NLC for pre conv
        let x0_nlc = swap_axes(&x0, 1, 2)?;
        let h = self.pre.forward(&x0_nlc)?;
        // Back to NCL
        let h = swap_axes(&h, 1, 2)?;
        let h = h.multiply(mask)?;

        // WNEncoder forward (expects/returns NCL)
        let h = self.enc.forward(&h, mask, g)?;

        // Convert NCL to NLC for post conv
        let h_nlc = swap_axes(&h, 1, 2)?;
        let stats = self.post.forward(&h_nlc)?;
        // Back to NCL
        let stats = swap_axes(&stats, 1, 2)?;
        let stats = stats.multiply(mask)?;

        let m = if self.mean_only {
            stats
        } else {
            let halves = split(&stats, 2, 1)?;
            halves[0].clone()
        };

        // Apply coupling
        let x1 = if reverse {
            x1.subtract(&m)?.multiply(mask)?
        } else {
            x1.add(&m)?.multiply(mask)?
        };

        // Concatenate
        concatenate_axis(&[&x0, &x1], 1)
    }
}

// ============================================================================
// ResidualCouplingBlock (flow)
// ============================================================================

/// Flow model with residual coupling layers
#[derive(Debug, Clone, ModuleParameters)]
pub struct ResidualCouplingBlock {
    #[param]
    pub flows: Vec<ResidualCouplingLayer>,
    pub n_flows: i32,
}

impl ResidualCouplingBlock {
    pub fn new(
        channels: i32,
        hidden_channels: i32,
        kernel_size: i32,
        n_layers: i32,
        n_flows: i32,
        gin_channels: i32,
    ) -> Result<Self, Exception> {
        let mut flows = Vec::with_capacity(n_flows as usize);
        for _ in 0..n_flows {
            flows.push(ResidualCouplingLayer::new(
                channels,
                hidden_channels,
                kernel_size,
                n_layers,
                gin_channels,
                true, // mean_only
            )?);
        }
        Ok(Self { flows, n_flows })
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: &Array,
        g: Option<&Array>,
        reverse: bool,
    ) -> Result<Array, Exception> {
        let mut h = x.clone();

        // Helper to flip channels (reverse along dim 1)
        fn flip_channels(x: &Array) -> Result<Array, Exception> {
            let n_channels = x.shape()[1] as i32;
            // Create reversed indices: [n-1, n-2, ..., 1, 0]
            let indices = Array::from_iter((0..n_channels).rev(), &[n_channels]);
            x.take_axis(&indices, 1)
        }

        if reverse {
            for flow in self.flows.iter_mut().rev() {
                // Flip: reverse entire channel dimension (like torch.flip(x, [1]))
                h = flip_channels(&h)?;
                // Apply coupling
                h = flow.forward(&h, mask, g, true)?;
            }
        } else {
            for flow in &mut self.flows {
                h = flow.forward(&h, mask, g, false)?;
                // Flip: reverse entire channel dimension
                h = flip_channels(&h)?;
            }
        }

        Ok(h)
    }
}

// ============================================================================
// HiFiGAN Generator (dec)
// ============================================================================

/// ResBlock for HiFiGAN
#[derive(Debug, Clone, ModuleParameters)]
pub struct HiFiGANResBlock {
    #[param]
    pub convs1: Vec<nn::Conv1d>,
    #[param]
    pub convs2: Vec<nn::Conv1d>,
}

impl HiFiGANResBlock {
    pub fn new(channels: i32, kernel_size: i32, dilations: &[i32]) -> Result<Self, Exception> {
        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();

        for &d in dilations {
            let padding = (kernel_size - 1) * d / 2;
            convs1.push(
                nn::Conv1dBuilder::new(channels, channels, kernel_size)
                    .padding(padding)
                    .dilation(d)
                    .build()?,
            );
            convs2.push(
                nn::Conv1dBuilder::new(channels, channels, kernel_size)
                    .padding((kernel_size - 1) / 2)
                    .build()?,
            );
        }

        Ok(Self { convs1, convs2 })
    }

    /// Forward pass (expects NLC input, returns NLC output)
    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // Process through all dilations with skip connection at each step
        // Matching Python: x = xt + x inside the loop
        let mut h = x.clone();
        for (c1, c2) in self.convs1.iter_mut().zip(self.convs2.iter_mut()) {
            let xt = nn::leaky_relu(&h, 0.1)?;
            let xt = c1.forward(&xt)?;
            let xt = nn::leaky_relu(&xt, 0.1)?;
            let xt = c2.forward(&xt)?;
            h = xt.add(&h)?;  // Skip connection inside loop
        }
        Ok(h)
    }
}

/// HiFiGAN Generator with Weight Normalization
///
/// Uses weight normalization on conv_pre, ups, and conv_post layers
/// to match PyTorch training behavior and prevent weight drift.
#[derive(Debug, Clone, ModuleParameters)]
pub struct HiFiGANGenerator {
    #[param]
    pub conv_pre: WeightNormConv1d,
    #[param]
    pub ups: Vec<WeightNormConvTranspose1d>,
    #[param]
    pub resblocks: Vec<HiFiGANResBlock>,
    #[param]
    pub conv_post: WeightNormConv1d,
    #[param]
    pub cond: nn::Conv1d,
    pub num_kernels: i32,
    pub num_upsamples: i32,
}

impl HiFiGANGenerator {
    pub fn new(config: &VITSConfig) -> Result<Self, Exception> {
        // conv_pre with weight normalization
        let conv_pre = WeightNormConv1d::new(
            config.hidden_channels,
            config.upsample_initial_channel,
            7,    // kernel_size
            1,    // stride
            3,    // padding
            1,    // dilation
            true, // bias
        )?;

        // Upsample layers with weight normalization
        let mut ups = Vec::new();
        let mut ch = config.upsample_initial_channel;
        for (_i, (&u, &k)) in config
            .upsample_rates
            .iter()
            .zip(config.upsample_kernel_sizes.iter())
            .enumerate()
        {
            let out_ch = ch / 2;
            ups.push(WeightNormConvTranspose1d::new(
                ch,           // in_channels
                out_ch,       // out_channels
                k,            // kernel_size
                u,            // stride
                (k - u) / 2,  // padding
                true,         // bias
            )?);
            ch = out_ch;
        }

        // ResBlocks (not weight normalized in original)
        let mut resblocks = Vec::new();
        ch = config.upsample_initial_channel;
        for _i in 0..config.upsample_rates.len() {
            ch = ch / 2;
            for (_j, (k, d)) in config
                .resblock_kernel_sizes
                .iter()
                .zip(config.resblock_dilation_sizes.iter())
                .enumerate()
            {
                resblocks.push(HiFiGANResBlock::new(ch, *k, d)?);
            }
        }

        // conv_post with weight normalization
        let final_ch = config.upsample_initial_channel
            / (2_i32.pow(config.upsample_rates.len() as u32));
        let conv_post = WeightNormConv1d::new(
            final_ch,
            1,    // out_channels
            7,    // kernel_size
            1,    // stride
            3,    // padding
            1,    // dilation
            false, // no bias on final layer
        )?;

        // cond layer (not weight normalized)
        let cond =
            nn::Conv1dBuilder::new(config.gin_channels, config.upsample_initial_channel, 1)
                .build()?;

        Ok(Self {
            conv_pre,
            ups,
            resblocks,
            conv_post,
            cond,
            num_kernels: config.resblock_kernel_sizes.len() as i32,
            num_upsamples: config.upsample_rates.len() as i32,
        })
    }

    /// Forward pass (expects NCL input, returns NCL output)
    pub fn forward(&mut self, x: &Array, g: Option<&Array>) -> Result<Array, Exception> {
        // Convert NCL to NLC for Conv1d
        let x_nlc = swap_axes(x, 1, 2)?;
        let mut h = self.conv_pre.forward(&x_nlc)?;

        // Add style conditioning (also in NLC)
        if let Some(style) = g {
            let style_nlc = swap_axes(style, 1, 2)?;
            let cond = self.cond.forward(&style_nlc)?;
            h = h.add(&cond)?;
        }

        let mut resblock_idx = 0;
        for up in self.ups.iter_mut() {
            h = nn::leaky_relu(&h, 0.1)?;
            h = up.forward(&h)?;

            // Apply resblocks (all in NLC)
            let mut xs = None::<Array>;
            for _ in 0..self.num_kernels {
                if resblock_idx < self.resblocks.len() {
                    let rb_out = self.resblocks[resblock_idx].forward(&h)?;
                    xs = Some(match xs {
                        Some(acc) => acc.add(&rb_out)?,
                        None => rb_out,
                    });
                    resblock_idx += 1;
                }
            }

            if let Some(x_sum) = xs {
                h = x_sum.divide(array!(self.num_kernels as f32))?;
            }
        }

        h = nn::leaky_relu(&h, 0.1)?;
        h = self.conv_post.forward(&h)?;
        let h = tanh(&h)?;

        // Convert NLC back to NCL
        swap_axes(&h, 1, 2)
    }
}

// ============================================================================
// MelStyleEncoder (ref_enc)
// ============================================================================

/// MelStyleEncoder for extracting style from reference mel spectrogram
#[derive(Debug, Clone, ModuleParameters)]
pub struct MelStyleEncoder {
    #[param]
    pub spectral_0: nn::Linear,
    #[param]
    pub spectral_1: nn::Linear,
    #[param]
    pub temporal_0: nn::Conv1d,
    #[param]
    pub temporal_1: nn::Conv1d,
    #[param]
    pub slf_attn_q: nn::Linear,
    #[param]
    pub slf_attn_k: nn::Linear,
    #[param]
    pub slf_attn_v: nn::Linear,
    #[param]
    pub slf_attn_fc: nn::Linear,
    #[param]
    pub fc: nn::Linear,
    pub hidden_dim: i32,
    pub out_dim: i32,
}

impl MelStyleEncoder {
    pub fn new(mel_channels: i32, hidden_dim: i32, out_dim: i32) -> Result<Self, Exception> {
        let spectral_0 = nn::LinearBuilder::new(mel_channels, hidden_dim)
            .bias(true)
            .build()?;
        let spectral_1 = nn::LinearBuilder::new(hidden_dim, hidden_dim)
            .bias(true)
            .build()?;

        // GLU convolutions
        let temporal_0 = nn::Conv1dBuilder::new(hidden_dim, hidden_dim * 2, 5)
            .padding(2)
            .build()?;
        let temporal_1 = nn::Conv1dBuilder::new(hidden_dim, hidden_dim * 2, 5)
            .padding(2)
            .build()?;

        // Self-attention
        let slf_attn_q = nn::LinearBuilder::new(hidden_dim, hidden_dim)
            .bias(true)
            .build()?;
        let slf_attn_k = nn::LinearBuilder::new(hidden_dim, hidden_dim)
            .bias(true)
            .build()?;
        let slf_attn_v = nn::LinearBuilder::new(hidden_dim, hidden_dim)
            .bias(true)
            .build()?;
        let slf_attn_fc = nn::LinearBuilder::new(hidden_dim, hidden_dim)
            .bias(true)
            .build()?;

        let fc = nn::LinearBuilder::new(hidden_dim, out_dim)
            .bias(true)
            .build()?;

        Ok(Self {
            spectral_0,
            spectral_1,
            temporal_0,
            temporal_1,
            slf_attn_q,
            slf_attn_k,
            slf_attn_v,
            slf_attn_fc,
            fc,
            hidden_dim,
            out_dim,
        })
    }

    fn mish(x: &Array) -> Result<Array, Exception> {
        // mish(x) = x * tanh(softplus(x))
        let softplus = x.exp()?.add(array!(1.0f32))?.log()?;
        x.multiply(&tanh(&softplus)?)
    }

    fn glu(x: &Array) -> Result<Array, Exception> {
        // GLU: x * sigmoid(gate)
        let halves = split(x, 2, -1)?;
        halves[0].multiply(&nn::sigmoid(&halves[1])?)
    }

    /// Forward pass (expects NCL input mel, returns [batch, out_dim, 1] style)
    pub fn forward(&mut self, mel: &Array) -> Result<Array, Exception> {
        // mel: [batch, mel_channels, time] NCL -> [batch, time, mel_channels] NLC
        let x = swap_axes(mel, 1, 2)?;

        // Spectral processing (Linear operates on last dim, so NLC is correct)
        let x = self.spectral_0.forward(&x)?;
        let x = Self::mish(&x)?;
        let x = self.spectral_1.forward(&x)?;
        let x = Self::mish(&x)?;

        // Temporal processing with GLU and RESIDUAL connection
        // Python Conv1dGLU: residual = x; x = conv(x); x = glu(x); x = residual + x
        // Conv1d in mlx-rs expects NLC format
        let residual = x.clone();
        let x = self.temporal_0.forward(&x)?; // NLC -> NLC (but doubled channels)
        let x = Self::glu(&x)?; // Split on last dim and apply GLU
        let x = residual.add(&x)?; // RESIDUAL connection

        let residual = x.clone();
        let x = self.temporal_1.forward(&x)?;
        let x = Self::glu(&x)?;
        let x = residual.add(&x)?; // RESIDUAL connection

        // Self-attention with RESIDUAL connection
        // Python: residual = x; ... output = fc(output) + residual
        let residual = x.clone();

        // Multi-head attention: n_head=2, d_k=d_v=hidden_dim/2=64
        // Q, K, V: [batch, time, hidden] -> [batch, time, n_head, d_k]
        let n_head = 2;
        let d_k = self.hidden_dim / n_head;
        let batch = x.dim(0);
        let seq_len = x.dim(1);

        let q = self.slf_attn_q.forward(&x)?;
        let k = self.slf_attn_k.forward(&x)?;
        let v = self.slf_attn_v.forward(&x)?;

        // Reshape for multi-head: [batch, time, hidden] -> [batch, time, n_head, d_k] -> [batch*n_head, time, d_k]
        let q = q.reshape(&[batch, seq_len, n_head, d_k])?;
        let q = q.transpose_axes(&[2, 0, 1, 3])?; // [n_head, batch, time, d_k]
        let q = q.reshape(&[n_head * batch, seq_len, d_k])?;

        let k = k.reshape(&[batch, seq_len, n_head, d_k])?;
        let k = k.transpose_axes(&[2, 0, 1, 3])?;
        let k = k.reshape(&[n_head * batch, seq_len, d_k])?;

        let v = v.reshape(&[batch, seq_len, n_head, d_k])?;
        let v = v.transpose_axes(&[2, 0, 1, 3])?;
        let v = v.reshape(&[n_head * batch, seq_len, d_k])?;

        // Attention scores: [n_head*batch, time, time]
        let scale = (self.hidden_dim as f32).sqrt(); // d_model not d_k for temperature
        let scores = matmul(&q, &swap_axes(&k, 1, 2)?)?;
        let attn = softmax_axis(&scores.divide(array!(scale))?, -1, false)?;
        let attn_out = matmul(&attn, &v)?; // [n_head*batch, time, d_k]

        // Reshape back: [n_head*batch, time, d_k] -> [n_head, batch, time, d_k] -> [batch, time, n_head, d_k] -> [batch, time, hidden]
        let attn_out = attn_out.reshape(&[n_head, batch, seq_len, d_k])?;
        let attn_out = attn_out.transpose_axes(&[1, 2, 0, 3])?; // [batch, time, n_head, d_k]
        let attn_out = attn_out.reshape(&[batch, seq_len, self.hidden_dim])?;

        let x = self.slf_attn_fc.forward(&attn_out)?;
        let x = x.add(&residual)?; // RESIDUAL connection for attention

        // Temporal average pooling: [batch, time, hidden] -> [batch, hidden]
        let x = x.mean_axis(1, false)?;

        // Final projection: [batch, out_dim]
        let style = self.fc.forward(&x)?;

        // Add trailing dimension for broadcasting: [batch, out_dim, 1]
        Ok(style.index((.., .., mlx_rs::ops::indexing::NewAxis)))
    }
}

// ============================================================================
// PosteriorEncoder (enc_q) - for training
// ============================================================================

/// Posterior Encoder that encodes spectrogram to latent distribution.
/// Used during training to provide supervision signal.
///
/// Input: linear spectrogram [batch, spec_channels, time]
/// Output: z, mean, log_variance, mask
#[derive(Debug, Clone, ModuleParameters)]
pub struct PosteriorEncoder {
    #[param]
    pub pre: nn::Conv1d,
    #[param]
    pub enc: WNEncoder,
    #[param]
    pub proj: nn::Conv1d,
    pub out_channels: i32,
}

impl PosteriorEncoder {
    /// Create a new PosteriorEncoder
    ///
    /// Args:
    /// - in_channels: Input channels (usually n_fft/2+1 = 1025 for spec)
    /// - out_channels: Output channels (usually hidden_channels = 192)
    /// - hidden_channels: Hidden channels for WN (usually 192)
    /// - kernel_size: Kernel size for WN (usually 5)
    /// - n_layers: Number of WN layers (usually 16)
    /// - gin_channels: Style conditioning channels (usually 512)
    pub fn new(
        in_channels: i32,
        out_channels: i32,
        hidden_channels: i32,
        kernel_size: i32,
        n_layers: i32,
        gin_channels: i32,
    ) -> Result<Self, Exception> {
        let pre = nn::Conv1dBuilder::new(in_channels, hidden_channels, 1).build()?;

        let enc = WNEncoder::new(hidden_channels, kernel_size, n_layers, gin_channels)?;

        // Output is mean and log_variance, so out_channels * 2
        let proj = nn::Conv1dBuilder::new(hidden_channels, out_channels * 2, 1).build()?;

        Ok(Self {
            pre,
            enc,
            proj,
            out_channels,
        })
    }

    /// Forward pass
    ///
    /// Args:
    /// - x: Linear spectrogram [batch, spec_channels, time] in NCL format
    /// - x_mask: Mask [batch, 1, time]
    /// - g: Optional style conditioning [batch, gin_channels, 1]
    ///
    /// Returns:
    /// - z: Sampled latent [batch, out_channels, time]
    /// - m: Mean [batch, out_channels, time]
    /// - logs: Log variance [batch, out_channels, time]
    pub fn forward(
        &mut self,
        x: &Array,
        x_mask: &Array,
        g: Option<&Array>,
    ) -> Result<(Array, Array, Array), Exception> {
        // Pre-projection (NCL -> NLC for conv, then back to NCL)
        let x_nlc = swap_axes(x, 1, 2)?;
        let h = self.pre.forward(&x_nlc)?;
        let h = swap_axes(&h, 1, 2)?; // Back to NCL
        let h = h.multiply(x_mask)?;

        // WN encoder (expects NCL)
        let h = self.enc.forward(&h, x_mask, g)?;

        // Output projection
        let h_nlc = swap_axes(&h, 1, 2)?;
        let stats = self.proj.forward(&h_nlc)?;
        let stats = swap_axes(&stats, 1, 2)?; // Back to NCL
        let stats = stats.multiply(x_mask)?;

        // Split into mean and log_variance
        let halves = split(&stats, 2, 1)?;
        let m = halves[0].clone();
        let logs = halves[1].clone();

        // Sample z = m + randn * exp(logs)
        let noise = random::normal::<f32>(m.shape(), None, None, None)?;
        let z = m.add(&noise.multiply(&exp(&logs)?)?)?;
        let z = z.multiply(x_mask)?;

        Ok((z, m, logs))
    }
}

// ============================================================================
// SynthesizerTrn (full VITS model)
// ============================================================================

/// SynthesizerTrn: Full VITS model for GPT-SoVITS
#[derive(Debug, Clone, ModuleParameters)]
pub struct SynthesizerTrn {
    pub config: VITSConfig,
    #[param]
    pub quantizer: RVQCodebook,
    #[param]
    pub enc_p: TextEncoder,
    #[param]
    pub enc_q: PosteriorEncoder,
    #[param]
    pub flow: ResidualCouplingBlock,
    #[param]
    pub dec: HiFiGANGenerator,
    #[param]
    pub ref_enc: MelStyleEncoder,
    #[param]
    pub ssl_proj: nn::Conv1d,
}

impl SynthesizerTrn {
    pub fn new(config: VITSConfig) -> Result<Self, Exception> {
        let quantizer = RVQCodebook::new(config.codebook_size, config.codebook_dim)?;

        let enc_p = TextEncoder::new(&config)?;

        // PosteriorEncoder (enc_q) - encodes spectrogram to latent
        // in_channels = 1025 (n_fft/2+1 for 2048 FFT)
        // out_channels = hidden_channels = 192
        let enc_q = PosteriorEncoder::new(
            1025, // spec_channels (n_fft/2+1 for 2048-point FFT)
            config.hidden_channels,
            config.hidden_channels,
            5,  // kernel_size
            16, // n_layers in WN
            config.gin_channels,
        )?;

        let flow = ResidualCouplingBlock::new(
            config.hidden_channels,
            config.hidden_channels,
            5, // kernel_size
            4, // n_layers in WN
            config.n_flows,
            config.gin_channels,
        )?;

        let dec = HiFiGANGenerator::new(&config)?;

        let ref_enc = MelStyleEncoder::new(704, 128, config.gin_channels)?;

        // SSL projection before quantizer
        // - 25hz models: kernel=2, stride=2 (2x downsampling)
        // - 50hz models: kernel=1, stride=1 (no downsampling)
        let ssl_proj = if config.semantic_frame_rate == "25hz" {
            nn::Conv1dBuilder::new(config.ssl_dim, config.ssl_dim, 2)
                .stride(2)
                .padding(0)
                .build()?
        } else {
            nn::Conv1dBuilder::new(config.ssl_dim, config.ssl_dim, 1)
                .stride(1)
                .padding(0)
                .build()?
        };

        Ok(Self {
            config,
            quantizer,
            enc_p,
            enc_q,
            flow,
            dec,
            ref_enc,
            ssl_proj,
        })
    }

    /// Decode semantic codes to audio
    ///
    /// Args:
    /// - codes: Semantic codes [1, 1, seq] from T2S
    /// - text: Phoneme indices [batch, text_seq]
    /// - refer: Reference mel spectrogram [batch, mel_channels, time] (optional)
    /// - noise_scale: Noise scale for sampling (default 0.5)
    /// - speed: Speed factor (default 1.0, >1.0 = faster speech, <1.0 = slower)
    pub fn decode(
        &mut self,
        codes: &Array,
        text: &Array,
        refer: Option<&Array>,
        noise_scale: f32,
        speed: f32,
    ) -> Result<Array, Exception> {
        // Get style embedding from reference
        // For v2, slice to first 704 channels: refer[:, :704, :]
        let ge = if let Some(r) = refer {
            let r_sliced = r.index((.., ..704, ..));
            Some(self.ref_enc.forward(&r_sliced)?)
        } else {
            None
        };

        // Decode quantized features from codes
        let quantized = self.quantizer.decode(codes)?;

        // Interpolate if needed (25hz -> 50hz for semantic_frame_rate="25hz")
        // Input: [1, dim, seq] -> Output: [1, dim, seq*2]
        // Each position is repeated: [a0, a1, a2] -> [a0, a0, a1, a1, a2, a2]
        let seq_len = quantized.shape()[2] as i32;
        let target_len = seq_len * 2;
        // Add axis at end: [1, dim, seq] -> [1, dim, seq, 1]
        let q_expanded = quantized.index((.., .., .., mlx_rs::ops::indexing::NewAxis));
        // Repeat along the new axis: [1, dim, seq, 2]
        let q_rep = Array::repeat_axis::<f32>(q_expanded, 2, 3)?;
        // Reshape: [1, dim, seq*2]
        let quantized = q_rep.reshape(&[1, self.config.codebook_dim, target_len])?;

        // Apply speed factor via linear interpolation
        // speed > 1.0 = faster (shorter sequence), speed < 1.0 = slower (longer sequence)
        // Python: y = F.interpolate(y, size=int(y.shape[-1] / speed)+1, mode="linear")
        let quantized = if (speed - 1.0).abs() > 1e-6 {
            let current_len = quantized.shape()[2] as i32;
            let new_len = (current_len as f32 / speed) as i32 + 1;
            interpolate_linear(&quantized, new_len)?
        } else {
            quantized
        };

        // TextEncoder forward
        let (_, m_p, logs_p, y_mask) =
            self.enc_p.forward(&quantized, text, ge.as_ref())?;

        // Sample from posterior
        // Clamp logs_p to prevent numerical overflow in exp()
        let logs_p_clamped = maximum(&minimum(&logs_p, &array!(10.0f32))?, &array!(-10.0f32))?;
        let z_p = if noise_scale > 0.0 {
            let noise = random::normal::<f32>(m_p.shape(), None, None, None)?;
            m_p.add(&noise.multiply(&exp(&logs_p_clamped)?)?.multiply(array!(noise_scale))?)?
        } else {
            m_p.clone()
        };

        // Flow reverse
        let z = self.flow.forward(&z_p, &y_mask, ge.as_ref(), true)?;

        // Decode to audio (Python: o = vits.dec(z * y_mask, g=ge))
        let audio = self.dec.forward(&z.multiply(&y_mask)?, ge.as_ref())?;

        Ok(audio)
    }

    /// Training forward pass
    ///
    /// This method performs the full forward pass for training, returning all
    /// intermediate values needed for loss computation.
    ///
    /// Args:
    /// - ssl_features: SSL features from HuBERT [batch, ssl_dim, ssl_len] in NCL
    /// - spec: Linear spectrogram [batch, spec_channels, spec_len] in NCL
    /// - spec_lengths: Spectrogram lengths [batch]
    /// - text: Phoneme indices [batch, text_len]
    /// - refer: Reference mel spectrogram [batch, mel_channels, time] in NCL
    ///
    /// Returns:
    /// - y_hat: Generated audio [batch, 1, segment_samples] (only segment_size frames decoded)
    /// - z_p: Flow-transformed latent [batch, hidden, time]
    /// - m_p: Prior mean from text encoder [batch, hidden, time]
    /// - logs_p: Prior log-variance from text encoder [batch, hidden, time]
    /// - z: Posterior latent [batch, hidden, time]
    /// - m_q: Posterior mean [batch, hidden, time]
    /// - logs_q: Posterior log-variance [batch, hidden, time]
    /// - y_mask: Mask for valid positions [batch, 1, time]
    /// - ids_slice: Random slice start indices [batch] (for slicing real audio)
    /// - commit_loss: VQ commitment loss (scalar) - called "kl_ssl" in Python training
    #[allow(clippy::type_complexity)]
    pub fn forward_train(
        &mut self,
        ssl_features: &Array,
        spec: &Array,
        spec_lengths: &Array,
        text: &Array,
        _refer: &Array,  // Not used in training - ref_enc uses spec instead!
    ) -> Result<(Array, Array, Array, Array, Array, Array, Array, Array, Array, Array), Exception> {
        // Create spectrogram mask from lengths FIRST (needed for ref_enc)
        // spec_lengths: [batch], spec: [batch, channels, time]
        let batch = spec.dim(0);
        let spec_time = spec.dim(2);

        // Create mask: [batch, 1, time]
        // For simplicity, use all-ones mask (assumes all positions are valid)
        // A full implementation would create proper masks from spec_lengths
        let _ = spec_lengths; // Mark as intentionally unused for now
        let y_mask = Array::ones::<f32>(&[batch, 1, spec_time])?;

        // CRITICAL: In training, ref_enc uses the SAME spec as enc_q, not a separate reference!
        // Python: ge = self.ref_enc(y[:,:704] * y_mask, y_mask)
        // where y IS the spec parameter, NOT a separate reference mel!
        let spec_sliced = spec.index((.., ..704, ..));
        let spec_masked = spec_sliced.multiply(&y_mask)?;  // Apply mask like Python
        let ge = self.ref_enc.forward(&spec_masked)?;

        // Encode spectrogram with posterior encoder (enc_q)
        // Returns z (sampled latent), m_q (mean), logs_q (log variance)
        let (z, m_q, logs_q) = self.enc_q.forward(spec, &y_mask, Some(&ge))?;

        // Decode quantized SSL features
        // Step 1: Apply ssl_proj (Conv1d with stride=2 for 25hz)
        let ssl_nlc = swap_axes(ssl_features, 1, 2)?;
        let ssl_proj = self.ssl_proj.forward(&ssl_nlc)?;
        let ssl_proj = swap_axes(&ssl_proj, 1, 2)?; // Back to NCL

        // Step 2: Apply VQ quantization with commitment loss
        // Python: quantized, codes, commit_loss, _ = self.quantizer(ssl, layers=[0])
        // This is the critical step that was missing - kl_ssl in Python training!
        let (ssl_quantized, commit_loss) = self.quantizer.forward_with_loss(&ssl_proj)?;

        // Step 3: For training with 25hz models, interpolate to match spec length
        // The spec is typically 2x the SSL length due to frame rate differences (50hz vs 25hz)
        let z_len = z.dim(2);
        let ssl_len = ssl_quantized.dim(2);
        let quantized = if ssl_len != z_len && ssl_len > 0 {
            // Nearest neighbor interpolation: repeat each frame
            let ratio = z_len / ssl_len;
            if ratio > 0 {
                // Upsample by repeating: add new axis, repeat, reshape
                let expanded = ssl_quantized.index((.., .., .., mlx_rs::ops::indexing::NewAxis));
                let repeated = Array::repeat_axis::<f32>(expanded, ratio, 3)?;
                repeated.reshape(&[batch, self.config.codebook_dim, z_len])?
            } else {
                // Downsample or same length - use as-is
                ssl_quantized
            }
        } else {
            ssl_quantized
        };

        // Encode text+SSL through TextEncoder to get prior
        let (_, m_p, logs_p, _) = self.enc_p.forward(&quantized, text, Some(&ge))?;

        // Apply flow forward (not reverse) on z to get z_p
        // This transforms from posterior to prior space
        let z_p = self.flow.forward(&z, &y_mask, Some(&ge), false)?;

        // Python: z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, segment_size)
        // Randomly slice z for decoding (saves memory, adds augmentation)
        let (z_slice, ids_slice) = rand_slice_segments(
            &z.multiply(&y_mask)?,
            Some(spec_lengths),
            self.config.segment_size,
        )?;

        // Decode z_slice to audio through HiFiGAN generator (only the segment)
        // Python: o = self.dec(z_slice, g=ge)
        let y_hat = self.dec.forward(&z_slice, Some(&ge))?;

        // Return all values including commit_loss (kl_ssl in Python)
        Ok((y_hat, z_p, m_p, logs_p, z, m_q, logs_q, y_mask, ids_slice, commit_loss))
    }

    /// Extract latent codes from SSL features (for reference audio encoding)
    /// Input: ssl_features in NCL format [batch, ssl_dim, time]
    /// Output: projected features in NCL format [batch, ssl_dim, time']
    pub fn extract_latent(&mut self, ssl_features: &Array) -> Result<Array, Exception> {
        // Convert NCL to NLC for Conv1d
        let ssl_nlc = swap_axes(ssl_features, 1, 2)?;
        let ssl = self.ssl_proj.forward(&ssl_nlc)?;
        // Convert back to NCL
        swap_axes(&ssl, 1, 2)
    }

    /// Extract semantic codes from HuBERT features
    ///
    /// This applies ssl_proj then quantizes to semantic codes.
    /// Used for preprocessing training data.
    ///
    /// Args:
    /// - hubert_features: Raw HuBERT features [batch, 768, time] in NCL format
    ///
    /// Returns:
    /// - Semantic codes [batch, 1, time_downsampled]
    ///
    /// Note: For 25hz models, time_downsampled = time / 2
    pub fn extract_semantic_codes(&mut self, hubert_features: &Array) -> Result<Array, Exception> {
        use mlx_rs::transforms::eval;

        // Apply ssl_proj (Conv1d with kernel=2, stride=2 for 25hz)
        // Input: [batch, 768, time] -> convert to [batch, time, 768] for Conv1d
        let ssl_nlc = swap_axes(hubert_features, 1, 2)?;
        let projected_nlc = self.ssl_proj.forward(&ssl_nlc)?;
        // Convert back to NCL: [batch, 768, time/2]
        let projected = swap_axes(&projected_nlc, 1, 2)?;
        eval([&projected])?;

        // Quantize to semantic codes
        let codes = self.quantizer.encode(&projected)?;
        eval([&codes])?;

        Ok(codes)
    }
}

// ============================================================================
// Weight Loading
// ============================================================================

/// Compute weight from weight normalization components.
/// Weight normalization: weight = g * v / ||v||
/// g: [out_channels, 1, 1]
/// v: [out_channels, in_channels, kernel_size]
fn weight_norm_conv(g: &Array, v: &Array) -> Result<Array, Exception> {
    use mlx_rs::transforms::eval;

    // Compute L2 norm of v along in_channels and kernel dimensions
    // v shape: [out, in, kernel]
    let v_squared = v.square()?;
    // Sum along last two dims, keep dims for broadcasting
    let norm_sq = v_squared.sum_axes(&[-2, -1], true)?;
    let norm = sqrt(&norm_sq.add(array!(1e-12f32))?)?;

    // weight = g * v / norm
    let weight = g.multiply(v)?.divide(&norm)?;
    eval([&weight])?;
    Ok(weight)
}

/// Compute weight from weight normalization for ConvTranspose.
/// g: [in_channels, 1, 1]
/// v: [in_channels, out_channels, kernel_size]
fn weight_norm_convt(g: &Array, v: &Array) -> Result<Array, Exception> {
    use mlx_rs::transforms::eval;

    // Compute L2 norm of v along out_channels and kernel dimensions
    let v_squared = v.square()?;
    let norm_sq = v_squared.sum_axes(&[-2, -1], true)?;
    let norm = sqrt(&norm_sq.add(array!(1e-12f32))?)?;

    // weight = g * v / norm
    let weight = g.multiply(v)?.divide(&norm)?;
    eval([&weight])?;
    Ok(weight)
}

/// Load VITS/SynthesizerTrn weights from safetensors
pub fn load_vits_weights(
    model: &mut SynthesizerTrn,
    weights: &HashMap<String, Array>,
) -> Result<(), Error> {
    let get_weight = |key: &str| -> Option<Array> { weights.get(key).cloned() };

    // Helper to transpose Conv1d weights from PyTorch [out, in, kernel] to mlx-rs [out, kernel, in]
    let transpose_conv = |w: Array| -> Result<Array, Exception> { swap_axes(&w, 1, 2) };

    // Helper to transpose ConvTranspose1d weights from PyTorch [in, out, kernel] to mlx-rs [out, kernel, in]
    let transpose_convt = |w: Array| -> Result<Array, Exception> {
        let w = swap_axes(&w, 0, 1)?; // [out, in, kernel]
        swap_axes(&w, 1, 2) // [out, kernel, in]
    };

    // Helper to load weight-normalized Conv1d
    // Returns transposed weight ready for mlx-rs
    let load_weight_norm_conv = |prefix: &str| -> Option<Result<Array, Exception>> {
        let g = weights.get(&format!("{}.weight_g", prefix))?;
        let v = weights.get(&format!("{}.weight_v", prefix))?;
        Some(weight_norm_conv(g, v).and_then(|w| transpose_conv(w)))
    };

    // Helper to load weight-normalized ConvTranspose1d
    let load_weight_norm_convt = |prefix: &str| -> Option<Result<Array, Exception>> {
        let g = weights.get(&format!("{}.weight_g", prefix))?;
        let v = weights.get(&format!("{}.weight_v", prefix))?;
        Some(weight_norm_convt(g, v).and_then(|w| transpose_convt(w)))
    };

    // Quantizer codebook
    if let Some(w) = get_weight("quantizer.vq.layers.0._codebook.embed") {
        model.quantizer.embed = Param::new(w);
    }

    // SSL projection
    if let Some(w) = get_weight("ssl_proj.weight") {
        model.ssl_proj.weight = Param::new(transpose_conv(w)?);
    }
    if let Some(b) = get_weight("ssl_proj.bias") {
        model.ssl_proj.bias = Param::new(Some(b));
    }

    // PosteriorEncoder (enc_q) - for training
    if let Some(w) = get_weight("enc_q.pre.weight") {
        model.enc_q.pre.weight = Param::new(transpose_conv(w)?);
    }
    if let Some(b) = get_weight("enc_q.pre.bias") {
        model.enc_q.pre.bias = Param::new(Some(b));
    }
    if let Some(w) = get_weight("enc_q.proj.weight") {
        model.enc_q.proj.weight = Param::new(transpose_conv(w)?);
    }
    if let Some(b) = get_weight("enc_q.proj.bias") {
        model.enc_q.proj.bias = Param::new(Some(b));
    }

    // enc_q WN encoder - try weight normalization first, fall back to regular
    let enc_q_cond_prefix = "enc_q.enc.cond_layer";
    if let Some(w_result) = load_weight_norm_conv(enc_q_cond_prefix) {
        model.enc_q.enc.cond_layer.weight = Param::new(w_result?);
    } else if let Some(w) = get_weight(&format!("{}.weight", enc_q_cond_prefix)) {
        model.enc_q.enc.cond_layer.weight = Param::new(transpose_conv(w)?);
    }
    if let Some(b) = get_weight(&format!("{}.bias", enc_q_cond_prefix)) {
        model.enc_q.enc.cond_layer.bias = Param::new(Some(b));
    }

    for j in 0..model.enc_q.enc.in_layers.len() {
        // in_layers
        let in_prefix = format!("enc_q.enc.in_layers.{}", j);
        if let Some(w_result) = load_weight_norm_conv(&in_prefix) {
            model.enc_q.enc.in_layers[j].weight = Param::new(w_result?);
        } else if let Some(w) = get_weight(&format!("{}.weight", in_prefix)) {
            model.enc_q.enc.in_layers[j].weight = Param::new(transpose_conv(w)?);
        }
        if let Some(b) = get_weight(&format!("{}.bias", in_prefix)) {
            model.enc_q.enc.in_layers[j].bias = Param::new(Some(b));
        }

        // res_skip_layers
        let skip_prefix = format!("enc_q.enc.res_skip_layers.{}", j);
        if let Some(w_result) = load_weight_norm_conv(&skip_prefix) {
            model.enc_q.enc.res_skip_layers[j].weight = Param::new(w_result?);
        } else if let Some(w) = get_weight(&format!("{}.weight", skip_prefix)) {
            model.enc_q.enc.res_skip_layers[j].weight = Param::new(transpose_conv(w)?);
        }
        if let Some(b) = get_weight(&format!("{}.bias", skip_prefix)) {
            model.enc_q.enc.res_skip_layers[j].bias = Param::new(Some(b));
        }
    }

    // TextEncoder (enc_p)
    if let Some(w) = get_weight("enc_p.ssl_proj.weight") {
        model.enc_p.ssl_proj.weight = Param::new(transpose_conv(w)?);
    }
    if let Some(b) = get_weight("enc_p.ssl_proj.bias") {
        model.enc_p.ssl_proj.bias = Param::new(Some(b));
    }

    if let Some(w) = get_weight("enc_p.text_embedding.weight") {
        model.enc_p.text_embedding.weight = Param::new(w);
    }

    if let Some(w) = get_weight("enc_p.proj.weight") {
        model.enc_p.proj.weight = Param::new(transpose_conv(w)?);
    }
    if let Some(b) = get_weight("enc_p.proj.bias") {
        model.enc_p.proj.bias = Param::new(Some(b));
    }

    // MRTE
    if let Some(w) = get_weight("enc_p.mrte.c_pre.weight") {
        model.enc_p.mrte.c_pre.weight = Param::new(transpose_conv(w)?);
    }
    if let Some(b) = get_weight("enc_p.mrte.c_pre.bias") {
        model.enc_p.mrte.c_pre.bias = Param::new(Some(b));
    }
    if let Some(w) = get_weight("enc_p.mrte.c_post.weight") {
        model.enc_p.mrte.c_post.weight = Param::new(transpose_conv(w)?);
    }
    if let Some(b) = get_weight("enc_p.mrte.c_post.bias") {
        model.enc_p.mrte.c_post.bias = Param::new(Some(b));
    }
    if let Some(w) = get_weight("enc_p.mrte.text_pre.weight") {
        model.enc_p.mrte.text_pre.weight = Param::new(transpose_conv(w)?);
    }
    if let Some(b) = get_weight("enc_p.mrte.text_pre.bias") {
        model.enc_p.mrte.text_pre.bias = Param::new(Some(b));
    }

    // MRTE cross attention
    if let Some(w) = get_weight("enc_p.mrte.cross_attention.conv_q.weight") {
        model.enc_p.mrte.cross_attention.conv_q.weight = Param::new(transpose_conv(w)?);
    }
    if let Some(b) = get_weight("enc_p.mrte.cross_attention.conv_q.bias") {
        model.enc_p.mrte.cross_attention.conv_q.bias = Param::new(Some(b));
    }
    if let Some(w) = get_weight("enc_p.mrte.cross_attention.conv_k.weight") {
        model.enc_p.mrte.cross_attention.conv_k.weight = Param::new(transpose_conv(w)?);
    }
    if let Some(b) = get_weight("enc_p.mrte.cross_attention.conv_k.bias") {
        model.enc_p.mrte.cross_attention.conv_k.bias = Param::new(Some(b));
    }
    if let Some(w) = get_weight("enc_p.mrte.cross_attention.conv_v.weight") {
        model.enc_p.mrte.cross_attention.conv_v.weight = Param::new(transpose_conv(w)?);
    }
    if let Some(b) = get_weight("enc_p.mrte.cross_attention.conv_v.bias") {
        model.enc_p.mrte.cross_attention.conv_v.bias = Param::new(Some(b));
    }
    if let Some(w) = get_weight("enc_p.mrte.cross_attention.conv_o.weight") {
        model.enc_p.mrte.cross_attention.conv_o.weight = Param::new(transpose_conv(w)?);
    }
    if let Some(b) = get_weight("enc_p.mrte.cross_attention.conv_o.bias") {
        model.enc_p.mrte.cross_attention.conv_o.bias = Param::new(Some(b));
    }

    // Helper to load transformer encoder weights
    let load_encoder_weights = |encoder: &mut TransformerEncoder,
                                prefix: &str,
                                weights: &HashMap<String, Array>|
     -> Result<(), Error> {
        for (i, layer) in encoder.layers.iter_mut().enumerate() {
            // Attention layers
            if let Some(w) = weights.get(&format!("{}.attn_layers.{}.conv_q.weight", prefix, i)) {
                layer.attn.conv_q.weight = Param::new(transpose_conv(w.clone())?);
            }
            if let Some(b) = weights.get(&format!("{}.attn_layers.{}.conv_q.bias", prefix, i)) {
                layer.attn.conv_q.bias = Param::new(Some(b.clone()));
            }
            if let Some(w) = weights.get(&format!("{}.attn_layers.{}.conv_k.weight", prefix, i)) {
                layer.attn.conv_k.weight = Param::new(transpose_conv(w.clone())?);
            }
            if let Some(b) = weights.get(&format!("{}.attn_layers.{}.conv_k.bias", prefix, i)) {
                layer.attn.conv_k.bias = Param::new(Some(b.clone()));
            }
            if let Some(w) = weights.get(&format!("{}.attn_layers.{}.conv_v.weight", prefix, i)) {
                layer.attn.conv_v.weight = Param::new(transpose_conv(w.clone())?);
            }
            if let Some(b) = weights.get(&format!("{}.attn_layers.{}.conv_v.bias", prefix, i)) {
                layer.attn.conv_v.bias = Param::new(Some(b.clone()));
            }
            if let Some(w) = weights.get(&format!("{}.attn_layers.{}.conv_o.weight", prefix, i)) {
                layer.attn.conv_o.weight = Param::new(transpose_conv(w.clone())?);
            }
            if let Some(b) = weights.get(&format!("{}.attn_layers.{}.conv_o.bias", prefix, i)) {
                layer.attn.conv_o.bias = Param::new(Some(b.clone()));
            }

            // Relative position embeddings
            if let Some(emb) = weights.get(&format!("{}.attn_layers.{}.emb_rel_k", prefix, i)) {
                layer.attn.emb_rel_k = Param::new(emb.clone());
            }
            if let Some(emb) = weights.get(&format!("{}.attn_layers.{}.emb_rel_v", prefix, i)) {
                layer.attn.emb_rel_v = Param::new(emb.clone());
            }

            // FFN layers
            if let Some(w) = weights.get(&format!("{}.ffn_layers.{}.conv_1.weight", prefix, i)) {
                layer.ffn.conv_1.weight = Param::new(transpose_conv(w.clone())?);
            }
            if let Some(b) = weights.get(&format!("{}.ffn_layers.{}.conv_1.bias", prefix, i)) {
                layer.ffn.conv_1.bias = Param::new(Some(b.clone()));
            }
            if let Some(w) = weights.get(&format!("{}.ffn_layers.{}.conv_2.weight", prefix, i)) {
                layer.ffn.conv_2.weight = Param::new(transpose_conv(w.clone())?);
            }
            if let Some(b) = weights.get(&format!("{}.ffn_layers.{}.conv_2.bias", prefix, i)) {
                layer.ffn.conv_2.bias = Param::new(Some(b.clone()));
            }

            // Layer norms
            if let Some(g) = weights.get(&format!("{}.norm_layers_1.{}.gamma", prefix, i)) {
                layer.norm1.gamma = Param::new(g.clone());
            }
            if let Some(b) = weights.get(&format!("{}.norm_layers_1.{}.beta", prefix, i)) {
                layer.norm1.beta = Param::new(b.clone());
            }
            if let Some(g) = weights.get(&format!("{}.norm_layers_2.{}.gamma", prefix, i)) {
                layer.norm2.gamma = Param::new(g.clone());
            }
            if let Some(b) = weights.get(&format!("{}.norm_layers_2.{}.beta", prefix, i)) {
                layer.norm2.beta = Param::new(b.clone());
            }
        }
        Ok(())
    };

    // Load encoder_ssl weights
    load_encoder_weights(&mut model.enc_p.encoder_ssl, "enc_p.encoder_ssl", weights)?;

    // Load encoder_text weights
    load_encoder_weights(&mut model.enc_p.encoder_text, "enc_p.encoder_text", weights)?;

    // Load encoder2 weights
    load_encoder_weights(&mut model.enc_p.encoder2, "enc_p.encoder2", weights)?;

    // Flow layers
    for i in [0, 2, 4, 6].iter() {
        let flow_idx = *i / 2;
        if flow_idx < model.flow.flows.len() {
            let flow = &mut model.flow.flows[flow_idx];

            if let Some(w) = get_weight(&format!("flow.flows.{}.pre.weight", i)) {
                flow.pre.weight = Param::new(transpose_conv(w)?);
            }
            if let Some(b) = get_weight(&format!("flow.flows.{}.pre.bias", i)) {
                flow.pre.bias = Param::new(Some(b));
            }
            if let Some(w) = get_weight(&format!("flow.flows.{}.post.weight", i)) {
                flow.post.weight = Param::new(transpose_conv(w)?);
            }
            if let Some(b) = get_weight(&format!("flow.flows.{}.post.bias", i)) {
                flow.post.bias = Param::new(Some(b));
            }

            // WN encoder - try weight normalization first, fall back to regular
            let cond_prefix = format!("flow.flows.{}.enc.cond_layer", i);
            if let Some(w_result) = load_weight_norm_conv(&cond_prefix) {
                flow.enc.cond_layer.weight = Param::new(w_result?);
            } else if let Some(w) = get_weight(&format!("{}.weight", cond_prefix)) {
                flow.enc.cond_layer.weight = Param::new(transpose_conv(w)?);
            }
            if let Some(b) = get_weight(&format!("{}.bias", cond_prefix)) {
                flow.enc.cond_layer.bias = Param::new(Some(b));
            }

            for j in 0..flow.enc.in_layers.len() {
                // in_layers - try weight normalization first, fall back to regular
                let in_prefix = format!("flow.flows.{}.enc.in_layers.{}", i, j);
                if let Some(w_result) = load_weight_norm_conv(&in_prefix) {
                    flow.enc.in_layers[j].weight = Param::new(w_result?);
                } else if let Some(w) = get_weight(&format!("{}.weight", in_prefix)) {
                    flow.enc.in_layers[j].weight = Param::new(transpose_conv(w)?);
                }
                if let Some(b) = get_weight(&format!("{}.bias", in_prefix)) {
                    flow.enc.in_layers[j].bias = Param::new(Some(b));
                }

                // res_skip_layers - try weight normalization first, fall back to regular
                let skip_prefix = format!("flow.flows.{}.enc.res_skip_layers.{}", i, j);
                if let Some(w_result) = load_weight_norm_conv(&skip_prefix) {
                    flow.enc.res_skip_layers[j].weight = Param::new(w_result?);
                } else if let Some(w) = get_weight(&format!("{}.weight", skip_prefix)) {
                    flow.enc.res_skip_layers[j].weight = Param::new(transpose_conv(w)?);
                }
                if let Some(b) = get_weight(&format!("{}.bias", skip_prefix)) {
                    flow.enc.res_skip_layers[j].bias = Param::new(Some(b));
                }
            }
        }
    }

    // HiFiGAN Generator (dec) - with weight normalization
    // conv_pre: load weight_g and weight_v separately
    if let Some(g) = get_weight("dec.conv_pre.weight_g") {
        model.dec.conv_pre.weight_g = Param::new(g);
    }
    if let Some(v) = get_weight("dec.conv_pre.weight_v") {
        // Transpose v from PyTorch [out, in, kernel] to MLX [out, kernel, in]
        model.dec.conv_pre.weight_v = Param::new(transpose_conv(v)?);
    } else if let Some(w) = get_weight("dec.conv_pre.weight") {
        // Fallback: load merged weight and decompose into g and v
        let w_t = transpose_conv(w)?;
        // Initialize weight_v = w, weight_g = ||w|| per output channel
        let w_squared = w_t.square()?;
        let norm_sq = w_squared.sum_axes(&[1, 2], true)?;
        let weight_g = sqrt(&norm_sq.add(array!(1e-12f32))?)?;
        model.dec.conv_pre.weight_g = Param::new(weight_g);
        model.dec.conv_pre.weight_v = Param::new(w_t);
    }
    if let Some(b) = get_weight("dec.conv_pre.bias") {
        model.dec.conv_pre.bias = Param::new(Some(b));
    }

    // conv_post: load weight_g and weight_v separately
    if let Some(g) = get_weight("dec.conv_post.weight_g") {
        model.dec.conv_post.weight_g = Param::new(g);
    }
    if let Some(v) = get_weight("dec.conv_post.weight_v") {
        model.dec.conv_post.weight_v = Param::new(transpose_conv(v)?);
    } else if let Some(w) = get_weight("dec.conv_post.weight") {
        // Fallback: load merged weight and decompose
        let w_t = transpose_conv(w)?;
        let w_squared = w_t.square()?;
        let norm_sq = w_squared.sum_axes(&[1, 2], true)?;
        let weight_g = sqrt(&norm_sq.add(array!(1e-12f32))?)?;
        model.dec.conv_post.weight_g = Param::new(weight_g);
        model.dec.conv_post.weight_v = Param::new(w_t);
    }

    // cond layer (not weight normalized)
    if let Some(w) = get_weight("dec.cond.weight") {
        model.dec.cond.weight = Param::new(transpose_conv(w)?);
    }
    if let Some(b) = get_weight("dec.cond.bias") {
        model.dec.cond.bias = Param::new(Some(b));
    }

    // Upsample layers (ConvTranspose1d) - load weight_g and weight_v separately
    for (i, up) in model.dec.ups.iter_mut().enumerate() {
        let prefix = format!("dec.ups.{}", i);

        // Try to load weight_g and weight_v separately
        if let Some(g) = get_weight(&format!("{}.weight_g", prefix)) {
            up.weight_g = Param::new(g);
        }
        if let Some(v) = get_weight(&format!("{}.weight_v", prefix)) {
            // Transpose v from PyTorch [in, out, kernel] to MLX [out, kernel, in]
            up.weight_v = Param::new(transpose_convt(v)?);
        } else if let Some(w) = get_weight(&format!("{}.weight", prefix)) {
            // Fallback: load merged weight and decompose
            let w_t = transpose_convt(w)?;
            // For ConvTranspose, norm over out and kernel dims (axes 0, 1)
            let w_squared = w_t.square()?;
            let norm_sq = w_squared.sum_axes(&[0, 1], true)?;
            // Transpose to [in, 1, 1] shape for weight_g
            let weight_g = sqrt(&norm_sq.add(array!(1e-12f32))?)?.transpose_axes(&[2, 0, 1])?;
            up.weight_g = Param::new(weight_g);
            up.weight_v = Param::new(w_t);
        }
        if let Some(b) = get_weight(&format!("{}.bias", prefix)) {
            up.bias = Param::new(Some(b));
        }
    }

    // ResBlocks - try weight normalization first, fall back to regular
    for (i, rb) in model.dec.resblocks.iter_mut().enumerate() {
        for (j, conv) in rb.convs1.iter_mut().enumerate() {
            let prefix = format!("dec.resblocks.{}.convs1.{}", i, j);
            if let Some(w_result) = load_weight_norm_conv(&prefix) {
                conv.weight = Param::new(w_result?);
            } else if let Some(w) = get_weight(&format!("{}.weight", prefix)) {
                conv.weight = Param::new(transpose_conv(w)?);
            }
            if let Some(b) = get_weight(&format!("{}.bias", prefix)) {
                conv.bias = Param::new(Some(b));
            }
        }
        for (j, conv) in rb.convs2.iter_mut().enumerate() {
            let prefix = format!("dec.resblocks.{}.convs2.{}", i, j);
            if let Some(w_result) = load_weight_norm_conv(&prefix) {
                conv.weight = Param::new(w_result?);
            } else if let Some(w) = get_weight(&format!("{}.weight", prefix)) {
                conv.weight = Param::new(transpose_conv(w)?);
            }
            if let Some(b) = get_weight(&format!("{}.bias", prefix)) {
                conv.bias = Param::new(Some(b));
            }
        }
    }

    // MelStyleEncoder (ref_enc)
    if let Some(w) = get_weight("ref_enc.spectral.0.fc.weight") {
        model.ref_enc.spectral_0.weight = Param::new(w);
    }
    if let Some(b) = get_weight("ref_enc.spectral.0.fc.bias") {
        model.ref_enc.spectral_0.bias = Param::new(Some(b));
    }
    if let Some(w) = get_weight("ref_enc.spectral.3.fc.weight") {
        model.ref_enc.spectral_1.weight = Param::new(w);
    }
    if let Some(b) = get_weight("ref_enc.spectral.3.fc.bias") {
        model.ref_enc.spectral_1.bias = Param::new(Some(b));
    }

    if let Some(w) = get_weight("ref_enc.temporal.0.conv1.conv.weight") {
        model.ref_enc.temporal_0.weight = Param::new(transpose_conv(w)?);
    }
    if let Some(b) = get_weight("ref_enc.temporal.0.conv1.conv.bias") {
        model.ref_enc.temporal_0.bias = Param::new(Some(b));
    }
    if let Some(w) = get_weight("ref_enc.temporal.1.conv1.conv.weight") {
        model.ref_enc.temporal_1.weight = Param::new(transpose_conv(w)?);
    }
    if let Some(b) = get_weight("ref_enc.temporal.1.conv1.conv.bias") {
        model.ref_enc.temporal_1.bias = Param::new(Some(b));
    }

    if let Some(w) = get_weight("ref_enc.slf_attn.w_qs.weight") {
        model.ref_enc.slf_attn_q.weight = Param::new(w);
    }
    if let Some(b) = get_weight("ref_enc.slf_attn.w_qs.bias") {
        model.ref_enc.slf_attn_q.bias = Param::new(Some(b));
    }
    if let Some(w) = get_weight("ref_enc.slf_attn.w_ks.weight") {
        model.ref_enc.slf_attn_k.weight = Param::new(w);
    }
    if let Some(b) = get_weight("ref_enc.slf_attn.w_ks.bias") {
        model.ref_enc.slf_attn_k.bias = Param::new(Some(b));
    }
    if let Some(w) = get_weight("ref_enc.slf_attn.w_vs.weight") {
        model.ref_enc.slf_attn_v.weight = Param::new(w);
    }
    if let Some(b) = get_weight("ref_enc.slf_attn.w_vs.bias") {
        model.ref_enc.slf_attn_v.bias = Param::new(Some(b));
    }
    if let Some(w) = get_weight("ref_enc.slf_attn.fc.weight") {
        model.ref_enc.slf_attn_fc.weight = Param::new(w);
    }
    if let Some(b) = get_weight("ref_enc.slf_attn.fc.bias") {
        model.ref_enc.slf_attn_fc.bias = Param::new(Some(b));
    }

    if let Some(w) = get_weight("ref_enc.fc.fc.weight") {
        model.ref_enc.fc.weight = Param::new(w);
    }
    if let Some(b) = get_weight("ref_enc.fc.fc.bias") {
        model.ref_enc.fc.bias = Param::new(Some(b));
    }

    Ok(())
}

/// Load VITS model from safetensors file
pub fn load_vits_model(weights_path: impl AsRef<Path>) -> Result<SynthesizerTrn, Error> {
    let path = weights_path.as_ref();

    // Load weights first to detect semantic_frame_rate
    // Convert float16 → float32 for numerical stability (attention softmax, layer norm, flow)
    let raw_weights = Array::load_safetensors(path)?;
    let weights: HashMap<String, Array> = raw_weights
        .into_iter()
        .map(|(k, v)| {
            let v32 = v.as_type::<f32>().unwrap_or(v);
            (k, v32)
        })
        .collect();

    // Auto-detect semantic_frame_rate from ssl_proj.weight shape
    // - 25hz models: ssl_proj.weight is [768, 768, 2] (kernel=2, after transpose)
    // - 50hz models: ssl_proj.weight is [768, 768, 1] (kernel=1, after transpose)
    // The raw PyTorch weight is [out, in, kernel] = [768, 768, kernel]
    let semantic_frame_rate = if let Some(ssl_proj_weight) = weights.get("ssl_proj.weight") {
        let shape = ssl_proj_weight.shape();
        // Shape is [out_channels, in_channels, kernel_size]
        let kernel_size = if shape.len() == 3 { shape[2] } else { 1 };
        if kernel_size == 2 {
            "25hz".to_string()
        } else {
            "50hz".to_string()
        }
    } else {
        // Default to 50hz if weight not found
        "50hz".to_string()
    };

    let config = VITSConfig {
        semantic_frame_rate,
        ..VITSConfig::default()
    };
    let mut model = SynthesizerTrn::new(config)?;

    load_vits_weights(&mut model, &weights)?;

    Ok(model)
}

/// Load VITS model with finetuned weights overlaid on pretrained base.
///
/// Finetuned weights only contain trainable layers (dec, ref_enc, ssl_proj),
/// so we first load the full pretrained model, then overlay finetuned weights.
pub fn load_vits_model_with_finetuned(
    pretrained_path: impl AsRef<Path>,
    finetuned_path: impl AsRef<Path>,
) -> Result<SynthesizerTrn, Error> {
    // First load the full pretrained model
    let mut model = load_vits_model(&pretrained_path)?;

    // Then overlay finetuned weights (only dec, ref_enc, ssl_proj will be present)
    let raw_ft = Array::load_safetensors(finetuned_path.as_ref())?;
    let finetuned_weights: HashMap<String, Array> = raw_ft
        .into_iter()
        .map(|(k, v)| {
            let v32 = v.as_type::<f32>().unwrap_or(v);
            (k, v32)
        })
        .collect();
    load_vits_weights(&mut model, &finetuned_weights)?;

    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::transforms::eval;

    #[test]
    fn test_rvq_codebook() {
        let codebook = RVQCodebook::new(1024, 768).unwrap();
        let codes = Array::zeros::<i32>(&[1, 1, 10]).unwrap();
        let quantized = codebook.decode(&codes).unwrap();
        eval([&quantized]).unwrap();
        assert_eq!(quantized.shape(), &[1, 768, 10]);
    }

    #[test]
    fn test_vits_config() {
        let config = VITSConfig::default();
        assert_eq!(config.hidden_channels, 192);
        assert_eq!(config.gin_channels, 512);
    }

    #[test]
    fn test_interpolate_linear_same_size() {
        // speed = 1.0 -> same size
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5]);
        let result = interpolate_linear(&x, 5).unwrap();
        eval([&result]).unwrap();
        assert_eq!(result.shape(), &[1, 1, 5]);
    }

    #[test]
    fn test_interpolate_linear_downsample() {
        // speed = 2.0 -> half the length (faster speech)
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let result = interpolate_linear(&x, 2).unwrap();
        eval([&result]).unwrap();
        assert_eq!(result.shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_interpolate_linear_upsample() {
        // speed = 0.5 -> double the length (slower speech)
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let result = interpolate_linear(&x, 8).unwrap();
        eval([&result]).unwrap();
        assert_eq!(result.shape(), &[1, 1, 8]);
    }

    #[test]
    fn test_interpolate_linear_speed_1_1() {
        // speed = 1.1 -> ~10% shorter (typical for Chinese voices)
        let x = Array::from_slice(&[1.0f32; 100], &[1, 1, 100]);
        // new_len = 100 / 1.1 + 1 = 91 + 1 = 92
        let new_len = (100.0 / 1.1) as i32 + 1;
        let result = interpolate_linear(&x, new_len).unwrap();
        eval([&result]).unwrap();
        assert_eq!(result.shape(), &[1, 1, new_len]);
    }

    #[test]
    fn test_interpolate_linear_batch() {
        // Test with batch > 1 and channels > 1
        let x = Array::from_slice(&[1.0f32; 24], &[2, 3, 4]); // [batch=2, channels=3, seq=4]
        let result = interpolate_linear(&x, 6).unwrap();
        eval([&result]).unwrap();
        assert_eq!(result.shape(), &[2, 3, 6]);
    }
}
