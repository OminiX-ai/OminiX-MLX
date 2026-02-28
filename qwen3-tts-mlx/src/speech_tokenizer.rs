//! Speech tokenizer decoder: converts 16-codebook discrete codes to 24kHz waveform.

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::{
    array,
    builder::Builder,
    module::{Module, Param},
    nn,
    ops::{
        arange, concatenate_axis,
        indexing::{IndexOp, NewAxis},
        zeros,
    },
    transforms::eval,
    Array,
};

use crate::config::DecoderConfig;
use crate::error::{Error, Result};

// ============================================================================
// Helper: Causal Conv1d (left-padding)
// ============================================================================

pub struct CausalConv1d {
    pub conv: nn::Conv1d,
    pub pad: i32,
}

impl CausalConv1d {
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        // x: [B, T, C] - pad left by (kernel_size - 1) * dilation
        let x = if self.pad > 0 {
            // Manual left-padding: concat zeros on left along time axis
            let b = x.dim(0) as i32;
            let c = x.dim(2) as i32;
            let pad_zeros = zeros::<f32>(&[b, self.pad, c])?;
            concatenate_axis(&[&pad_zeros, x], 1)?
        } else {
            x.clone()
        };
        Ok(self.conv.forward(&x)?)
    }
}

// ============================================================================
// Helper: Causal ConvTranspose1d
// ============================================================================

pub struct CausalConvTranspose1d {
    pub conv_t: nn::ConvTranspose1d,
    pub trim_right: i32,
}

impl CausalConvTranspose1d {
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let mut y = self.conv_t.forward(&x)?;
        // Trim right to maintain causal property
        if self.trim_right > 0 {
            let t = y.dim(1) as i32;
            let keep = t - self.trim_right;
            if keep > 0 {
                y = y.index((.., ..keep, ..));
            }
        }
        Ok(y)
    }
}

// ============================================================================
// SnakeBeta activation: x + (1/beta) * sin^2(alpha * x)
// ============================================================================

pub struct SnakeBeta {
    pub alpha: Array, // [C]
    pub beta: Array,  // [C]
}

impl SnakeBeta {
    pub fn forward(&self, x: &Array) -> Result<Array> {
        // x: [B, T, C], alpha/beta stored in LOG SPACE: [C] → exp() then broadcast via [1, 1, C]
        // Formula: x + (1 / (exp(beta) + 1e-9)) * sin^2(exp(alpha) * x)
        let alpha = self.alpha.reshape(&[1, 1, -1])?.exp()?;
        let beta = self.beta.reshape(&[1, 1, -1])?.exp()?;

        let sin_val = (x.multiply(&alpha)?).sin()?;
        let sin_sq = sin_val.multiply(&sin_val)?;
        let recip_beta = beta.add(array!(1e-9f32))?.reciprocal()?;
        Ok(x.add(sin_sq.multiply(&recip_beta)?)?)
    }
}

// ============================================================================
// Residual Unit
// ============================================================================

pub struct ResidualUnit {
    pub act1: SnakeBeta,
    pub conv1: CausalConv1d,
    pub act2: SnakeBeta,
    pub conv2: CausalConv1d,
}

impl ResidualUnit {
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let h = self.act1.forward(x)?;
        SpeechTokenizerDecoder::debug_tensor("    ru.act1", &h);
        let h = self.conv1.forward(&h)?;
        SpeechTokenizerDecoder::debug_tensor("    ru.conv1", &h);
        let h = self.act2.forward(&h)?;
        SpeechTokenizerDecoder::debug_tensor("    ru.act2", &h);
        let h = self.conv2.forward(&h)?;
        SpeechTokenizerDecoder::debug_tensor("    ru.conv2", &h);
        Ok(x.add(h)?)
    }
}

// ============================================================================
// Decoder Block: SnakeBeta → ConvTranspose1d → 3 ResidualUnits
// ============================================================================

pub struct DecoderBlock {
    pub snake: SnakeBeta,
    pub conv_t: CausalConvTranspose1d,
    pub res_units: Vec<ResidualUnit>,
}

impl DecoderBlock {
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let mut h = self.snake.forward(x)?;
        SpeechTokenizerDecoder::debug_tensor("  block.snake", &h);
        h = self.conv_t.forward(&h)?;
        SpeechTokenizerDecoder::debug_tensor("  block.conv_t", &h);
        for (i, ru) in self.res_units.iter_mut().enumerate() {
            h = ru.forward(&h)?;
            SpeechTokenizerDecoder::debug_tensor(&format!("  block.res_unit_{i}"), &h);
        }
        Ok(h)
    }
}

// ============================================================================
// ConvNeXt Block
// ============================================================================

pub struct ConvNeXtBlock {
    pub dwconv: CausalConv1d,
    pub norm_weight: Array,
    pub norm_bias: Array,
    pub pwconv1_weight: Array,
    pub pwconv1_bias: Array,
    pub pwconv2_weight: Array,
    pub pwconv2_bias: Array,
    pub gamma: Array,
}

impl ConvNeXtBlock {
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let residual = x.clone();

        // Depthwise conv
        let mut h = self.dwconv.forward(x)?;

        // Layer norm along last dim
        let mean = h.mean_axis(-1, true)?;
        let var = h.var_axis(-1, true, None)?;
        let norm_weight = self.norm_weight.reshape(&[1, 1, -1])?;
        let norm_bias = self.norm_bias.reshape(&[1, 1, -1])?;
        let inv_std = var.add(array!(1e-5f32))?.rsqrt()?;
        h = h
            .subtract(&mean)?
            .multiply(&inv_std)?
            .multiply(&norm_weight)?
            .add(&norm_bias)?;

        // Pointwise MLP (implemented as matmul since these are Linear weights)
        h = h
            .matmul(&self.pwconv1_weight.t())?
            .add(&self.pwconv1_bias)?;
        h = nn::gelu(h)?;
        h = h
            .matmul(&self.pwconv2_weight.t())?
            .add(&self.pwconv2_bias)?;

        // Layer scale
        let gamma = self.gamma.reshape(&[1, 1, -1])?;
        h = h.multiply(&gamma)?;

        Ok(residual.add(h)?)
    }
}

// ============================================================================
// Decoder Transformer Layer (with LayerScale)
// ============================================================================

pub struct DecoderTransformerLayer {
    pub input_layernorm: nn::RmsNorm,
    pub q_proj: nn::Linear,
    pub k_proj: nn::Linear,
    pub v_proj: nn::Linear,
    pub o_proj: nn::Linear,
    pub attn_layer_scale: Array,
    pub post_attention_layernorm: nn::RmsNorm,
    pub gate_proj: nn::Linear,
    pub up_proj: nn::Linear,
    pub down_proj: nn::Linear,
    pub mlp_layer_scale: Array,

    pub n_heads: i32,
    pub head_dim: i32,
    pub rope: nn::Rope,
}

impl DecoderTransformerLayer {
    #[allow(non_snake_case)]
    pub fn forward(&mut self, x: &Array, mask: Option<&Array>, offset: i32) -> Result<Array> {
        let B = x.dim(0) as i32;
        let L = x.dim(1) as i32;
        let scale = (self.head_dim as f32).sqrt().recip();

        let normed = self.input_layernorm.forward(x)?;
        let q = self.q_proj.forward(&normed)?
            .reshape(&[B, L, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = self.k_proj.forward(&normed)?
            .reshape(&[B, L, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = self.v_proj.forward(&normed)?
            .reshape(&[B, L, self.n_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let q = self.rope.forward(nn::RopeInputBuilder::new(&q).offset(offset).build().unwrap())?;
        let k = self.rope.forward(nn::RopeInputBuilder::new(&k).offset(offset).build().unwrap())?;

        let attn_out = mlx_rs::fast::scaled_dot_product_attention(
            q, k, v, scale,
            mask.map(mlx_rs::fast::ScaledDotProductAttentionMask::Array),
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[B, L, -1])?;

        let attn_out = self.o_proj.forward(&attn_out)?;
        let attn_scale = self.attn_layer_scale.reshape(&[1, 1, -1])?;
        let attn_out = attn_out.multiply(&attn_scale)?;
        let h = x.add(attn_out)?;

        let normed = self.post_attention_layernorm.forward(&h)?;
        let gate = nn::silu(self.gate_proj.forward(&normed)?)?;
        let up = self.up_proj.forward(&normed)?;
        let mlp_out = self.down_proj.forward(&gate.multiply(up)?)?;
        let mlp_scale = self.mlp_layer_scale.reshape(&[1, 1, -1])?;
        let mlp_out = mlp_out.multiply(&mlp_scale)?;

        Ok(h.add(mlp_out)?)
    }
}

// ============================================================================
// Full Speech Tokenizer Decoder
// ============================================================================

pub struct SpeechTokenizerDecoder {
    pub semantic_codebook: Array,
    pub acoustic_codebooks: Vec<Array>,
    pub rvq_first_output_proj: nn::Conv1d,
    pub rvq_rest_output_proj: nn::Conv1d,

    pub pre_conv: CausalConv1d,

    pub pre_transformer_input_proj: nn::Linear,
    pub pre_transformer_output_proj: nn::Linear,
    pub pre_transformer_norm: nn::RmsNorm,
    pub pre_transformer_layers: Vec<DecoderTransformerLayer>,

    pub upsample_convs: Vec<CausalConvTranspose1d>,
    pub upsample_convnext: Vec<ConvNeXtBlock>,

    pub initial_conv: CausalConv1d,
    pub decoder_blocks: Vec<DecoderBlock>,
    pub final_snake: SnakeBeta,
    pub final_conv: CausalConv1d,

    pub config: DecoderConfig,
}

impl SpeechTokenizerDecoder {
    fn debug_tensor(name: &str, t: &Array) {
        use mlx_rs::transforms::eval;
        let flat = t.flatten(0, -1).unwrap();
        let min = flat.min_axis(0, None).unwrap();
        let max = flat.max_axis(0, None).unwrap();
        let mean = flat.mean_axis(0, None).unwrap();
        eval([&min, &max, &mean]).unwrap();
        tracing::debug!(
            "  {} shape={:?} min={:.4} max={:.4} mean={:.4}",
            name,
            t.shape(),
            min.item::<f32>(),
            max.item::<f32>(),
            mean.item::<f32>(),
        );
    }

    /// Decode 16-codebook codes to waveform samples.
    pub fn decode(&mut self, codes: &[[u32; 16]]) -> Result<Vec<f32>> {
        let num_frames = codes.len();
        if num_frames == 0 {
            return Ok(vec![]);
        }

        // Step 1: Dequantize
        let quantized = self.dequantize(codes)?;
        Self::debug_tensor("dequantized", &quantized);

        // Step 2: Pre-conv
        let h = self.pre_conv.forward(&quantized)?;
        Self::debug_tensor("pre_conv", &h);

        // Step 3: Pre-transformer
        let mut h = self.pre_transformer_input_proj.forward(&h)?;
        Self::debug_tensor("input_proj", &h);
        let mask = create_sliding_window_mask(num_frames as i32, self.config.sliding_window)?;
        for (i, layer) in self.pre_transformer_layers.iter_mut().enumerate() {
            h = layer.forward(&h, Some(&mask), 0)?;
            if i == 0 || i == 7 {
                Self::debug_tensor(&format!("transformer_layer_{i}"), &h);
            }
        }
        h = self.pre_transformer_norm.forward(&h)?;
        h = self.pre_transformer_output_proj.forward(&h)?;
        Self::debug_tensor("output_proj", &h);

        // Step 4: ConvNeXt upsample
        for i in 0..self.upsample_convs.len() {
            h = self.upsample_convs[i].forward(&h)?;
            h = self.upsample_convnext[i].forward(&h)?;
            Self::debug_tensor(&format!("upsample_{i}"), &h);
        }

        // Step 5: Audio decoder
        h = self.initial_conv.forward(&h)?;
        Self::debug_tensor("initial_conv", &h);
        for (i, block) in self.decoder_blocks.iter_mut().enumerate() {
            h = block.forward(&h)?;
            Self::debug_tensor(&format!("decoder_block_{i}"), &h);
        }
        h = self.final_snake.forward(&h)?;
        Self::debug_tensor("final_snake", &h);
        h = self.final_conv.forward(&h)?;
        Self::debug_tensor("final_conv_before_tanh", &h);

        // Clamp to [-1, 1] via tanh
        h = mlx_rs::ops::tanh(&h)?;
        eval(std::iter::once(&h))?;

        let h = h.reshape(&[-1])?;
        eval(std::iter::once(&h))?;
        Ok(h.as_slice::<f32>().to_vec())
    }

    fn dequantize(&mut self, codes: &[[u32; 16]]) -> Result<Array> {
        let num_frames = codes.len() as i32;

        // Semantic codebook (first quantizer)
        let semantic_indices: Vec<i32> = codes.iter().map(|c| c[0] as i32).collect();
        let semantic_idx = Array::from_slice(&semantic_indices, &[1, num_frames]);
        // Gather embeddings: [codebook_size, dim] indexed by [1, num_frames] → [1, num_frames, dim]
        let semantic_embed = gather_embeddings(&self.semantic_codebook, &semantic_idx)?;
        let semantic_out = self.rvq_first_output_proj.forward(&semantic_embed)?;

        // Acoustic codebooks (15 codebooks)
        let codebook_dim = self.acoustic_codebooks[0].dim(1) as i32;
        let mut acoustic_sum = zeros::<f32>(&[1, num_frames, codebook_dim])?;
        for g in 0..15 {
            let indices: Vec<i32> = codes.iter().map(|c| c[g + 1] as i32).collect();
            let idx = Array::from_slice(&indices, &[1, num_frames]);
            let embed = gather_embeddings(&self.acoustic_codebooks[g], &idx)?;
            acoustic_sum = acoustic_sum.add(embed)?;
        }
        let acoustic_out = self.rvq_rest_output_proj.forward(&acoustic_sum)?;

        Ok(semantic_out.add(acoustic_out)?)
    }
}

/// Gather embeddings: codebook[indices]
fn gather_embeddings(codebook: &Array, indices: &Array) -> Result<Array> {
    // codebook: [vocab_size, dim], indices: [B, T]
    // result: [B, T, dim]
    let flat = indices.flatten(0, -1)?;
    Ok(codebook.index(flat).reshape(&[
        indices.dim(0) as i32,
        indices.dim(1) as i32,
        codebook.dim(1) as i32,
    ])?)
}

fn create_sliding_window_mask(seq_len: i32, window_size: i32) -> Result<Array> {
    let rows = arange::<_, i32>(0, seq_len, None)?;
    let cols = arange::<_, i32>(0, seq_len, None)?;

    let rows = rows.index((.., NewAxis));
    let cols = cols.index(NewAxis);

    // Causal + sliding window: attend if row >= col AND row - col <= window_size
    let causal = rows.ge(&cols)?;
    let window = rows.subtract(&cols)?.le(&array!(window_size))?;
    let bool_mask = causal.logical_and(&window)?;

    // Convert to float additive mask: 0.0 where attend, -inf where masked
    // scaled_dot_product_attention adds this mask to the attention logits
    let float_mask = mlx_rs::ops::r#where(
        &bool_mask,
        &array!(0.0f32),
        &array!(f32::NEG_INFINITY),
    )?;

    Ok(float_mask)
}

// ============================================================================
// Weight Loading
// ============================================================================

fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::WeightNotFound(key.to_string()))
}

fn transpose_conv_weight(w: &Array) -> Result<Array> {
    Ok(w.transpose_axes(&[0, 2, 1])?)
}

fn transpose_conv_t_weight(w: &Array) -> Result<Array> {
    Ok(w.transpose_axes(&[1, 2, 0])?)
}

fn load_causal_conv1d(
    weights: &HashMap<String, Array>,
    prefix: &str,
    dilation: i32,
) -> Result<CausalConv1d> {
    let w = get_weight(weights, &format!("{prefix}.conv.weight"))?;
    let b = get_weight(weights, &format!("{prefix}.conv.bias"))?;
    let w = transpose_conv_weight(&w)?;

    let out_ch = w.dim(0) as i32;
    let kernel_size = w.dim(1) as i32;
    let in_ch = w.dim(2) as i32;
    let pad = (kernel_size - 1) * dilation;

    let groups = if in_ch == 1 && out_ch > 1 { out_ch } else { 1 };
    let actual_in = in_ch * groups;

    let mut conv = nn::Conv1dBuilder::new(actual_in, out_ch, kernel_size)
        .dilation(dilation)
        .groups(groups)
        .build()?;
    conv.weight = Param::new(w);
    conv.bias = Param::new(Some(b));

    Ok(CausalConv1d { conv, pad })
}

fn load_causal_conv_transpose1d(
    weights: &HashMap<String, Array>,
    prefix: &str,
    stride: i32,
) -> Result<CausalConvTranspose1d> {
    let w = get_weight(weights, &format!("{prefix}.conv.weight"))?;
    let b = get_weight(weights, &format!("{prefix}.conv.bias"))?;
    let w = transpose_conv_t_weight(&w)?;

    let out_ch = w.dim(0) as i32;
    let kernel_size = w.dim(1) as i32;
    let in_ch = w.dim(2) as i32;

    let mut conv_t = nn::ConvTranspose1dBuilder::new(in_ch, out_ch, kernel_size)
        .stride(stride)
        .build()?;
    conv_t.weight = Param::new(w);
    conv_t.bias = Param::new(Some(b));

    // Causal ConvTranspose1d: trim excess = kernel_size - stride from right
    let trim_right = kernel_size - stride;
    Ok(CausalConvTranspose1d {
        conv_t,
        trim_right,
    })
}

fn load_snake_beta(weights: &HashMap<String, Array>, prefix: &str) -> Result<SnakeBeta> {
    let alpha = get_weight(weights, &format!("{prefix}.alpha"))?;
    let beta = get_weight(weights, &format!("{prefix}.beta"))?;
    Ok(SnakeBeta { alpha, beta })
}

fn load_residual_unit(
    weights: &HashMap<String, Array>,
    prefix: &str,
    dilation: i32,
) -> Result<ResidualUnit> {
    Ok(ResidualUnit {
        act1: load_snake_beta(weights, &format!("{prefix}.act1"))?,
        conv1: load_causal_conv1d(weights, &format!("{prefix}.conv1"), dilation)?,
        act2: load_snake_beta(weights, &format!("{prefix}.act2"))?,
        conv2: load_causal_conv1d(weights, &format!("{prefix}.conv2"), 1)?,
    })
}

fn load_linear(weights: &HashMap<String, Array>, prefix: &str) -> Result<nn::Linear> {
    let w = get_weight(weights, &format!("{prefix}.weight"))?;
    let b = weights.get(&format!("{prefix}.bias")).cloned();
    Ok(nn::Linear {
        weight: Param::new(w),
        bias: Param::new(b),
    })
}

fn load_decoder_transformer_layer(
    weights: &HashMap<String, Array>,
    prefix: &str,
    config: &DecoderConfig,
) -> Result<DecoderTransformerLayer> {
    let rope = nn::RopeBuilder::new(config.head_dim)
        .base(config.rope_theta)
        .build()
        .map_err(|e| Error::Model(format!("RoPE build: {e}")))?;

    Ok(DecoderTransformerLayer {
        input_layernorm: nn::RmsNorm {
            weight: Param::new(get_weight(weights, &format!("{prefix}.input_layernorm.weight"))?),
            eps: config.rms_norm_eps,
        },
        q_proj: load_linear(weights, &format!("{prefix}.self_attn.q_proj"))?,
        k_proj: load_linear(weights, &format!("{prefix}.self_attn.k_proj"))?,
        v_proj: load_linear(weights, &format!("{prefix}.self_attn.v_proj"))?,
        o_proj: load_linear(weights, &format!("{prefix}.self_attn.o_proj"))?,
        attn_layer_scale: get_weight(weights, &format!("{prefix}.self_attn_layer_scale.scale"))?,
        post_attention_layernorm: nn::RmsNorm {
            weight: Param::new(get_weight(weights, &format!("{prefix}.post_attention_layernorm.weight"))?),
            eps: config.rms_norm_eps,
        },
        gate_proj: load_linear(weights, &format!("{prefix}.mlp.gate_proj"))?,
        up_proj: load_linear(weights, &format!("{prefix}.mlp.up_proj"))?,
        down_proj: load_linear(weights, &format!("{prefix}.mlp.down_proj"))?,
        mlp_layer_scale: get_weight(weights, &format!("{prefix}.mlp_layer_scale.scale"))?,
        n_heads: config.num_attention_heads,
        head_dim: config.head_dim,
        rope,
    })
}

fn normalize_codebook(embedding_sum: &Array, cluster_usage: &Array) -> Result<Array> {
    // embedding = embedding_sum / clamp(cluster_usage, min=1e-5).unsqueeze(-1)
    let clamped = mlx_rs::ops::maximum(cluster_usage, &array!(1e-5f32))?;
    Ok(embedding_sum.divide(clamped.index((.., NewAxis)))?)
}

pub fn load_speech_tokenizer(model_dir: &Path, config: &DecoderConfig) -> Result<SpeechTokenizerDecoder> {
    let st_dir = model_dir.join("speech_tokenizer");
    let path = st_dir.join("model.safetensors");
    let weights: HashMap<String, Array> = Array::load_safetensors(&path)?;

    // SplitRVQ codebooks
    let sem_sum = get_weight(&weights, "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum")?;
    let sem_usage = get_weight(&weights, "decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage")?;
    let semantic_codebook = normalize_codebook(&sem_sum, &sem_usage)?;

    let mut acoustic_codebooks = Vec::with_capacity(15);
    for i in 0..15 {
        let sum = get_weight(&weights, &format!("decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.embedding_sum"))?;
        let usage = get_weight(&weights, &format!("decoder.quantizer.rvq_rest.vq.layers.{i}._codebook.cluster_usage"))?;
        acoustic_codebooks.push(normalize_codebook(&sum, &usage)?);
    }

    // RVQ output projections (Conv1d k=1)
    let rvq_first_w = transpose_conv_weight(&get_weight(&weights, "decoder.quantizer.rvq_first.output_proj.weight")?)?;
    let out_dim = rvq_first_w.dim(0) as i32;
    let in_dim = rvq_first_w.dim(2) as i32;
    let mut rvq_first_output_proj = nn::Conv1dBuilder::new(in_dim, out_dim, 1).bias(false).build()?;
    rvq_first_output_proj.weight = Param::new(rvq_first_w);

    let rvq_rest_w = transpose_conv_weight(&get_weight(&weights, "decoder.quantizer.rvq_rest.output_proj.weight")?)?;
    let out_dim = rvq_rest_w.dim(0) as i32;
    let in_dim = rvq_rest_w.dim(2) as i32;
    let mut rvq_rest_output_proj = nn::Conv1dBuilder::new(in_dim, out_dim, 1).bias(false).build()?;
    rvq_rest_output_proj.weight = Param::new(rvq_rest_w);

    // Pre-conv
    let pre_conv = load_causal_conv1d(&weights, "decoder.pre_conv", 1)?;

    // Pre-transformer
    let pre_transformer_input_proj = load_linear(&weights, "decoder.pre_transformer.input_proj")?;
    let pre_transformer_output_proj = load_linear(&weights, "decoder.pre_transformer.output_proj")?;
    let pre_transformer_norm = nn::RmsNorm {
        weight: Param::new(get_weight(&weights, "decoder.pre_transformer.norm.weight")?),
        eps: config.rms_norm_eps,
    };

    let mut pre_transformer_layers = Vec::with_capacity(config.num_hidden_layers as usize);
    for i in 0..config.num_hidden_layers {
        pre_transformer_layers.push(load_decoder_transformer_layer(
            &weights,
            &format!("decoder.pre_transformer.layers.{i}"),
            config,
        )?);
    }

    // ConvNeXt upsample
    let mut upsample_convs = Vec::new();
    let mut upsample_convnext = Vec::new();
    for (i, &ratio) in config.upsampling_ratios.iter().enumerate() {
        upsample_convs.push(load_causal_conv_transpose1d(&weights, &format!("decoder.upsample.{i}.0"), ratio)?);
        let prefix = format!("decoder.upsample.{i}.1");
        upsample_convnext.push(ConvNeXtBlock {
            dwconv: load_causal_conv1d(&weights, &format!("{prefix}.dwconv"), 1)?,
            norm_weight: get_weight(&weights, &format!("{prefix}.norm.weight"))?,
            norm_bias: get_weight(&weights, &format!("{prefix}.norm.bias"))?,
            pwconv1_weight: get_weight(&weights, &format!("{prefix}.pwconv1.weight"))?,
            pwconv1_bias: get_weight(&weights, &format!("{prefix}.pwconv1.bias"))?,
            pwconv2_weight: get_weight(&weights, &format!("{prefix}.pwconv2.weight"))?,
            pwconv2_bias: get_weight(&weights, &format!("{prefix}.pwconv2.bias"))?,
            gamma: get_weight(&weights, &format!("{prefix}.gamma"))?,
        });
    }

    // Audio decoder
    let initial_conv = load_causal_conv1d(&weights, "decoder.decoder.0", 1)?;
    let dilations = [1, 3, 9];
    let mut decoder_blocks = Vec::new();
    for (b_idx, &rate) in config.upsample_rates.iter().enumerate() {
        let bi = b_idx + 1;
        decoder_blocks.push(DecoderBlock {
            snake: load_snake_beta(&weights, &format!("decoder.decoder.{bi}.block.0"))?,
            conv_t: load_causal_conv_transpose1d(&weights, &format!("decoder.decoder.{bi}.block.1"), rate)?,
            res_units: dilations.iter().enumerate().map(|(r_idx, &dil)| {
                let ri = r_idx + 2;
                load_residual_unit(&weights, &format!("decoder.decoder.{bi}.block.{ri}"), dil)
            }).collect::<Result<Vec<_>>>()?,
        });
    }

    let final_snake = load_snake_beta(&weights, "decoder.decoder.5")?;
    let final_conv = load_causal_conv1d(&weights, "decoder.decoder.6", 1)?;

    Ok(SpeechTokenizerDecoder {
        semantic_codebook,
        acoustic_codebooks,
        rvq_first_output_proj,
        rvq_rest_output_proj,
        pre_conv,
        pre_transformer_input_proj,
        pre_transformer_output_proj,
        pre_transformer_norm,
        pre_transformer_layers,
        upsample_convs,
        upsample_convnext,
        initial_conv,
        decoder_blocks,
        final_snake,
        final_conv,
        config: config.clone(),
    })
}
