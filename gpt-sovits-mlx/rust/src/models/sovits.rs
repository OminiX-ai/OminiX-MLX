//! SoVITS Vocoder for GPT-SoVITS
//!
//! Converts semantic tokens to audio waveforms.
//!
//! Pipeline:
//! 1. Decode semantic tokens via RVQ codebook
//! 2. Predict durations for each token
//! 3. Expand features according to durations
//! 4. Apply MRTE for style conditioning (optional)
//! 5. Upsample to audio waveform
//!
//! Key characteristics:
//! - RVQ codebook: 1024 codes, 256-dim embeddings
//! - Duration predictor: 3-layer CNN
//! - Upsampler: 640x (50Hz features to 32kHz audio)

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::{
    array,
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, Param},
    nn,
    ops::{indexing::IndexOp, concatenate_axis, maximum, squeeze_axes, stack_axis, swap_axes, tanh},
    Array,
};
use serde::Deserialize;

use crate::error::Error;

/// Configuration for SoVITS vocoder
#[derive(Debug, Clone, Deserialize)]
pub struct SoVITSConfig {
    /// Semantic embedding dimension (input from GPT)
    #[serde(default = "default_semantic_dim")]
    pub semantic_dim: i32,
    /// Audio feature dimension (from CNHubert)
    #[serde(default = "default_audio_feature_dim")]
    pub audio_feature_dim: i32,
    /// Duration predictor hidden channels
    #[serde(default = "default_duration_channels")]
    pub duration_channels: i32,
    /// Duration predictor kernel size
    #[serde(default = "default_duration_kernel_size")]
    pub duration_kernel_size: i32,
    /// Number of RVQ codebooks
    #[serde(default = "default_num_codebooks")]
    pub num_codebooks: i32,
    /// RVQ codebook size
    #[serde(default = "default_codebook_size")]
    pub codebook_size: i32,
    /// RVQ codebook dimension
    #[serde(default = "default_codebook_dim")]
    pub codebook_dim: i32,
    /// Upsampler channels
    #[serde(default = "default_upsample_channels")]
    pub upsample_channels: i32,
    /// Output sample rate
    #[serde(default = "default_sample_rate")]
    pub sample_rate: i32,
}

fn default_semantic_dim() -> i32 { 512 }
fn default_audio_feature_dim() -> i32 { 768 }
fn default_duration_channels() -> i32 { 256 }
fn default_duration_kernel_size() -> i32 { 3 }
fn default_num_codebooks() -> i32 { 8 }
fn default_codebook_size() -> i32 { 1024 }
fn default_codebook_dim() -> i32 { 256 }
fn default_upsample_channels() -> i32 { 256 }
fn default_sample_rate() -> i32 { 32000 }

impl Default for SoVITSConfig {
    fn default() -> Self {
        Self {
            semantic_dim: default_semantic_dim(),
            audio_feature_dim: default_audio_feature_dim(),
            duration_channels: default_duration_channels(),
            duration_kernel_size: default_duration_kernel_size(),
            num_codebooks: default_num_codebooks(),
            codebook_size: default_codebook_size(),
            codebook_dim: default_codebook_dim(),
            upsample_channels: default_upsample_channels(),
            sample_rate: default_sample_rate(),
        }
    }
}

impl SoVITSConfig {
    /// Upsample factor: 50Hz features to sample_rate
    pub fn upsample_factor(&self) -> i32 {
        self.sample_rate / 50
    }
}

/// Vector Quantizer codebook
///
/// Single codebook that maps indices to embeddings
#[derive(Debug, Clone, ModuleParameters)]
pub struct VectorQuantizer {
    #[param]
    pub codebook: nn::Embedding,
    pub codebook_size: i32,
    pub codebook_dim: i32,
}

impl VectorQuantizer {
    pub fn new(codebook_size: i32, codebook_dim: i32) -> Result<Self, Exception> {
        let codebook = nn::Embedding::new(codebook_size, codebook_dim)?;
        Ok(Self {
            codebook,
            codebook_size,
            codebook_dim,
        })
    }

    /// Decode indices to embeddings
    pub fn decode(&mut self, indices: &Array) -> Result<Array, Exception> {
        self.codebook.forward(indices)
    }
}

/// Semantic to Acoustic decoder
///
/// Maps semantic tokens to acoustic features via RVQ codebook lookup
#[derive(Debug, Clone, ModuleParameters)]
pub struct SemanticToAcoustic {
    /// First RVQ codebook for semantic token decoding
    #[param]
    pub quantizer: VectorQuantizer,
    /// Output projection
    #[param]
    pub output_proj: nn::Linear,
}

impl SemanticToAcoustic {
    pub fn new(config: &SoVITSConfig) -> Result<Self, Exception> {
        let quantizer = VectorQuantizer::new(config.codebook_size, config.codebook_dim)?;
        let output_proj = nn::LinearBuilder::new(config.codebook_dim, config.codebook_dim)
            .bias(true)
            .build()?;

        Ok(Self {
            quantizer,
            output_proj,
        })
    }
}

impl Module<&Array> for SemanticToAcoustic {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, semantic_tokens: &Array) -> Result<Self::Output, Self::Error> {
        // Decode from codebook
        let acoustic = self.quantizer.decode(semantic_tokens)?;
        // Apply output projection
        self.output_proj.forward(&acoustic)
    }

    fn training_mode(&mut self, mode: bool) {
        self.quantizer.codebook.training_mode(mode);
        self.output_proj.training_mode(mode);
    }
}

/// Convolutional block for duration predictor
#[derive(Debug, Clone, ModuleParameters)]
pub struct ConvBlock {
    #[param]
    pub conv: nn::Conv1d,
    #[param]
    pub norm: nn::LayerNorm,
}

impl ConvBlock {
    pub fn new(in_channels: i32, out_channels: i32, kernel_size: i32) -> Result<Self, Exception> {
        let padding = (kernel_size - 1) / 2;
        let conv = nn::Conv1dBuilder::new(in_channels, out_channels, kernel_size)
            .padding(padding)
            .build()?;
        let norm = nn::LayerNormBuilder::new(out_channels)
            .eps(1e-5)
            .build()?;

        Ok(Self { conv, norm })
    }
}

impl Module<&Array> for ConvBlock {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let h = self.conv.forward(x)?;
        let h = self.norm.forward(&h)?;
        nn::relu(&h)
    }

    fn training_mode(&mut self, mode: bool) {
        self.conv.training_mode(mode);
        self.norm.training_mode(mode);
    }
}

/// Duration Predictor
///
/// CNN-based predictor that outputs duration (in log scale) for each input token
#[derive(Debug, Clone, ModuleParameters)]
pub struct DurationPredictor {
    #[param]
    pub layers: Vec<ConvBlock>,
    #[param]
    pub proj: nn::Conv1d,
}

impl DurationPredictor {
    pub fn new(
        in_channels: i32,
        hidden_channels: i32,
        kernel_size: i32,
        num_layers: i32,
    ) -> Result<Self, Exception> {
        let mut layers = Vec::with_capacity(num_layers as usize);
        for i in 0..num_layers {
            let in_ch = if i == 0 { in_channels } else { hidden_channels };
            layers.push(ConvBlock::new(in_ch, hidden_channels, kernel_size)?);
        }

        let proj = nn::Conv1dBuilder::new(hidden_channels, 1, 1).build()?;

        Ok(Self { layers, proj })
    }
}

impl Module<&Array> for DurationPredictor {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let mut h = x.clone();
        for layer in &mut self.layers {
            h = layer.forward(&h)?;
        }
        // Project to duration (log scale)
        let duration = self.proj.forward(&h)?;
        // Squeeze last dimension: [batch, time, 1] -> [batch, time]
        squeeze_axes(&duration, &[-1])
    }

    fn training_mode(&mut self, mode: bool) {
        for layer in &mut self.layers {
            layer.training_mode(mode);
        }
        self.proj.training_mode(mode);
    }
}

/// Length Regulator
///
/// Expands features based on predicted durations
pub struct LengthRegulator;

impl LengthRegulator {
    /// Expand features according to durations
    ///
    /// Args:
    ///     x: Input features [batch, time, channels]
    ///     durations: Duration for each frame [batch, time]
    ///
    /// Returns:
    ///     Expanded features [batch, expanded_time, channels]
    pub fn regulate(x: &Array, durations: &Array) -> Result<Array, Exception> {
        let shape = x.shape();
        let batch_size = shape[0];
        let in_time = shape[1];
        let channels = shape[2];

        // Round durations to integers, minimum 1
        let durations = durations.round(None)?.as_type::<i32>()?;
        let durations = maximum(&durations, &array!(1i32))?;

        // Calculate total output length per batch
        // For simplicity, we'll use the max total duration
        let total_durations = durations.sum_axis(1, false)?;
        let max_out_len = total_durations.max(None)?.item::<i32>();

        // Expand each sequence using repeat
        // This is a simplified version - in practice would need proper batching
        let mut outputs = Vec::with_capacity(batch_size as usize);

        for b in 0..batch_size {
            let feat = x.index(b);  // [in_time, channels]
            let dur = durations.index(b);  // [in_time]

            let mut expanded_frames = Vec::new();
            for t in 0..in_time {
                let d = dur.index(t).item::<i32>();
                let frame = feat.index(t);  // [channels]
                // Repeat frame d times
                let frame_expanded = frame.index(mlx_rs::ops::indexing::NewAxis);  // [1, channels]
                let repeated = Array::repeat_axis::<f32>(frame_expanded, d, 0)?;  // [d, channels]
                expanded_frames.push(repeated);
            }

            // Concatenate all frames
            let expanded = if expanded_frames.is_empty() {
                Array::zeros::<f32>(&[0, channels])?
            } else {
                let refs: Vec<&Array> = expanded_frames.iter().collect();
                concatenate_axis(&refs, 0)?
            };

            // Pad to max_out_len
            let out_time = expanded.shape()[0] as i32;
            let expanded = if out_time < max_out_len {
                let pad_len = max_out_len - out_time;
                let padding = Array::zeros::<f32>(&[pad_len, channels])?;
                concatenate_axis(&[&expanded, &padding], 0)?
            } else if out_time > max_out_len {
                expanded.index(..max_out_len)
            } else {
                expanded
            };

            outputs.push(expanded);
        }

        // Stack batches
        let refs: Vec<&Array> = outputs.iter().collect();
        stack_axis(&refs, 0)
    }
}

/// Multi-Resolution Temporal Encoding (MRTE)
///
/// Captures temporal patterns at multiple scales for style conditioning
#[derive(Debug, Clone, ModuleParameters)]
pub struct MRTE {
    /// Multi-scale convolutions (one list per kernel size)
    #[param]
    pub convs: Vec<Vec<nn::Conv1d>>,
    /// Output projection
    #[param]
    pub proj: nn::Conv1d,
    pub channels: i32,
}

impl MRTE {
    pub fn new(
        channels: i32,
        hidden_channels: i32,
        kernel_sizes: &[i32],
        num_layers: i32,
    ) -> Result<Self, Exception> {
        let mut convs = Vec::new();
        for &kernel_size in kernel_sizes {
            let padding = (kernel_size - 1) / 2;
            let mut conv_layers = Vec::with_capacity(num_layers as usize);
            for i in 0..num_layers {
                let in_ch = if i == 0 { channels } else { hidden_channels };
                let conv = nn::Conv1dBuilder::new(in_ch, hidden_channels, kernel_size)
                    .padding(padding)
                    .build()?;
                conv_layers.push(conv);
            }
            convs.push(conv_layers);
        }

        // Output projection: combine all scales
        let total_out = hidden_channels * kernel_sizes.len() as i32;
        let proj = nn::Conv1dBuilder::new(total_out, channels, 1).build()?;

        Ok(Self {
            convs,
            proj,
            channels,
        })
    }
}

impl Module<&Array> for MRTE {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let original_time = x.shape()[1];
        let mut outputs = Vec::new();

        for conv_layers in &mut self.convs {
            let mut h = x.clone();
            for conv in conv_layers {
                h = conv.forward(&h)?;
                h = nn::leaky_relu(&h, 0.1)?;
            }
            // Ensure time dimension matches
            if h.shape()[1] != original_time {
                h = h.index((.., ..original_time as i32, ..));
            }
            outputs.push(h);
        }

        // Concatenate along channel dimension
        let refs: Vec<&Array> = outputs.iter().collect();
        let concat = concatenate_axis(&refs, 2)?;

        // Project back to original channels
        self.proj.forward(&concat)
    }

    fn training_mode(&mut self, mode: bool) {
        for conv_layers in &mut self.convs {
            for conv in conv_layers {
                conv.training_mode(mode);
            }
        }
        self.proj.training_mode(mode);
    }
}

/// Simplified Upsampler
///
/// Upsamples features to audio waveform using repetition + Conv1d refinement
#[derive(Debug, Clone, ModuleParameters)]
pub struct SimplifiedUpsampler {
    #[param]
    pub layers: Vec<nn::Conv1d>,
    pub upsample_factor: i32,
}

impl SimplifiedUpsampler {
    pub fn new(
        in_channels: i32,
        hidden_channels: i32,
        out_channels: i32,
        upsample_factor: i32,
        num_layers: i32,
    ) -> Result<Self, Exception> {
        let mut layers = Vec::with_capacity(num_layers as usize);
        let mut channels = in_channels;

        for i in 0..num_layers {
            let out_ch = if i < num_layers - 1 {
                hidden_channels
            } else {
                out_channels
            };
            let conv = nn::Conv1dBuilder::new(channels, out_ch, 3)
                .padding(1)
                .build()?;
            layers.push(conv);
            channels = out_ch;
        }

        Ok(Self {
            layers,
            upsample_factor,
        })
    }
}

impl Module<&Array> for SimplifiedUpsampler {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, x: &Array) -> Result<Self::Output, Self::Error> {
        let shape = x.shape();
        let batch = shape[0] as i32;
        let time = shape[1] as i32;
        let channels = shape[2] as i32;
        let target_time = time * self.upsample_factor;

        // Nearest-neighbor upsampling: repeat each frame
        // [batch, time, channels] -> [batch, time, upsample, channels] -> [batch, time*upsample, channels]
        let x = x.index((.., .., mlx_rs::ops::indexing::NewAxis, ..));  // [batch, time, 1, channels]
        let x = Array::repeat_axis::<f32>(x, self.upsample_factor, 2)?;  // [batch, time, upsample, channels]
        let mut x = x.reshape(&[batch, target_time, channels])?;

        // Apply refinement layers
        let num_layers = self.layers.len();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            x = layer.forward(&x)?;
            if i < num_layers - 1 {
                x = nn::leaky_relu(&x, 0.1)?;
            }
        }

        // Final tanh activation
        tanh(&x)
    }

    fn training_mode(&mut self, mode: bool) {
        for layer in &mut self.layers {
            layer.training_mode(mode);
        }
    }
}

/// SoVITS Vocoder
///
/// Full vocoder that converts semantic tokens to audio waveforms
#[derive(Debug, Clone, ModuleParameters)]
pub struct SoVITSVocoder {
    pub config: SoVITSConfig,

    /// Semantic to acoustic decoder
    #[param]
    pub semantic_decoder: SemanticToAcoustic,

    /// Duration predictor
    #[param]
    pub duration_predictor: DurationPredictor,

    /// Audio feature projection (for reference audio)
    #[param]
    pub audio_feature_proj: nn::Linear,

    /// MRTE for style conditioning
    #[param]
    pub mrte: MRTE,

    /// Audio projection to upsampler input
    #[param]
    pub audio_proj: nn::Conv1d,

    /// Upsampler
    #[param]
    pub upsampler: SimplifiedUpsampler,
}

impl SoVITSVocoder {
    pub fn new(config: SoVITSConfig) -> Result<Self, Exception> {
        let semantic_decoder = SemanticToAcoustic::new(&config)?;

        let duration_predictor = DurationPredictor::new(
            config.codebook_dim,
            config.duration_channels,
            config.duration_kernel_size,
            3,  // num_layers
        )?;

        let audio_feature_proj =
            nn::LinearBuilder::new(config.audio_feature_dim, config.codebook_dim)
                .bias(true)
                .build()?;

        let mrte = MRTE::new(
            config.codebook_dim,
            config.codebook_dim,
            &[3, 5, 7],  // kernel_sizes
            3,           // num_layers
        )?;

        let audio_proj = nn::Conv1dBuilder::new(config.codebook_dim, config.upsample_channels, 1)
            .build()?;

        let upsampler = SimplifiedUpsampler::new(
            config.upsample_channels,
            config.upsample_channels,
            1,  // out_channels (mono audio)
            config.upsample_factor(),
            4,  // num_layers
        )?;

        Ok(Self {
            config,
            semantic_decoder,
            duration_predictor,
            audio_feature_proj,
            mrte,
            audio_proj,
            upsampler,
        })
    }
}

/// Input for vocoder synthesis
pub struct VocoderInput<'a> {
    /// Semantic tokens [batch, seq]
    pub semantic_tokens: &'a Array,
    /// Optional reference audio features for style [batch, time, feat_dim]
    pub audio_features: Option<&'a Array>,
    /// Optional pre-computed durations [batch, seq]
    pub durations: Option<&'a Array>,
    /// Speed factor (1.0 = normal)
    pub speed_factor: f32,
}

impl Module<VocoderInput<'_>> for SoVITSVocoder {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: VocoderInput<'_>) -> Result<Self::Output, Self::Error> {
        let VocoderInput {
            semantic_tokens,
            audio_features,
            durations,
            speed_factor,
        } = input;

        // 1. Decode semantic tokens to acoustic features
        let acoustic = self.semantic_decoder.forward(semantic_tokens)?;  // [batch, seq, dim]

        // 2. Predict durations if not provided
        let durations = match durations {
            Some(d) => d.clone(),
            None => {
                let log_durations = self.duration_predictor.forward(&acoustic)?;
                log_durations.exp()?
            }
        };

        // Apply speed factor
        let durations = durations.divide(array!(speed_factor))?;

        // 3. Expand features according to durations
        let expanded = LengthRegulator::regulate(&acoustic, &durations)?;

        // 4. Apply MRTE for style conditioning (if reference audio provided)
        let expanded = if let Some(audio_feat) = audio_features {
            // Project audio features
            let audio_proj = self.audio_feature_proj.forward(audio_feat)?;
            // Apply MRTE
            let style = self.mrte.forward(&audio_proj)?;
            // Interpolate style to match expanded length
            let style = if style.shape()[1] != expanded.shape()[1] {
                let ratio = expanded.shape()[1] as f32 / style.shape()[1] as f32;
                let repeat_count = (ratio.ceil() as i32).max(1);
                let style_repeated = Array::repeat_axis::<f32>(style, repeat_count, 1)?;
                style_repeated.index((.., ..expanded.shape()[1] as i32, ..))
            } else {
                style
            };
            // Add style with small weight
            expanded.add(&style.multiply(array!(0.1f32))?)?
        } else {
            expanded
        };

        // 5. Project to upsampler input dimension
        let x = self.audio_proj.forward(&expanded)?;
        let x = nn::leaky_relu(&x, 0.1)?;

        // 6. Upsample to audio
        self.upsampler.forward(&x)
    }

    fn training_mode(&mut self, mode: bool) {
        self.semantic_decoder.training_mode(mode);
        self.duration_predictor.training_mode(mode);
        self.audio_feature_proj.training_mode(mode);
        self.mrte.training_mode(mode);
        self.audio_proj.training_mode(mode);
        self.upsampler.training_mode(mode);
    }
}

/// Load SoVITS vocoder weights
pub fn load_sovits_weights(
    model: &mut SoVITSVocoder,
    weights: &HashMap<String, Array>,
) -> Result<(), Error> {
    // Helper to try multiple weight names
    let get_weight = |keys: &[&str]| -> Option<Array> {
        for key in keys {
            if let Some(w) = weights.get(*key) {
                return Some(w.clone());
            }
        }
        None
    };

    // The SoVITS model in GPT-SoVITS uses weight names like:
    // - ssl_proj.* - SSL (HuBERT) feature projection
    // - enc_p.* - Phoneme encoder
    // - dec.* - Decoder
    // - quantizer.* - VQ codebook
    // - ref_enc.* - Reference encoder (MRTE)

    // Semantic decoder (RVQ codebook)
    // Try different naming conventions
    if let Some(w) = get_weight(&[
        "quantizer.codebook.weight",
        "vq.codebook.weight",
        "ssl_proj.weight",  // May be used as initial projection
    ]) {
        model.semantic_decoder.quantizer.codebook.weight = Param::new(w);
    }

    if let Some(w) = get_weight(&[
        "semantic_decoder.output_proj.weight",
        "ssl_proj.weight",
    ]) {
        model.semantic_decoder.output_proj.weight = Param::new(w);
    }
    if let Some(b) = get_weight(&[
        "semantic_decoder.output_proj.bias",
        "ssl_proj.bias",
    ]) {
        model.semantic_decoder.output_proj.bias = Param::new(Some(b));
    }

    // Duration predictor
    // Note: PyTorch conv weights are [out, in, kernel], mlx-rs expects [out, kernel, in]
    for (i, layer) in model.duration_predictor.layers.iter_mut().enumerate() {
        if let Some(w) = get_weight(&[
            &format!("duration_predictor.layers.{}.conv.weight", i),
            &format!("dp.convs.{}.weight", i),
        ]) {
            let w = swap_axes(&w, 1, 2).unwrap_or(w);
            layer.conv.weight = Param::new(w);
        }
        if let Some(b) = get_weight(&[
            &format!("duration_predictor.layers.{}.conv.bias", i),
            &format!("dp.convs.{}.bias", i),
        ]) {
            layer.conv.bias = Param::new(Some(b));
        }
        if let Some(w) = get_weight(&[
            &format!("duration_predictor.layers.{}.norm.weight", i),
            &format!("dp.norms.{}.weight", i),
        ]) {
            layer.norm.weight = Param::new(Some(w));
        }
        if let Some(b) = get_weight(&[
            &format!("duration_predictor.layers.{}.norm.bias", i),
            &format!("dp.norms.{}.bias", i),
        ]) {
            layer.norm.bias = Param::new(Some(b));
        }
    }
    if let Some(w) = get_weight(&[
        "duration_predictor.proj.weight",
        "dp.proj.weight",
    ]) {
        let w = swap_axes(&w, 1, 2).unwrap_or(w);
        model.duration_predictor.proj.weight = Param::new(w);
    }
    if let Some(b) = get_weight(&[
        "duration_predictor.proj.bias",
        "dp.proj.bias",
    ]) {
        model.duration_predictor.proj.bias = Param::new(Some(b));
    }

    // Audio feature projection
    if let Some(w) = get_weight(&[
        "audio_feature_proj.weight",
        "ref_enc.proj.weight",
    ]) {
        model.audio_feature_proj.weight = Param::new(w);
    }
    if let Some(b) = get_weight(&[
        "audio_feature_proj.bias",
        "ref_enc.proj.bias",
    ]) {
        model.audio_feature_proj.bias = Param::new(Some(b));
    }

    // MRTE (Multi-Resolution Temporal Encoding) convolutions
    for (scale_idx, conv_layers) in model.mrte.convs.iter_mut().enumerate() {
        for (layer_idx, conv) in conv_layers.iter_mut().enumerate() {
            if let Some(w) = get_weight(&[
                &format!("mrte.convs.{}.{}.weight", scale_idx, layer_idx),
                &format!("ref_enc.convs.{}.{}.weight", scale_idx, layer_idx),
            ]) {
                let w = swap_axes(&w, 1, 2).unwrap_or(w);
                conv.weight = Param::new(w);
            }
            if let Some(b) = get_weight(&[
                &format!("mrte.convs.{}.{}.bias", scale_idx, layer_idx),
                &format!("ref_enc.convs.{}.{}.bias", scale_idx, layer_idx),
            ]) {
                conv.bias = Param::new(Some(b));
            }
        }
    }
    if let Some(w) = get_weight(&[
        "mrte.proj.weight",
        "ref_enc.out_proj.weight",
    ]) {
        let w = swap_axes(&w, 1, 2).unwrap_or(w);
        model.mrte.proj.weight = Param::new(w);
    }
    if let Some(b) = get_weight(&[
        "mrte.proj.bias",
        "ref_enc.out_proj.bias",
    ]) {
        model.mrte.proj.bias = Param::new(Some(b));
    }

    // Audio projection (before upsampler)
    if let Some(w) = get_weight(&[
        "audio_proj.weight",
        "dec.conv_pre.weight",
    ]) {
        let w = swap_axes(&w, 1, 2).unwrap_or(w);
        model.audio_proj.weight = Param::new(w);
    }
    if let Some(b) = get_weight(&[
        "audio_proj.bias",
        "dec.conv_pre.bias",
    ]) {
        model.audio_proj.bias = Param::new(Some(b));
    }

    // Upsampler layers
    for (i, layer) in model.upsampler.layers.iter_mut().enumerate() {
        if let Some(w) = get_weight(&[
            &format!("upsampler.layers.{}.weight", i),
            &format!("dec.ups.{}.weight", i),
            &format!("dec.resblocks.{}.convs1.0.weight", i),
        ]) {
            let w = swap_axes(&w, 1, 2).unwrap_or(w);
            layer.weight = Param::new(w);
        }
        if let Some(b) = get_weight(&[
            &format!("upsampler.layers.{}.bias", i),
            &format!("dec.ups.{}.bias", i),
            &format!("dec.resblocks.{}.convs1.0.bias", i),
        ]) {
            layer.bias = Param::new(Some(b));
        }
    }

    Ok(())
}

/// Load SoVITS model from safetensors file
pub fn load_sovits_model(weights_path: impl AsRef<Path>) -> Result<SoVITSVocoder, Error> {
    let path = weights_path.as_ref();

    let config = SoVITSConfig::default();
    let mut model = SoVITSVocoder::new(config)?;

    let weights = Array::load_safetensors(path)?;
    load_sovits_weights(&mut model, &weights)?;

    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::transforms::eval;

    #[test]
    fn test_sovits_config_default() {
        let config = SoVITSConfig::default();
        assert_eq!(config.codebook_dim, 256);
        assert_eq!(config.upsample_factor(), 640);  // 32000 / 50
    }

    #[test]
    fn test_vector_quantizer() {
        let mut vq = VectorQuantizer::new(1024, 256).unwrap();

        let indices = Array::zeros::<i32>(&[1, 10]).unwrap();
        let output = vq.decode(&indices).unwrap();
        eval([&output]).unwrap();

        assert_eq!(output.shape(), &[1, 10, 256]);
    }

    #[test]
    fn test_semantic_to_acoustic() {
        let config = SoVITSConfig::default();
        let mut decoder = SemanticToAcoustic::new(&config).unwrap();

        let semantic_tokens = Array::zeros::<i32>(&[1, 10]).unwrap();
        let output = decoder.forward(&semantic_tokens).unwrap();
        eval([&output]).unwrap();

        assert_eq!(output.shape(), &[1, 10, 256]);
    }

    #[test]
    fn test_duration_predictor() {
        let mut predictor = DurationPredictor::new(256, 256, 3, 3).unwrap();

        let x = Array::zeros::<f32>(&[1, 10, 256]).unwrap();
        let output = predictor.forward(&x).unwrap();
        eval([&output]).unwrap();

        assert_eq!(output.shape(), &[1, 10]);
    }

    #[test]
    fn test_simplified_upsampler() {
        let mut upsampler = SimplifiedUpsampler::new(256, 256, 1, 640, 4).unwrap();

        let x = Array::zeros::<f32>(&[1, 10, 256]).unwrap();
        let output = upsampler.forward(&x).unwrap();
        eval([&output]).unwrap();

        // 10 * 640 = 6400
        assert_eq!(output.shape(), &[1, 6400, 1]);
    }
}
