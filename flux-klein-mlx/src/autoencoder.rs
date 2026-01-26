//! VAE AutoEncoder for FLUX
//!
//! Updated to support HuggingFace diffusers format (AutoencoderKLFlux2)

use mlx_rs::{
    array,
    builder::Builder,
    error::Exception,
    module::Module,
    nn::{Conv2d, Conv2dBuilder, GroupNorm, GroupNormBuilder, Linear, Upsample, UpsampleMode},
    ops,
    Array,
};
use mlx_macros::ModuleParameters;

// ============================================================================
// AutoEncoder Configuration
// ============================================================================

/// VAE configuration
#[derive(Debug, Clone)]
pub struct AutoEncoderConfig {
    /// Resolution of the latent space
    pub resolution: i32,
    /// Number of input/output channels (RGB = 3)
    pub in_channels: i32,
    /// Base channel multiplier
    pub ch: i32,
    /// Output channels (latent channels)
    pub out_ch: i32,
    /// Channel multipliers per resolution
    pub ch_mult: Vec<i32>,
    /// Number of resnet blocks per resolution
    pub num_res_blocks: i32,
    /// Latent channels (z_channels)
    pub z_channels: i32,
    /// Scaling factor for latent space
    pub scale_factor: f32,
    /// Shift factor for latent space
    pub shift_factor: f32,
}

impl Default for AutoEncoderConfig {
    fn default() -> Self {
        // FLUX.1 VAE configuration
        Self {
            resolution: 256,
            in_channels: 3,
            ch: 128,
            out_ch: 3,
            ch_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
            z_channels: 16,
            scale_factor: 0.3611,
            shift_factor: 0.1159,
        }
    }
}

impl AutoEncoderConfig {
    /// FLUX.2 VAE configuration (AutoencoderKLFlux2)
    ///
    /// Has 32 latent channels instead of 16
    pub fn flux2() -> Self {
        Self {
            resolution: 1024,
            in_channels: 3,
            ch: 128,
            out_ch: 3,
            ch_mult: vec![1, 2, 4, 4],  // [128, 256, 512, 512]
            num_res_blocks: 2,
            z_channels: 32,  // 32 for FLUX.2 (not 16)
            scale_factor: 0.3611,
            shift_factor: 0.1159,
        }
    }
}

// ============================================================================
// ResNet Block
// ============================================================================

/// Residual block with GroupNorm and SiLU activation
#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct ResnetBlock {
    pub in_channels: i32,
    pub out_channels: i32,

    #[param]
    pub norm1: GroupNorm,
    #[param]
    pub conv1: Conv2d,
    #[param]
    pub norm2: GroupNorm,
    #[param]
    pub conv2: Conv2d,

    #[param]
    pub conv_shortcut: Option<Conv2d>,
}

impl ResnetBlock {
    /// Create a new ResNet block
    pub fn new(in_channels: i32, out_channels: i32) -> Result<Self, Exception> {
        let norm1 = GroupNormBuilder::new(32, in_channels)
            .pytorch_compatible(true)
            .build()?;

        let conv1 = Conv2dBuilder::new(in_channels, out_channels, (3, 3))
            .padding((1, 1))
            .build()?;

        let norm2 = GroupNormBuilder::new(32, out_channels)
            .pytorch_compatible(true)
            .build()?;

        let conv2 = Conv2dBuilder::new(out_channels, out_channels, (3, 3))
            .padding((1, 1))
            .build()?;

        let conv_shortcut = if in_channels != out_channels {
            Some(Conv2dBuilder::new(in_channels, out_channels, (1, 1)).build()?)
        } else {
            None
        };

        Ok(Self {
            in_channels,
            out_channels,
            norm1,
            conv1,
            norm2,
            conv2,
            conv_shortcut,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let h = self.norm1.forward(x)?;
        let h = mlx_rs::nn::silu(&h)?;
        let h = self.conv1.forward(&h)?;

        let h = self.norm2.forward(&h)?;
        let h = mlx_rs::nn::silu(&h)?;
        let h = self.conv2.forward(&h)?;

        let shortcut = if let Some(ref mut conv) = self.conv_shortcut {
            conv.forward(x)?
        } else {
            x.clone()
        };

        ops::add(&h, &shortcut)
    }
}

// ============================================================================
// Attention Block (using Linear layers like diffusers)
// ============================================================================

/// Self-attention block for the autoencoder
/// Uses Linear layers for Q/K/V projections (matches diffusers format)
#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct AttnBlock {
    pub channels: i32,

    #[param]
    pub group_norm: GroupNorm,
    #[param]
    pub to_q: Linear,
    #[param]
    pub to_k: Linear,
    #[param]
    pub to_v: Linear,
    #[param]
    pub to_out: Linear,
}

impl AttnBlock {
    pub fn new(channels: i32) -> Result<Self, Exception> {
        Ok(Self {
            channels,
            group_norm: GroupNormBuilder::new(32, channels)
                .pytorch_compatible(true)
                .build()?,
            to_q: Linear::new(channels, channels)?,
            to_k: Linear::new(channels, channels)?,
            to_v: Linear::new(channels, channels)?,
            to_out: Linear::new(channels, channels)?,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let h = self.group_norm.forward(x)?;

        // x shape: [batch, height, width, channels] (NHWC)
        let shape = h.shape();
        let batch = shape[0];
        let height = shape[1];
        let width = shape[2];
        let channels = shape[3];

        // Flatten spatial dimensions for Linear layers: [batch, h*w, channels]
        let h_flat = h.reshape(&[batch, height * width, channels])?;

        let q = self.to_q.forward(&h_flat)?;
        let k = self.to_k.forward(&h_flat)?;
        let v = self.to_v.forward(&h_flat)?;

        // Scaled dot-product attention
        let scale = (channels as f32).sqrt();
        let attn = ops::matmul(&q, &k.transpose_axes(&[0, 2, 1])?)?;
        let attn = ops::divide(&attn, &array!(scale))?;
        let attn = ops::softmax_axis(&attn, -1, None)?;

        let out = ops::matmul(&attn, &v)?;

        // Project out
        let out = self.to_out.forward(&out)?;

        // Reshape back to [batch, height, width, channels]
        let out = out.reshape(&[batch, height, width, channels])?;

        ops::add(x, &out)
    }
}

// ============================================================================
// Decoder
// ============================================================================

/// VAE Decoder - converts latent codes to images
#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct Decoder {
    pub config: AutoEncoderConfig,

    // Post quantization conv (applied to latents before decoder)
    #[param]
    pub post_quant_conv: Conv2d,

    // Initial convolution
    #[param]
    pub conv_in: Conv2d,

    // Middle block
    #[param]
    pub mid_block_resnets_0: ResnetBlock,
    #[param]
    pub mid_block_attentions_0: AttnBlock,
    #[param]
    pub mid_block_resnets_1: ResnetBlock,

    // Up blocks (stored in order matching checkpoint: 0=lowest res, 3=highest res)
    #[param]
    pub up_blocks: Vec<UpBlock>,

    // Output
    #[param]
    pub conv_norm_out: GroupNorm,
    #[param]
    pub conv_out: Conv2d,
}

/// Upsampling block containing ResNet blocks and optional upsampling
#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct UpBlock {
    #[param]
    pub resnets: Vec<ResnetBlock>,
    pub upsample: Option<Upsample>,
    #[param]
    pub upsamplers_0_conv: Option<Conv2d>,
}

impl Decoder {
    pub fn new(config: AutoEncoderConfig) -> Result<Self, Exception> {
        let ch = config.ch;
        let ch_mult = &config.ch_mult;
        let num_res_blocks = config.num_res_blocks;
        let num_resolutions = ch_mult.len();

        // Post quantization conv (1x1 conv, same channels in/out)
        // Initialize with identity weights (many VAE checkpoints don't include this layer)
        let mut post_quant_conv = Conv2dBuilder::new(config.z_channels, config.z_channels, (1, 1))
            .build()?;

        // Set to identity: weight[i, 0, 0, i] = 1.0, bias = 0
        // MLX Conv2d weight shape: [out_ch, kH, kW, in_ch] for NHWC input
        let z_ch = config.z_channels as usize;
        let mut identity_weight = vec![0.0f32; z_ch * z_ch];
        for i in 0..z_ch {
            identity_weight[i * z_ch + i] = 1.0;  // Diagonal = 1
        }
        let identity_weight = Array::from_slice(&identity_weight, &[config.z_channels, 1, 1, config.z_channels]);
        let identity_bias = Array::zeros::<f32>(&[config.z_channels])?;
        *post_quant_conv.weight = identity_weight;
        *post_quant_conv.bias = Some(identity_bias);

        // Compute channel sizes
        let block_in = ch * ch_mult[num_resolutions - 1];

        // Initial conv from latent to block_in channels
        let conv_in = Conv2dBuilder::new(config.z_channels, block_in, (3, 3))
            .padding((1, 1))
            .build()?;

        // Middle blocks
        let mid_block_resnets_0 = ResnetBlock::new(block_in, block_in)?;
        let mid_block_attentions_0 = AttnBlock::new(block_in)?;
        let mid_block_resnets_1 = ResnetBlock::new(block_in, block_in)?;

        // Up blocks (in forward order: 0=lowest res, 3=highest res)
        // But we build them in reverse order to match the decoding process
        let mut up_blocks = Vec::new();
        let mut curr_channels = block_in;

        for i in (0..num_resolutions).rev() {
            let out_channels = ch * ch_mult[i];

            let mut resnets = Vec::new();
            for j in 0..=num_res_blocks {
                let in_ch = if j == 0 { curr_channels } else { out_channels };
                resnets.push(ResnetBlock::new(in_ch, out_channels)?);
            }

            // Upsample (except for the last resolution - which is up_blocks[3] = i=0)
            let (upsample, upsamplers_0_conv) = if i > 0 {
                let up = Upsample::new(2.0, UpsampleMode::Nearest);
                let conv = Conv2dBuilder::new(out_channels, out_channels, (3, 3))
                    .padding((1, 1))
                    .build()?;
                (Some(up), Some(conv))
            } else {
                (None, None)
            };

            up_blocks.push(UpBlock {
                resnets,
                upsample,
                upsamplers_0_conv,
            });

            curr_channels = out_channels;
        }

        // Output layers
        let conv_norm_out = GroupNormBuilder::new(32, ch)
            .pytorch_compatible(true)
            .build()?;

        let conv_out = Conv2dBuilder::new(ch, config.out_ch, (3, 3))
            .padding((1, 1))
            .build()?;

        Ok(Self {
            config,
            post_quant_conv,
            conv_in,
            mid_block_resnets_0,
            mid_block_attentions_0,
            mid_block_resnets_1,
            up_blocks,
            conv_norm_out,
            conv_out,
        })
    }

    /// Decode latent codes to images
    ///
    /// # Arguments
    /// * `z` - Latent codes [batch, height, width, z_channels]
    pub fn forward(&mut self, z: &Array) -> Result<Array, Exception> {
        // Scale and shift latents (inverse of encoder's normalization)
        let z = ops::divide(z, &array!(self.config.scale_factor))?;
        let z = ops::add(&z, &array!(self.config.shift_factor))?;

        // Apply post_quant_conv
        let z = self.post_quant_conv.forward(&z)?;

        // Initial conv
        let mut h = self.conv_in.forward(&z)?;

        // Middle
        h = self.mid_block_resnets_0.forward(&h)?;
        h = self.mid_block_attentions_0.forward(&h)?;
        h = self.mid_block_resnets_1.forward(&h)?;

        // Upsampling
        for up_block in &mut self.up_blocks {
            for resnet in &mut up_block.resnets {
                h = resnet.forward(&h)?;
            }

            if let (Some(ref mut up), Some(ref mut conv)) =
                (&mut up_block.upsample, &mut up_block.upsamplers_0_conv)
            {
                h = up.forward(&h)?;
                h = conv.forward(&h)?;
            }
        }

        // Output
        h = self.conv_norm_out.forward(&h)?;
        h = mlx_rs::nn::silu(&h)?;
        self.conv_out.forward(&h)
    }
}

// ============================================================================
// Encoder
// ============================================================================

/// Downsampling block containing ResNet blocks and optional downsampling
#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct DownBlock {
    #[param]
    pub resnets: Vec<ResnetBlock>,
    #[param]
    pub downsamplers_0_conv: Option<Conv2d>,
}

/// VAE Encoder - converts images to latent codes
#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct Encoder {
    pub config: AutoEncoderConfig,

    // Initial convolution
    #[param]
    pub conv_in: Conv2d,

    // Down blocks
    #[param]
    pub down_blocks: Vec<DownBlock>,

    // Middle block
    #[param]
    pub mid_block_resnets_0: ResnetBlock,
    #[param]
    pub mid_block_attentions_0: AttnBlock,
    #[param]
    pub mid_block_resnets_1: ResnetBlock,

    // Output
    #[param]
    pub conv_norm_out: GroupNorm,
    #[param]
    pub conv_out: Conv2d,

    // Quantization conv
    #[param]
    pub quant_conv: Conv2d,
}

impl Encoder {
    pub fn new(config: AutoEncoderConfig) -> Result<Self, Exception> {
        let ch = config.ch;
        let ch_mult = &config.ch_mult;
        let num_res_blocks = config.num_res_blocks;
        let num_resolutions = ch_mult.len();

        // Initial conv from RGB to base channels
        let conv_in = Conv2dBuilder::new(config.in_channels, ch, (3, 3))
            .padding((1, 1))
            .build()?;

        // Down blocks
        let mut down_blocks = Vec::new();
        let mut curr_channels = ch;

        for i in 0..num_resolutions {
            let out_channels = ch * ch_mult[i];

            let mut resnets = Vec::new();
            for j in 0..num_res_blocks {
                let in_ch = if j == 0 { curr_channels } else { out_channels };
                resnets.push(ResnetBlock::new(in_ch, out_channels)?);
            }

            // Downsample (except for the last resolution)
            let downsamplers_0_conv = if i < num_resolutions - 1 {
                Some(Conv2dBuilder::new(out_channels, out_channels, (3, 3))
                    .stride((2, 2))
                    .padding((0, 0))  // Asymmetric padding handled manually
                    .build()?)
            } else {
                None
            };

            down_blocks.push(DownBlock {
                resnets,
                downsamplers_0_conv,
            });

            curr_channels = out_channels;
        }

        // Middle blocks
        let block_in = ch * ch_mult[num_resolutions - 1];
        let mid_block_resnets_0 = ResnetBlock::new(block_in, block_in)?;
        let mid_block_attentions_0 = AttnBlock::new(block_in)?;
        let mid_block_resnets_1 = ResnetBlock::new(block_in, block_in)?;

        // Output layers
        let conv_norm_out = GroupNormBuilder::new(32, block_in)
            .pytorch_compatible(true)
            .build()?;

        // Output to 2 * z_channels (for mean and logvar)
        let conv_out = Conv2dBuilder::new(block_in, 2 * config.z_channels, (3, 3))
            .padding((1, 1))
            .build()?;

        // Quant conv (2 * z_channels -> 2 * z_channels)
        let quant_conv = Conv2dBuilder::new(2 * config.z_channels, 2 * config.z_channels, (1, 1))
            .build()?;

        Ok(Self {
            config,
            conv_in,
            down_blocks,
            mid_block_resnets_0,
            mid_block_attentions_0,
            mid_block_resnets_1,
            conv_norm_out,
            conv_out,
            quant_conv,
        })
    }

    /// Encode image to latent distribution parameters (mean, logvar)
    ///
    /// # Arguments
    /// * `x` - Input image [batch, height, width, 3] in range [-1, 1]
    ///
    /// # Returns
    /// Tuple of (mean, logvar) each [batch, h/8, w/8, z_channels]
    pub fn forward(&mut self, x: &Array) -> Result<(Array, Array), Exception> {
        // Initial conv
        let mut h = self.conv_in.forward(x)?;

        // Downsampling
        for down_block in &mut self.down_blocks {
            for resnet in &mut down_block.resnets {
                h = resnet.forward(&h)?;
            }

            if let Some(ref mut conv) = down_block.downsamplers_0_conv {
                // Asymmetric padding (0, 1, 0, 1) for stride=2 conv
                // pad(array, width, value, mode, stream)
                let pad_width: &[(i32, i32)] = &[(0, 0), (0, 1), (0, 1), (0, 0)];
                h = ops::pad(&h, pad_width, None, None)?;
                h = conv.forward(&h)?;
            }
        }

        // Middle
        h = self.mid_block_resnets_0.forward(&h)?;
        h = self.mid_block_attentions_0.forward(&h)?;
        h = self.mid_block_resnets_1.forward(&h)?;

        // Output
        h = self.conv_norm_out.forward(&h)?;
        h = mlx_rs::nn::silu(&h)?;
        h = self.conv_out.forward(&h)?;

        // Quant conv
        h = self.quant_conv.forward(&h)?;

        // Split into mean and logvar along last axis
        let parts = ops::split(&h, 2, Some(-1))?;
        let mean = parts[0].clone();
        let logvar = parts[1].clone();

        Ok((mean, logvar))
    }

    /// Encode image to latent codes (sample from distribution)
    ///
    /// # Arguments
    /// * `x` - Input image [batch, height, width, 3] in range [-1, 1]
    ///
    /// # Returns
    /// Latent codes [batch, h/8, w/8, z_channels]
    pub fn encode(&mut self, x: &Array) -> Result<Array, Exception> {
        let (mean, logvar) = self.forward(x)?;

        // Sample from distribution: z = mean + std * noise
        let std = ops::multiply(&logvar, &array!(0.5f32))?;
        let std = ops::exp(&std)?;

        let noise = mlx_rs::random::normal::<f32>(mean.shape(), None, None, None)?;
        let z = ops::add(&mean, &ops::multiply(&std, &noise)?)?;

        // Apply scaling (inverse of decoder's unscaling)
        let z = ops::subtract(&z, &array!(self.config.shift_factor))?;
        let z = ops::multiply(&z, &array!(self.config.scale_factor))?;

        Ok(z)
    }

    /// Encode image to latent codes (deterministic, use mean only)
    pub fn encode_deterministic(&mut self, x: &Array) -> Result<Array, Exception> {
        let (mean, _logvar) = self.forward(x)?;

        // Apply scaling
        let z = ops::subtract(&mean, &array!(self.config.shift_factor))?;
        let z = ops::multiply(&z, &array!(self.config.scale_factor))?;

        Ok(z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resnet_block() {
        let mut block = ResnetBlock::new(64, 128).unwrap();
        let x = mlx_rs::random::uniform::<_, f32>(0.0, 1.0, &[1, 8, 8, 64], None).unwrap();
        let out = block.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 8, 128]);
    }

    #[test]
    fn test_resnet_block_same_channels() {
        let mut block = ResnetBlock::new(64, 64).unwrap();
        let x = mlx_rs::random::uniform::<_, f32>(0.0, 1.0, &[1, 8, 8, 64], None).unwrap();
        let out = block.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 8, 64]);
    }

    #[test]
    fn test_decoder_config() {
        let config = AutoEncoderConfig::default();
        assert_eq!(config.z_channels, 16);
        assert_eq!(config.ch_mult, vec![1, 2, 4, 4]);
    }

    #[test]
    fn test_encoder() {
        let config = AutoEncoderConfig::flux2();
        let mut encoder = Encoder::new(config).unwrap();
        let x = mlx_rs::random::uniform::<_, f32>(-1.0, 1.0, &[1, 64, 64, 3], None).unwrap();
        let z = encoder.encode_deterministic(&x).unwrap();
        // 64/8 = 8
        assert_eq!(z.shape(), &[1, 8, 8, 32]);
    }
}
