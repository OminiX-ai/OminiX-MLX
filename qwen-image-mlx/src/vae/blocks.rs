//! VAE building blocks: ResBlock3D, MidBlock3D, UpBlock3D, DownBlock3D
//!
//! Reference: diffusers QwenImageResidualBlock, QwenImageMidBlock, QwenImageUpBlock

use mlx_macros::ModuleParameters;
use mlx_rs::error::Exception;
use mlx_rs::module::Module;
use mlx_rs::nn;
use mlx_rs::ops;
use mlx_rs::Array;

use super::{QwenImageAttentionBlock3D, QwenImageCausalConv3D, QwenImageRMSNorm, QwenImageResample3D, ResampleMode};

/// 3D Residual Block for VAE
/// Reference: diffusers QwenImageResidualBlock
/// "RMS normalization, causal 3D convolutions, and optional dropout"
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenImageResBlock3D {
    pub in_channels: i32,
    pub out_channels: i32,
    #[param]
    pub norm1: QwenImageRMSNorm,
    #[param]
    pub conv1: QwenImageCausalConv3D,
    #[param]
    pub norm2: QwenImageRMSNorm,
    #[param]
    pub conv2: QwenImageCausalConv3D,
    #[param]
    pub skip: Option<QwenImageCausalConv3D>, // Only if in_channels != out_channels
}

impl QwenImageResBlock3D {
    pub fn new(in_channels: i32, out_channels: i32) -> Result<Self, Exception> {
        let norm1 = QwenImageRMSNorm::new(in_channels, 1e-12, false)?;
        let conv1 = QwenImageCausalConv3D::new(
            in_channels,
            out_channels,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            true,
        )?;
        let norm2 = QwenImageRMSNorm::new(out_channels, 1e-12, false)?;
        let conv2 = QwenImageCausalConv3D::new(
            out_channels,
            out_channels,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            true,
        )?;
        let skip = if in_channels != out_channels {
            Some(QwenImageCausalConv3D::new(
                in_channels,
                out_channels,
                (1, 1, 1),
                (1, 1, 1),
                (0, 0, 0),
                true,
            )?)
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
            skip,
        })
    }
}

impl Module<&Array> for QwenImageResBlock3D {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, mode: bool) {
        self.norm1.training_mode(mode);
        self.conv1.training_mode(mode);
        self.norm2.training_mode(mode);
        self.conv2.training_mode(mode);
        if let Some(ref mut skip) = self.skip {
            skip.training_mode(mode);
        }
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        // First block: norm -> silu -> conv
        let h = self.norm1.forward(x)?;
        let h = nn::silu(&h)?;
        let h = self.conv1.forward(&h)?;

        // Second block: norm -> silu -> conv
        let h = self.norm2.forward(&h)?;
        let h = nn::silu(&h)?;
        let h = self.conv2.forward(&h)?;

        // Residual connection
        let residual = if let Some(ref mut skip) = self.skip {
            skip.forward(x)?
        } else {
            x.clone()
        };

        ops::add(&h, &residual)
    }
}

/// Mid block with alternating residual and attention blocks
/// Reference: diffusers QwenImageMidBlock
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenImageMidBlock3D {
    #[param]
    pub resnets: Vec<QwenImageResBlock3D>,
    #[param]
    pub attentions: Vec<QwenImageAttentionBlock3D>,
}

impl QwenImageMidBlock3D {
    pub fn new(channels: i32, attention_layers: i32) -> Result<Self, Exception> {
        let mut resnets = vec![QwenImageResBlock3D::new(channels, channels)?];
        for _ in 0..attention_layers {
            resnets.push(QwenImageResBlock3D::new(channels, channels)?);
        }
        let attentions = (0..attention_layers)
            .map(|_| QwenImageAttentionBlock3D::new(channels))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { resnets, attentions })
    }
}

impl Module<&Array> for QwenImageMidBlock3D {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, mode: bool) {
        for res in &mut self.resnets {
            res.training_mode(mode);
        }
        for attn in &mut self.attentions {
            attn.training_mode(mode);
        }
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let mut h = self.resnets[0].forward(x)?;
        for (attn, res) in self.attentions.iter_mut().zip(self.resnets[1..].iter_mut()) {
            h = attn.forward(&h)?;
            h = res.forward(&h)?;
        }
        Ok(h)
    }
}

/// Up block for decoder
/// Reference: diffusers QwenImageUpBlock
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenImageUpBlock3D {
    #[param]
    pub resnets: Vec<QwenImageResBlock3D>,
    #[param]
    pub upsamplers: Vec<QwenImageResample3D>,
}

impl QwenImageUpBlock3D {
    pub fn new(
        in_channels: i32,
        out_channels: i32,
        num_res_blocks: i32,
        upsample_mode: Option<ResampleMode>,
    ) -> Result<Self, Exception> {
        let mut resnets = Vec::new();
        for i in 0..=num_res_blocks {
            let in_ch = if i == 0 { in_channels } else { out_channels };
            resnets.push(QwenImageResBlock3D::new(in_ch, out_channels)?);
        }
        let upsamplers = if let Some(mode) = upsample_mode {
            vec![QwenImageResample3D::new(out_channels, mode)?]
        } else {
            vec![]
        };
        Ok(Self { resnets, upsamplers })
    }
}

impl Module<&Array> for QwenImageUpBlock3D {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, mode: bool) {
        for res in &mut self.resnets {
            res.training_mode(mode);
        }
        for up in &mut self.upsamplers {
            up.training_mode(mode);
        }
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let mut h = x.clone();
        for res in &mut self.resnets {
            h = res.forward(&h)?;
        }
        if let Some(up) = self.upsamplers.first_mut() {
            h = up.forward(&h)?;
        }
        Ok(h)
    }
}

/// Down block for encoder
#[derive(Debug, Clone, ModuleParameters)]
pub struct QwenImageDownBlock3D {
    #[param]
    pub resnets: Vec<QwenImageResBlock3D>,
    #[param]
    pub downsamplers: Vec<QwenImageResample3D>,
}

impl QwenImageDownBlock3D {
    pub fn new(
        in_channels: i32,
        out_channels: i32,
        num_res_blocks: i32,
        downsample_mode: Option<ResampleMode>,
    ) -> Result<Self, Exception> {
        let mut resnets = Vec::new();
        let mut current_ch = in_channels;
        for _ in 0..num_res_blocks {
            resnets.push(QwenImageResBlock3D::new(current_ch, out_channels)?);
            current_ch = out_channels;
        }
        let downsamplers = if let Some(mode) = downsample_mode {
            vec![QwenImageResample3D::new(out_channels, mode)?]
        } else {
            vec![]
        };
        Ok(Self {
            resnets,
            downsamplers,
        })
    }
}

impl Module<&Array> for QwenImageDownBlock3D {
    type Output = Array;
    type Error = Exception;

    fn training_mode(&mut self, mode: bool) {
        for res in &mut self.resnets {
            res.training_mode(mode);
        }
        for down in &mut self.downsamplers {
            down.training_mode(mode);
        }
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let mut h = x.clone();
        for res in &mut self.resnets {
            h = res.forward(&h)?;
        }
        if let Some(down) = self.downsamplers.first_mut() {
            h = down.forward(&h)?;
        }
        Ok(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_res_block_same_channels() {
        let block = QwenImageResBlock3D::new(64, 64).unwrap();
        let x = Array::zeros::<f32>(&[1, 64, 1, 8, 8]).unwrap();
        let mut block = block;
        let y = block.forward(&x).unwrap();
        assert_eq!(y.shape(), x.shape());
    }

    #[test]
    fn test_res_block_diff_channels() {
        let block = QwenImageResBlock3D::new(64, 128).unwrap();
        let x = Array::zeros::<f32>(&[1, 64, 1, 8, 8]).unwrap();
        let mut block = block;
        let y = block.forward(&x).unwrap();
        assert_eq!(y.dim(1), 128);
    }

    #[test]
    fn test_mid_block() {
        let block = QwenImageMidBlock3D::new(64, 1).unwrap();
        let x = Array::zeros::<f32>(&[1, 64, 1, 8, 8]).unwrap();
        let mut block = block;
        let y = block.forward(&x).unwrap();
        assert_eq!(y.shape(), x.shape());
    }
}
