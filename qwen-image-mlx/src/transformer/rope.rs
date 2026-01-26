//! 3-axis Rotary Position Embedding for Qwen-Image
//!
//! Reference: diffusers QwenEmbedRope
//! "Implements rotary embeddings for video/image sequences with frame, height, and width dimensions"

use mlx_rs::error::Exception;
use mlx_rs::ops;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::Dtype;
use mlx_rs::Array;

/// 3-axis Rotary Position Embedding for Qwen-Image
/// Different from Z-Image: theta=10000, axes=[16, 56, 56], scaled=true
#[derive(Debug, Clone)]
pub struct QwenEmbedRope {
    pub theta: i32,
    pub axes_dimensions: [i32; 3], // [16, 56, 56] for (frame, height, width)
    pub scale_rope: bool,

    // Pre-computed frequencies
    positive_cos: Vec<Array>,
    positive_sin: Vec<Array>,
    negative_cos: Vec<Array>,
    negative_sin: Vec<Array>,
}

impl QwenEmbedRope {
    const MAX_INDEX: i32 = 4096;

    pub fn new(theta: i32, axes_dimensions: [i32; 3], scale_rope: bool) -> Result<Self, Exception> {
        let positive_indices: Vec<i32> = (0..Self::MAX_INDEX).collect();
        let negative_indices: Vec<i32> = positive_indices.iter().rev().map(|i| -(i + 1) as i32).collect();

        let mut positive_cos = Vec::with_capacity(3);
        let mut positive_sin = Vec::with_capacity(3);
        let mut negative_cos = Vec::with_capacity(3);
        let mut negative_sin = Vec::with_capacity(3);

        for &dim in axes_dimensions.iter() {
            let (pc, ps) = Self::compute_rope_params(&positive_indices, dim, theta)?;
            let (nc, ns) = Self::compute_rope_params(&negative_indices, dim, theta)?;
            positive_cos.push(pc);
            positive_sin.push(ps);
            negative_cos.push(nc);
            negative_sin.push(ns);
        }

        Ok(Self {
            theta,
            axes_dimensions,
            scale_rope,
            positive_cos,
            positive_sin,
            negative_cos,
            negative_sin,
        })
    }

    fn compute_rope_params(
        indices: &[i32],
        dim: i32,
        theta: i32,
    ) -> Result<(Array, Array), Exception> {
        let half_dim = dim / 2;

        // omega = 1 / theta^(2i/dim)
        let omega: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / (theta as f32).powf(2.0 * i as f32 / dim as f32))
            .collect();

        let mut cos_vals = Vec::with_capacity(indices.len() * half_dim as usize);
        let mut sin_vals = Vec::with_capacity(indices.len() * half_dim as usize);

        for &idx in indices {
            for &w in &omega {
                let angle = idx as f32 * w;
                cos_vals.push(angle.cos());
                sin_vals.push(angle.sin());
            }
        }

        let cos = Array::from_slice(&cos_vals, &[indices.len() as i32, half_dim]);
        let sin = Array::from_slice(&sin_vals, &[indices.len() as i32, half_dim]);
        Ok((
            cos.as_dtype(Dtype::Float32)?,
            sin.as_dtype(Dtype::Float32)?,
        ))
    }

    /// Build rotary embeddings for image and text
    /// Returns ((image_cos, image_sin), (text_cos, text_sin))
    pub fn forward(
        &self,
        video_segments: &[(i32, i32, i32)], // [(frame, height, width), ...]
        text_sequence_lengths: &[i32],
    ) -> Result<((Array, Array), (Array, Array)), Exception> {
        // Build image frequencies
        let mut cos_segments = Vec::new();
        let mut sin_segments = Vec::new();
        let mut max_video_index = 0;

        for (seg_idx, &(frame, height, width)) in video_segments.iter().enumerate() {
            let (seg_cos, seg_sin) =
                self.build_video_frequencies(frame, height, width, seg_idx as i32)?;
            cos_segments.push(seg_cos);
            sin_segments.push(seg_sin);

            let candidate = if self.scale_rope {
                (height / 2).max(width / 2)
            } else {
                height.max(width)
            };
            max_video_index = max_video_index.max(candidate);
        }

        let img_cos = if cos_segments.len() == 1 {
            cos_segments.pop().unwrap()
        } else {
            let refs: Vec<&Array> = cos_segments.iter().collect();
            ops::concatenate_axis(&refs, 0)?
        };
        let img_sin = if sin_segments.len() == 1 {
            sin_segments.pop().unwrap()
        } else {
            let refs: Vec<&Array> = sin_segments.iter().collect();
            ops::concatenate_axis(&refs, 0)?
        };

        // Build text frequencies
        let text_len = text_sequence_lengths.iter().max().copied().unwrap_or(0);
        let cos_refs: Vec<&Array> = self.positive_cos.iter().collect();
        let sin_refs: Vec<&Array> = self.positive_sin.iter().collect();
        let cos_full = ops::concatenate_axis(&cos_refs, 1)?;
        let sin_full = ops::concatenate_axis(&sin_refs, 1)?;

        let start = max_video_index;
        let end = start + text_len;
        let text_cos = cos_full.index((start..end, ..));
        let text_sin = sin_full.index((start..end, ..));

        Ok(((img_cos, img_sin), (text_cos, text_sin)))
    }

    fn build_video_frequencies(
        &self,
        frame: i32,
        height: i32,
        width: i32,
        frame_offset: i32,
    ) -> Result<(Array, Array), Exception> {
        let [dim_f, dim_h, dim_w] = self.axes_dimensions;

        // Frame frequencies
        let cos_f = self.positive_cos[0]
            .index((frame_offset..frame_offset + frame, ..))
            .reshape(&[frame, 1, 1, dim_f / 2])?;
        let sin_f = self.positive_sin[0]
            .index((frame_offset..frame_offset + frame, ..))
            .reshape(&[frame, 1, 1, dim_f / 2])?;

        // Height frequencies (with optional scaling)
        let (cos_h, sin_h) = if self.scale_rope {
            let half = height / 2;
            let pos = self.positive_cos[1].index((0..half, ..));
            let pos_sin = self.positive_sin[1].index((0..half, ..));
            let neg_len = height - half;
            let neg_start = Self::MAX_INDEX - neg_len;
            let neg = self.negative_cos[1].index((neg_start..neg_start + neg_len, ..));
            let neg_sin = self.negative_sin[1].index((neg_start..neg_start + neg_len, ..));
            (
                ops::concatenate_axis(&[&neg, &pos], 0)?,
                ops::concatenate_axis(&[&neg_sin, &pos_sin], 0)?,
            )
        } else {
            (
                self.positive_cos[1].index((0..height, ..)),
                self.positive_sin[1].index((0..height, ..)),
            )
        };
        let cos_h = cos_h.reshape(&[1, height, 1, dim_h / 2])?;
        let sin_h = sin_h.reshape(&[1, height, 1, dim_h / 2])?;

        // Width frequencies (similar to height)
        let (cos_w, sin_w) = if self.scale_rope {
            let half = width / 2;
            let pos = self.positive_cos[2].index((0..half, ..));
            let pos_sin = self.positive_sin[2].index((0..half, ..));
            let neg_len = width - half;
            let neg_start = Self::MAX_INDEX - neg_len;
            let neg = self.negative_cos[2].index((neg_start..neg_start + neg_len, ..));
            let neg_sin = self.negative_sin[2].index((neg_start..neg_start + neg_len, ..));
            (
                ops::concatenate_axis(&[&neg, &pos], 0)?,
                ops::concatenate_axis(&[&neg_sin, &pos_sin], 0)?,
            )
        } else {
            (
                self.positive_cos[2].index((0..width, ..)),
                self.positive_sin[2].index((0..width, ..)),
            )
        };
        let cos_w = cos_w.reshape(&[1, 1, width, dim_w / 2])?;
        let sin_w = sin_w.reshape(&[1, 1, width, dim_w / 2])?;

        // Tile and concatenate
        let cos_f_tiled = ops::tile(&cos_f, &[1, height, width, 1])?;
        let cos_h_tiled = ops::tile(&cos_h, &[frame, 1, width, 1])?;
        let cos_w_tiled = ops::tile(&cos_w, &[frame, height, 1, 1])?;
        let cos = ops::concatenate_axis(&[&cos_f_tiled, &cos_h_tiled, &cos_w_tiled], -1)?;

        let sin_f_tiled = ops::tile(&sin_f, &[1, height, width, 1])?;
        let sin_h_tiled = ops::tile(&sin_h, &[frame, 1, width, 1])?;
        let sin_w_tiled = ops::tile(&sin_w, &[frame, height, 1, 1])?;
        let sin = ops::concatenate_axis(&[&sin_f_tiled, &sin_h_tiled, &sin_w_tiled], -1)?;

        // Flatten to [F*H*W, dim]
        let total = frame * height * width;
        let dim = cos.dim(3);
        Ok((cos.reshape(&[total, dim])?, sin.reshape(&[total, dim])?))
    }
}

/// Apply RoPE to query and key tensors
/// Uses even/odd split method (mathematically equivalent to rotation matrix)
pub fn apply_rope(
    q: &Array,
    k: &Array,
    rotary: &(Array, Array),
) -> Result<(Array, Array), Exception> {
    let (cos, sin) = rotary;

    // Get dimensions
    let head_dim = q.dim(3);
    let half_dim = head_dim / 2;

    // Split into even/odd pairs
    let q_even = q.index((.., .., .., 0..half_dim));
    let q_odd = q.index((.., .., .., half_dim..));
    let k_even = k.index((.., .., .., 0..half_dim));
    let k_odd = k.index((.., .., .., half_dim..));

    // Expand cos/sin for broadcasting: [seq, dim/2] -> [1, 1, seq, dim/2]
    let cos = cos.expand_dims(0)?.expand_dims(0)?;
    let sin = sin.expand_dims(0)?.expand_dims(0)?;

    // Apply rotation: [cos, -sin; sin, cos] @ [even; odd]
    let _neg_sin = ops::negative(&sin)?;
    let q_rot_even = ops::subtract(
        &ops::multiply(&q_even, &cos)?,
        &ops::multiply(&q_odd, &sin)?,
    )?;
    let q_rot_odd = ops::add(
        &ops::multiply(&q_even, &sin)?,
        &ops::multiply(&q_odd, &cos)?,
    )?;

    let k_rot_even = ops::subtract(
        &ops::multiply(&k_even, &cos)?,
        &ops::multiply(&k_odd, &sin)?,
    )?;
    let k_rot_odd = ops::add(
        &ops::multiply(&k_even, &sin)?,
        &ops::multiply(&k_odd, &cos)?,
    )?;

    // Concatenate back
    let q_rot = ops::concatenate_axis(&[&q_rot_even, &q_rot_odd], -1)?;
    let k_rot = ops::concatenate_axis(&[&k_rot_even, &k_rot_odd], -1)?;

    Ok((q_rot, k_rot))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_creation() {
        let rope = QwenEmbedRope::new(10000, [16, 56, 56], true).unwrap();
        assert_eq!(rope.axes_dimensions, [16, 56, 56]);
    }

    #[test]
    fn test_rope_forward() {
        let rope = QwenEmbedRope::new(10000, [16, 56, 56], true).unwrap();
        let ((img_cos, img_sin), (txt_cos, txt_sin)) = rope.forward(&[(1, 8, 8)], &[10]).unwrap();
        assert_eq!(img_cos.dim(0), 64); // 1 * 8 * 8
        assert_eq!(txt_cos.dim(0), 10);
    }

    #[test]
    fn test_apply_rope() {
        let q = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
        let k = Array::zeros::<f32>(&[1, 4, 10, 64]).unwrap();
        let cos = Array::zeros::<f32>(&[10, 32]).unwrap();
        let sin = Array::zeros::<f32>(&[10, 32]).unwrap();
        let (q_rot, k_rot) = apply_rope(&q, &k, &(cos, sin)).unwrap();
        assert_eq!(q_rot.shape(), q.shape());
        assert_eq!(k_rot.shape(), k.shape());
    }
}
