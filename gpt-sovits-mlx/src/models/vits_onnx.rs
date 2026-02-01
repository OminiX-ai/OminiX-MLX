//! ONNX Runtime VITS backend for batched decode.
//!
//! Loads a VITS model exported to ONNX format and runs inference on CPU/CoreML.
//! This produces audio matching Python/PyTorch numerics, enabling batched decode
//! of all chunks in a single call (eliminating per-chunk noise artifacts).

use std::path::Path;

use ort::{inputs, session::Session, value::Tensor};

/// ONNX-based VITS decoder.
pub struct VitsOnnx {
    session: Session,
}

impl VitsOnnx {
    /// Load VITS ONNX model from file.
    pub fn load(onnx_path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let cache_dir = onnx_path.parent().unwrap_or(Path::new(".")).join("vits_coreml_cache");
        std::fs::create_dir_all(&cache_dir).ok();

        eprintln!("[VITS-ONNX] Loading from {}", onnx_path.display());

        // Use CPU execution provider (CoreML doesn't support complex VITS ops)
        let session = Session::builder()?
            .with_intra_threads(4)?
            .commit_from_file(onnx_path)?;

        eprintln!("[VITS-ONNX] Model loaded successfully");
        Ok(Self { session })
    }

    /// Run batched VITS decode.
    ///
    /// - `speed`: Speed factor (1.0 = normal, >1.0 = faster speech)
    ///   Applied via linear interpolation on output audio.
    pub fn decode(
        &mut self,
        codes: &[i32],
        text: &[i32],
        refer_data: &[f32],
        refer_channels: usize,
        refer_time: usize,
        noise_scale: f32,
        speed: f32,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // codes: [1, 1, T_codes] as i64
        let codes_i64: Vec<i64> = codes.iter().map(|&x| x as i64).collect();
        let codes_tensor = Tensor::from_array(([1usize, 1, codes.len()], codes_i64.into_boxed_slice()))?;

        // text: [1, T_text] as i64
        let text_i64: Vec<i64> = text.iter().map(|&x| x as i64).collect();
        let text_tensor = Tensor::from_array(([1usize, text.len()], text_i64.into_boxed_slice()))?;

        // refer: [1, refer_channels, T_refer] as f32
        let refer_tensor = Tensor::from_array(([1usize, refer_channels, refer_time], refer_data.to_vec().into_boxed_slice()))?;

        // noise_scale: scalar f32 (shape [])
        let noise_tensor = Tensor::from_array(ndarray::arr0(noise_scale))?;

        eprintln!("[VITS-ONNX] Running decode: codes={}, text={}, refer=[1,{},{}], noise_scale={}",
                 codes.len(), text.len(), refer_channels, refer_time, noise_scale);

        let outputs = self.session.run(inputs![
            "codes" => codes_tensor,
            "text" => text_tensor,
            "refer" => refer_tensor,
            "noise_scale" => noise_tensor,
        ])?;

        // Output: audio [1, 1, T_audio]
        let audio_value = &outputs[0];
        let (_, audio_data) = audio_value.try_extract_tensor::<f32>()?;
        let raw_samples: Vec<f32> = audio_data.to_vec();

        // WORKAROUND: The ONNX model produces much quieter output than the
        // Python model. Python's postprocessing normalizes: if max > threshold,
        // divide by max. We do similar normalization to fill the dynamic range.
        //
        // The ONNX model's raw output typically has max amplitude around 0.05-0.1
        // while Python produces 0.5-1.0. We normalize to 0.95 of full scale.
        // TODO: Fix the ONNX export process to produce correct output levels.
        let max_abs = raw_samples.iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);

        const TARGET_AMPLITUDE: f32 = 0.95;
        let scale = if max_abs > 0.001 {
            TARGET_AMPLITUDE / max_abs
        } else {
            1.0  // Avoid division by near-zero
        };

        let samples: Vec<f32> = raw_samples.iter()
            .map(|&s| (s * scale).clamp(-1.0, 1.0))
            .collect();

        eprintln!("[VITS-ONNX] Output: {} samples ({:.2}s at 32kHz), raw_max={:.4}, scale={:.2}",
                 samples.len(), samples.len() as f32 / 32000.0, max_abs, scale);

        // Apply speed factor via linear interpolation on audio
        // speed > 1.0 = faster (shorter), speed < 1.0 = slower (longer)
        let samples = if (speed - 1.0).abs() > 1e-6 {
            let new_len = (samples.len() as f32 / speed) as usize;
            resample_linear(&samples, new_len)
        } else {
            samples
        };

        Ok(samples)
    }
}

/// Linear interpolation resampling for audio
///
/// Note: This changes pitch along with speed. For pitch-preserving speed change,
/// use a proper time-stretching algorithm (WSOLA, phase vocoder, etc.)
fn resample_linear(samples: &[f32], new_len: usize) -> Vec<f32> {
    if new_len == 0 || samples.is_empty() {
        return Vec::new();
    }
    if new_len == samples.len() {
        return samples.to_vec();
    }

    let mut result = Vec::with_capacity(new_len);
    let scale = (samples.len() - 1) as f32 / (new_len - 1).max(1) as f32;

    for i in 0..new_len {
        let src_pos = i as f32 * scale;
        let left = src_pos.floor() as usize;
        let right = (left + 1).min(samples.len() - 1);
        let frac = src_pos - left as f32;

        let sample = samples[left] * (1.0 - frac) + samples[right] * frac;
        result.push(sample);
    }

    result
}
