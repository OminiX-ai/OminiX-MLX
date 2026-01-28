//! ONNX Runtime VITS backend for batched decode.
//!
//! Loads a VITS model exported to ONNX format and runs inference on CPU/CoreML.
//! This produces audio matching Python/PyTorch numerics, enabling batched decode
//! of all chunks in a single call (eliminating per-chunk noise artifacts).

use std::path::Path;

use ort::{ep, inputs, session::Session, value::Tensor};

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
    pub fn decode(
        &mut self,
        codes: &[i32],
        text: &[i32],
        refer_data: &[f32],
        refer_channels: usize,
        refer_time: usize,
        noise_scale: f32,
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
        let samples: Vec<f32> = audio_data.to_vec();

        eprintln!("[VITS-ONNX] Output: {} samples ({:.2}s at 32kHz)",
                 samples.len(), samples.len() as f32 / 32000.0);

        Ok(samples)
    }
}
