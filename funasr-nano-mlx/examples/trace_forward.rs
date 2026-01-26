//! Trace forward pass to find explosion.

use funasr_nano_mlx::model::FunASRNano;
use funasr_nano_mlx::audio::{self, AudioConfig};
use mlx_rs::transforms::eval;
use mlx_rs::Dtype;
use mlx_rs::module::Module;

fn stats(name: &str, arr: &mlx_rs::Array) {
    let arr_f32 = arr.as_dtype(Dtype::Float32).unwrap();
    eval([&arr_f32]).unwrap();
    let data: Vec<f32> = arr_f32.try_as_slice().unwrap().to_vec();
    let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    println!("{}: min={:.4}, max={:.4}, mean={:.4}", name, min, max, mean);
}

fn main() {
    println!("Loading model...");
    let mut model = FunASRNano::load("./Fun-ASR-Nano-2512").expect("Failed to load model");
    
    // Get LFR input
    let (samples, sample_rate) = audio::load_wav("./Fun-ASR-Nano-2512/example/zh.wav").unwrap();
    let config = AudioConfig::default();
    let samples = audio::resample(&samples, sample_rate, config.sample_rate).unwrap();
    let mel = audio::compute_mel_spectrogram(&samples, &config).unwrap();
    let mel_lfr = audio::apply_lfr(&mel, 7, 6).unwrap();
    
    eval([&mel_lfr]).unwrap();
    stats("Input (LFR)", &mel_lfr);
    
    // Get first layer
    let layer = &mut model.encoder.encoders0[0];
    let x = &mel_lfr;
    
    // Step 1: LayerNorm
    let h = layer.norm1.forward(x).unwrap();
    eval([&h]).unwrap();
    stats("After norm1", &h);
    
    // Step 2: Self-attention QKV projection  
    let qkv = layer.self_attn.linear_q_k_v.forward(&h).unwrap();
    eval([&qkv]).unwrap();
    stats("After QKV proj", &qkv);
    
    // Step 3: Split Q, K, V
    let qkv_parts = mlx_rs::ops::split(&qkv, 3, -1).unwrap();
    let (q, k, v) = (&qkv_parts[0], &qkv_parts[1], &qkv_parts[2]);
    stats("Q", q);
    stats("V", v);
    
    // Step 4: FSMN on V
    let v_fsmn = layer.self_attn.fsmn.forward(v).unwrap();
    eval([&v_fsmn]).unwrap();
    stats("FSMN output", &v_fsmn);
    
    let v_with_fsmn = v.add(&v_fsmn).unwrap();
    eval([&v_with_fsmn]).unwrap();
    stats("V + FSMN", &v_with_fsmn);
    
    // Step 5: Reshape for attention
    let batch = 1i32;
    let seq_len = 94i32;
    let n_heads = 4i32;
    let head_dim = 128i32;
    
    let q_r = q.reshape(&[batch, seq_len, n_heads, head_dim]).unwrap();
    let k_r = k.reshape(&[batch, seq_len, n_heads, head_dim]).unwrap();
    let v_r = v_with_fsmn.reshape(&[batch, seq_len, n_heads, head_dim]).unwrap();
    
    let q_t = q_r.transpose_axes(&[0, 2, 1, 3]).unwrap();
    let k_t = k_r.transpose_axes(&[0, 2, 1, 3]).unwrap();
    let v_t = v_r.transpose_axes(&[0, 2, 1, 3]).unwrap();
    
    // Step 6: SDPA
    let scale = (head_dim as f32).powf(-0.5);
    let attn_out = mlx_rs::fast::scaled_dot_product_attention(
        q_t, k_t, v_t, scale, None
    ).unwrap();
    eval([&attn_out]).unwrap();
    stats("SDPA output", &attn_out);
    
    // Step 7: Reshape and output projection
    let attn_out = attn_out.transpose_axes(&[0, 2, 1, 3]).unwrap();
    let attn_out = attn_out.reshape(&[batch, seq_len, n_heads * head_dim]).unwrap();
    let attn_out = layer.self_attn.linear_out.forward(&attn_out).unwrap();
    eval([&attn_out]).unwrap();
    stats("After out proj", &attn_out);
}
