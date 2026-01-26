//! Check specific weight values match between model and safetensors.

use funasr_nano_mlx::model::FunASRNano;
use mlx_rs::module::ModuleParameters;
use mlx_rs::transforms::eval;
use mlx_rs::Dtype;
use mlx_rs::Array;

fn print_weight_stats(name: &str, arr: &Array) {
    let arr_f32 = arr.as_dtype(Dtype::Float32).unwrap();
    eval([&arr_f32]).unwrap();
    let data: Vec<f32> = arr_f32.try_as_slice().unwrap().to_vec();
    let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    let first5: Vec<f32> = data.iter().take(5).copied().collect();
    println!("{}: shape={:?}, min={:.4}, max={:.4}, mean={:.4}", name, arr.shape(), min, max, mean);
    println!("  first 5: {:?}", first5);
}

fn main() {
    println!("Loading model...");
    let model = FunASRNano::load("./Fun-ASR-Nano-2512").expect("Failed to load model");
    
    let params = model.parameters();
    let flattened = params.flatten();
    
    // Check encoders0 weights
    let keys = [
        "encoder.encoders0.0.norm1.weight",
        "encoder.encoders0.0.self_attn.linear_q_k_v.weight",
        "encoder.encoders0.0.self_attn.fsmn.weight",
        "encoder.encoders.0.self_attn.linear_q_k_v.weight",
    ];
    
    for key in keys {
        if let Some(arr) = flattened.get(key) {
            eval([*arr]).unwrap();
            print_weight_stats(key, arr);
        } else {
            println!("{}: NOT FOUND", key);
        }
    }
    
    // Now load directly from safetensors and compare
    println!("\n=== Direct from safetensors ===");
    let st_weights = Array::load_safetensors("./Fun-ASR-Nano-2512/model.safetensors")
        .expect("Failed to load safetensors");
    
    let st_keys = [
        "encoder.encoders0.0.norm1.weight",
        "encoder.encoders0.0.attn.qkv.weight",
        "encoder.encoders0.0.attn.fsmn.weight",
        "encoder.encoders.0.attn.qkv.weight",
    ];
    
    for key in st_keys {
        if let Some(arr) = st_weights.get(key) {
            eval([arr]).unwrap();
            print_weight_stats(key, arr);
        } else {
            println!("{}: NOT FOUND", key);
        }
    }
}
