//! Verify weights are loaded correctly.

use funasr_nano_mlx::model::FunASRNano;
use mlx_rs::module::ModuleParameters;
use mlx_rs::transforms::eval;

fn main() {
    let model_dir = "./Fun-ASR-Nano-2512";
    
    println!("Loading model...");
    let model = FunASRNano::load(&model_dir).expect("Failed to load model");
    
    let params = model.parameters();
    let flattened = params.flatten();
    
    // Check some specific weights
    let keys_to_check = [
        "encoder.encoders0.0.self_attn.linear_q_k_v.weight",
        "encoder.encoders0.0.self_attn.fsmn.weight",
        "encoder.encoders.0.self_attn.linear_q_k_v.weight",
        "llm.embed_tokens.weight",
    ];
    
    for key in keys_to_check {
        if let Some(arr) = flattened.get(key) {
            eval([*arr]).unwrap();
            println!("\n{}:", key);
            println!("  shape: {:?}", arr.shape());
            
            let data: Vec<f32> = arr.try_as_slice().unwrap().to_vec();
            let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
            let std: f32 = (data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32).sqrt();
            
            println!("  min: {:.6}, max: {:.6}", min, max);
            println!("  mean: {:.6}, std: {:.6}", mean, std);
            
            // Print first few values
            println!("  first 5: {:?}", &data[..5.min(data.len())]);
        } else {
            println!("\n{}: NOT FOUND", key);
        }
    }
}
