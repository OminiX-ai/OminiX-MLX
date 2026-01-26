//! Debug weight loading.

use funasr_nano_mlx::model::FunASRNano;
use mlx_rs::module::ModuleParameters;

fn main() {
    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "./Fun-ASR-Nano-2512".to_string());

    println!("Loading model from {}...", model_dir);
    let model = FunASRNano::load(&model_dir).expect("Failed to load model");

    // Get all parameter names
    let params = model.parameters().flatten();

    // Count and display parameters
    let mut fsmn_count = 0;
    let mut fsmn_shapes = Vec::new();

    for (name, param) in params.iter() {
        if name.contains("fsmn") {
            fsmn_count += 1;
            if fsmn_shapes.len() < 10 {
                fsmn_shapes.push((name.to_string(), param.shape().to_vec()));
            }
        }
        if name.contains("after_norm") || name.contains("tp_norm") {
            println!("{} -> shape {:?}", name, param.shape());
        }
    }

    println!("\nTotal parameters: {}", params.len());
    println!("FSMN parameters: {}", fsmn_count);

    println!("\nFirst FSMN parameters:");
    for (name, shape) in &fsmn_shapes {
        println!("  {} -> shape {:?}", name, shape);
    }
}
