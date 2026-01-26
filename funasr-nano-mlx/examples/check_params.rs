//! Check parameter names in the model.

use funasr_nano_mlx::model::FunASRNanoConfig;
use funasr_nano_mlx::model::FunASRNano;
use mlx_rs::module::ModuleParameters;

fn main() {
    let config = FunASRNanoConfig::default();
    let model = FunASRNano::new(config).expect("Failed to create model");
    
    let params = model.parameters();
    let flattened = params.flatten();
    
    println!("Total parameters: {}", flattened.len());
    
    // Print all encoder params
    println!("\nAll encoder params:");
    let mut enc_params: Vec<_> = flattened.keys()
        .filter(|k| k.starts_with("encoder"))
        .collect();
    enc_params.sort();
    for k in &enc_params {
        println!("  {}", k);
    }
}
