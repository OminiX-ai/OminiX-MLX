//! Load and test Fun-ASR-Nano model.

use funasr_nano_mlx::model::FunASRNano;

fn main() {
    println!("Loading Fun-ASR-Nano model...");

    let model_dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "./Fun-ASR-Nano-2512".to_string());

    match FunASRNano::load(&model_dir) {
        Ok(model) => {
            println!("\nModel loaded successfully!");
            println!("Encoder output dim: {}", model.encoder.output_dim());
            println!("LLM hidden size: {}", model.llm.hidden_size());
        }
        Err(e) => {
            eprintln!("Failed to load model: {}", e);
            std::process::exit(1);
        }
    }
}
