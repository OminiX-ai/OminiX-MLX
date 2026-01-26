//! Check LLM embedding values.

use funasr_nano_mlx::model::FunASRNano;
use mlx_rs::transforms::eval;
use mlx_rs::Array;

fn main() {
    let model_dir = "./Fun-ASR-Nano-2512";
    println!("Loading model from {}...", model_dir);
    let mut model = FunASRNano::load(model_dir).expect("Failed to load model");

    // Get embedding for some sample tokens
    let test_tokens = Array::from_slice(&[151646, 151647, 1, 100, 1000], &[1, 5]);
    let embeddings = model.llm.get_token_embeddings(&test_tokens).expect("Failed to get embeddings");
    eval([&embeddings]).expect("eval failed");

    println!("Token embeddings: shape={:?}, range=[{:.4}, {:.4}]",
        embeddings.shape(),
        embeddings.min(None).expect("min").item::<f32>(),
        embeddings.max(None).expect("max").item::<f32>());

    // Check embedding mean
    let mean = embeddings.mean(None).expect("mean");
    eval([&mean]).expect("eval failed");
    println!("Embedding mean={:.4}", mean.item::<f32>());
}
