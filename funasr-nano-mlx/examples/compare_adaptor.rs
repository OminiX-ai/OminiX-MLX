//! Compare adaptor output with Python reference.

use funasr_nano_mlx::adaptor::AudioAdaptor;
use funasr_nano_mlx::model::FunASRNano;
use mlx_rs::module::{Module, ModuleParameters as ModuleParametersTrait};
use mlx_rs::transforms::eval;
use mlx_rs::Array;

fn main() {
    println!("=== Comparing Rust adaptor with Python reference ===\n");

    // Load the model to get properly initialized adaptor
    let model_dir = "./Fun-ASR-Nano-2512";
    println!("Loading model from {}...", model_dir);
    let mut model = FunASRNano::load(model_dir).expect("Failed to load model");

    // Load the encoder output from Python
    let encoder_out_path = "/tmp/encoder_out.safetensors";
    let tensors = Array::load_safetensors(encoder_out_path)
        .expect("Failed to load encoder_out.safetensors");
    let encoder_out = tensors.get("encoder_out").expect("No encoder_out key");

    eval([encoder_out]).expect("eval failed");
    println!("Encoder output: shape={:?}, range=[{:.4}, {:.4}]",
        encoder_out.shape(),
        encoder_out.min(None).expect("min").item::<f32>(),
        encoder_out.max(None).expect("max").item::<f32>());

    // Run through adaptor step by step
    println!("\n=== Rust adaptor step by step ===");

    // Step 1: linear1 + relu
    let h = model.adaptor.linear1.forward(encoder_out).expect("linear1 failed");
    let h = mlx_rs::nn::relu(&h).expect("relu failed");
    eval([&h]).expect("eval failed");
    println!("After linear1+relu: range=[{:.4}, {:.4}]",
        h.min(None).expect("min").item::<f32>(),
        h.max(None).expect("max").item::<f32>());

    // Step 2: linear2
    let h = model.adaptor.linear2.forward(&h).expect("linear2 failed");
    eval([&h]).expect("eval failed");
    println!("After linear2: range=[{:.4}, {:.4}]",
        h.min(None).expect("min").item::<f32>(),
        h.max(None).expect("max").item::<f32>());

    // Step 3: Transformer blocks
    let mut h = h;
    for (i, block) in model.adaptor.blocks.iter_mut().enumerate() {
        h = block.forward(&h).expect("block failed");
        eval([&h]).expect("eval failed");
        println!("After block {}: range=[{:.4}, {:.4}]",
            i,
            h.min(None).expect("min").item::<f32>(),
            h.max(None).expect("max").item::<f32>());
    }

    println!("\nAdaptor output: shape={:?}, range=[{:.4}, {:.4}]",
        h.shape(),
        h.min(None).expect("min").item::<f32>(),
        h.max(None).expect("max").item::<f32>());

    // Compare with Python reference
    let python_ref_path = "/tmp/adaptor_out.safetensors";
    if let Ok(ref_tensors) = Array::load_safetensors(python_ref_path) {
        if let Some(python_out) = ref_tensors.get("adaptor_out") {
            eval([python_out]).expect("eval failed");
            println!("\n=== Python reference ===");
            println!("Adaptor output: shape={:?}, range=[{:.4}, {:.4}]",
                python_out.shape(),
                python_out.min(None).expect("min").item::<f32>(),
                python_out.max(None).expect("max").item::<f32>());

            // Compute difference
            let diff = h.subtract(python_out).expect("diff failed");
            let diff_abs = mlx_rs::ops::abs(&diff).expect("abs failed");
            eval([&diff_abs]).expect("eval failed");
            println!("\nDifference: max_abs={:.6}, mean_abs={:.6}",
                diff_abs.max(None).expect("max").item::<f32>(),
                diff_abs.mean(None).expect("mean").item::<f32>());
        }
    }
}
