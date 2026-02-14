// Compare pretrained vs finetuned weight_g values (magnitude in weight normalization)
use gpt_sovits_mlx::models::vits::load_vits_model;
use mlx_rs::{Array, transforms::eval, module::ModuleParameters};

fn main() {
    println!("Testing VITS weight normalization statistics...\n");

    let pre_path = "/Users/yuechen/.OminiX/models/gpt-sovits-mlx/vits_pretrained_v2.safetensors";
    let ft_path = "/tmp/vits_weightnorm.generator.safetensors";

    // Load pretrained
    println!("=== Pretrained Model ===");
    let model = load_vits_model(pre_path).unwrap();

    // Check ups.0 weight_g (magnitude)
    let ups0_g = model.dec.ups[0].weight_g.as_ref();
    eval([ups0_g]).unwrap();
    let ups0_g_sum: f32 = ups0_g.sum(false).unwrap().item();
    let ups0_g_mean: f32 = ups0_g.mean(false).unwrap().item();
    println!("  dec.ups.0.weight_g: sum={:.4}, mean={:.6}", ups0_g_sum, ups0_g_mean);

    // Also check the computed weight
    let ups0_weight = model.dec.ups[0].weight().unwrap();
    eval([&ups0_weight]).unwrap();
    let ups0_w_sum: f32 = ups0_weight.sum(false).unwrap().item();
    println!("  dec.ups.0.weight (computed): sum={:.2}", ups0_w_sum);

    // Load finetuned weights
    println!("\n=== Finetuned Model ===");
    let finetuned_weights = Array::load_safetensors(ft_path).unwrap();

    // Check ups.0.weight_g
    if let Some(ft_g) = finetuned_weights.get("dec.ups.0.weight_g") {
        eval([ft_g]).unwrap();
        let ft_g_sum: f32 = ft_g.sum(false).unwrap().item();
        let ft_g_mean: f32 = ft_g.mean(false).unwrap().item();
        println!("  dec.ups.0.weight_g: sum={:.4}, mean={:.6}", ft_g_sum, ft_g_mean);

        // Calculate drift
        let g_drift = ((ft_g_sum - ups0_g_sum).abs() / ups0_g_sum.abs()) * 100.0;
        println!("\n=== Weight_g Comparison ===");
        println!("  ups.0.weight_g drift: {:.1}%", g_drift);
        println!("  Pretrained: {:.4}, Finetuned: {:.4}", ups0_g_sum, ft_g_sum);

        if g_drift < 10.0 {
            println!("\n✓ Weight_g drift is minimal (<10%) - weight normalization working correctly");
        } else {
            println!("\n⚠ Weight_g drift is significant (>10%)");
        }
    } else {
        println!("  dec.ups.0.weight_g not found - checking merged weight...");

        // Try loading as merged weight
        if let Some(ft_w) = finetuned_weights.get("dec.ups.0.weight") {
            let transposed = ft_w.transpose_axes(&[1, 2, 0]).unwrap();
            eval([&transposed]).unwrap();
            let ft_w_sum: f32 = transposed.sum(false).unwrap().item();
            println!("  dec.ups.0.weight: sum={:.2}", ft_w_sum);

            let w_drift = ((ft_w_sum - ups0_w_sum).abs() / ups0_w_sum.abs()) * 100.0;
            println!("\n=== Weight Comparison ===");
            println!("  ups.0.weight drift: {:.1}%", w_drift);
        }
    }
}
