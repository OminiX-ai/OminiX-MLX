//! M0 weight/config inspection: confirm layer 0/5 keys, v_proj presence,
//! per-module quant bits.
//!
//! Usage:
//!   cargo run -p gemma4-mlx --example inspect_gemma4 --release -- <MODEL_DIR>
//!
//! Validates design risks:
//!   R3 — do global (full-attention) layers lack v_proj?
//!   R6 — per-module quant bits correct (MLP=8-bit, attn=4-bit)?

use std::path::Path;

use gemma4_mlx::config::{ModelArgs, QuantConfig};
use gemma4_mlx::weights::weight_keys;

fn main() -> anyhow::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: inspect_gemma4 <MODEL_DIR>");
    let dir = Path::new(&dir);

    let cfg = std::fs::read_to_string(dir.join("config.json"))?;
    let args = ModelArgs::from_config_str(&cfg)?;
    let quant = QuantConfig::from_config_str(&cfg)?;
    let keys = weight_keys(dir)?;

    println!(
        "layers={}, head_dim={}, global_head_dim={}, k_eq_v={}",
        args.num_hidden_layers, args.head_dim, args.global_head_dim, args.attention_k_eq_v
    );
    println!("total weight keys: {}", keys.len());

    for li in [0usize, 5usize] {
        let kind = args.layer_types[li];
        let pfx = format!("language_model.model.layers.{li}");
        println!("\n--- layer {li} ({kind:?}) ---");

        for k in keys.iter().filter(|k| k.starts_with(&pfx) && k.ends_with(".weight")) {
            println!("  key: {k}");
        }

        let has_v = keys
            .iter()
            .any(|k| *k == format!("{pfx}.self_attn.v_proj.weight"));
        println!("  v_proj present: {has_v}");

        let has_layer_scalar = keys.iter().any(|k| *k == format!("{pfx}.layer_scalar"));
        println!("  layer_scalar present: {has_layer_scalar}");

        if let Some(q) = &quant {
            for m in [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "mlp.gate_proj",
                "mlp.down_proj",
            ] {
                let (b, g) = q.quant_for(&format!("{pfx}.{m}"));
                println!("  quant {m}: bits={b} group_size={g}");
            }
        }
    }

    Ok(())
}
