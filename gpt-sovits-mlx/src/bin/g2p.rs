//! CLI tool to run G2P on Chinese text - outputs JSON like Python
//!
//! Usage: cargo run --features jieba --bin g2p -- "你好世界"

use gpt_sovits_mlx::text::preprocessor::{chinese_g2p, normalize_chinese};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <text>", args[0]);
        std::process::exit(1);
    }

    let text = &args[1];
    let normalized = normalize_chinese(text);
    let (phones, word2ph) = chinese_g2p(&normalized);

    // Output JSON like Python
    println!(
        r#"{{"input": "{}", "normalized": "{}", "phones": {:?}, "word2ph": {:?}}}"#,
        text.replace('"', "\\\""),
        normalized.replace('"', "\\\""),
        phones,
        word2ph
    );
}
