//! Debug script to compare Python and Rust processing of "合并，并将"
//!
//! This helps identify where the pronunciation diverges.

use gpt_sovits_mlx::text::preprocessor::{TextPreprocessor, PreprocessorConfig};
use gpt_sovits_mlx::text::symbols::symbol_to_id;

fn main() {
    // Test text that causes "he1 bing4" instead of "he2 bing4"
    let test_texts = [
        "合并",                    // Works correctly
        "合并两个文件",            // Works correctly
        "合并之后将",              // Works correctly
        "合并，并将",              // Sounds like he1 bing4
        "合并。并将",              // Sounds like he1 bing4
        "合并，并",                // Minimal failing case
    ];

    // Use the SAME config as inference::preprocess_text (NO BOS/EOS)
    let config = PreprocessorConfig {
        add_bos: false,
        add_eos: false,
        ..PreprocessorConfig::default()
    };
    let preprocessor = TextPreprocessor::new(config);

    for text in &test_texts {
        println!("\n{}", "=".repeat(60));
        println!("Text: {}", text);
        println!("{}", "=".repeat(60));

        let output = preprocessor.preprocess(text, None);

        println!("\nNormalized text: {}", output.text_normalized);
        println!("Phonemes: {:?}", output.phonemes);
        println!("word2ph: {:?}", output.word2ph);

        // Show phoneme IDs
        let phoneme_ids: Vec<i32> = output.phonemes.iter()
            .map(|p| symbol_to_id(p))
            .collect();
        println!("Phoneme IDs: {:?}", phoneme_ids);

        // Detailed breakdown
        println!("\nDetailed breakdown:");
        let mut ph_idx = 0;
        for (i, (w2p, c)) in output.word2ph.iter()
            .zip(output.text_normalized.chars())
            .enumerate()
        {
            let count = *w2p as usize;
            let phs: Vec<&str> = output.phonemes[ph_idx..ph_idx + count]
                .iter()
                .map(|s| s.as_str())
                .collect();
            let ids: Vec<i32> = phs.iter().map(|p| symbol_to_id(p)).collect();
            println!("  [{}] '{}': {:?} -> IDs {:?}", i, c, phs, ids);
            ph_idx += count;
        }

        // Check for "e2" vs "e1" in phonemes
        let has_e2 = output.phonemes.iter().any(|p| p == "e2");
        let has_e1 = output.phonemes.iter().any(|p| p == "e1");
        println!("\nTone check:");
        println!("  Contains e2 (tone 2): {}", has_e2);
        println!("  Contains e1 (tone 1): {}", has_e1);

        // Show specific IDs
        println!("\nKey symbol IDs:");
        println!("  e1 = {}", symbol_to_id("e1"));
        println!("  e2 = {}", symbol_to_id("e2"));
        println!("  h = {}", symbol_to_id("h"));
        println!("  b = {}", symbol_to_id("b"));
        println!("  ing4 = {}", symbol_to_id("ing4"));
    }

    // Show what the model sees
    println!("\n\n{}", "=".repeat(60));
    println!("COMPARISON: Phoneme sequence differences");
    println!("{}", "=".repeat(60));

    println!("\nFor '合并' (working):");
    let output1 = preprocessor.preprocess("合并", None);
    println!("  Phonemes: {:?}", output1.phonemes);

    println!("\nFor '合并，并' (failing):");
    let output2 = preprocessor.preprocess("合并，并", None);
    println!("  Phonemes: {:?}", output2.phonemes);

    // The key difference is the context after 合并
    // In Python, the BERT features provide context
    // Let's check if word2ph affects feature alignment
    println!("\nword2ph comparison:");
    println!("  '合并': {:?}", output1.word2ph);
    println!("  '合并，并': {:?}", output2.word2ph);
}
