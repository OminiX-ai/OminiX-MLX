//! Debug BERT tokenization and feature alignment for "合并，并"
//!
//! This checks if the BERT features are correctly aligned with phonemes.

use std::path::Path;
use tokenizers::Tokenizer;

fn main() {
    // Load the BERT tokenizer
    let tokenizer_path = "~/.OminiX/models/gpt-sovits-mlx/chinese-roberta-tokenizer/tokenizer.json";

    if !Path::new(tokenizer_path).exists() {
        eprintln!("Tokenizer not found at {}", tokenizer_path);
        eprintln!("Please ensure the tokenizer is downloaded.");
        return;
    }

    let tokenizer = Tokenizer::from_file(tokenizer_path).expect("Failed to load tokenizer");

    // Test texts
    let test_texts = [
        "合并",
        "合并，并将",
        "合并两个文件",
    ];

    for text in &test_texts {
        println!("\n{}", "=".repeat(60));
        println!("Text: {} (len={})", text, text.chars().count());
        println!("{}", "=".repeat(60));

        // Tokenize
        let encoding = tokenizer.encode(*text, true).expect("Tokenization failed");
        let token_ids = encoding.get_ids();
        let tokens = encoding.get_tokens();

        println!("\nBERT Tokenization:");
        println!("  Token IDs: {:?}", token_ids);
        println!("  Tokens: {:?}", tokens);
        println!("  Num tokens (with CLS/SEP): {}", token_ids.len());
        println!("  Num tokens (without CLS/SEP): {}", token_ids.len() - 2);

        // Expected word2ph based on phoneme generation
        // (This should match what preprocess_text produces)
        let word2ph: Vec<i32> = text.chars().map(|c| {
            if c == '，' || c == '。' || c == ',' || c == '.' {
                1  // punctuation -> 1 phoneme (SP)
            } else {
                2  // Chinese char -> 2 phonemes (initial + final)
            }
        }).collect();

        println!("\nword2ph analysis:");
        println!("  Text chars: {}", text.chars().count());
        println!("  word2ph: {:?}", word2ph);
        println!("  Sum(word2ph): {} (total phonemes)", word2ph.iter().sum::<i32>());

        // Check alignment
        let bert_tokens_without_cls_sep = token_ids.len() - 2;
        let text_char_count = text.chars().count();

        println!("\nAlignment check:");
        println!("  BERT tokens (no CLS/SEP): {}", bert_tokens_without_cls_sep);
        println!("  Text characters: {}", text_char_count);
        println!("  word2ph length: {}", word2ph.len());

        if bert_tokens_without_cls_sep == text_char_count {
            println!("  ALIGNED: BERT tokens == text chars");
        } else {
            println!("  MISALIGNED: BERT tokens ({}) != text chars ({})",
                     bert_tokens_without_cls_sep, text_char_count);
            println!("  This may cause incorrect BERT feature alignment!");
        }

        // Show character-to-token mapping
        println!("\nDetailed token mapping:");
        for (i, (c, tok)) in text.chars().zip(tokens[1..tokens.len()-1].iter()).enumerate() {
            println!("  [{}] '{}' -> token '{}'", i, c, tok);
        }

        // If there's a mismatch, show what the BERT feature expansion will do
        if bert_tokens_without_cls_sep != word2ph.len() {
            println!("\nFeature expansion fallback will be used!");
            println!("  This may cause misalignment between BERT context and phonemes.");
        }
    }

    // Show key insight
    println!("\n{}", "=".repeat(60));
    println!("KEY INSIGHT");
    println!("{}", "=".repeat(60));
    println!();
    println!("If BERT token count != word2ph length, the Rust code uses a");
    println!("fallback that may misalign BERT features with phonemes.");
    println!();
    println!("For Chinese text, BERT tokenizer usually produces 1 token per");
    println!("Chinese character, but punctuation handling may differ.");
}
