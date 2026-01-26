//! Compare Rust g2p_en vs Python g2p_en

fn main() {
    use gpt_sovits_mlx::text::{g2p_en, cmudict};

    let words = ["resturant", "steak", "vegetable", "salad", "price", "hello", "world"];

    println!("=== Rust g2p_en ===");
    for word in &words {
        let in_cmu = cmudict::lookup(word).is_some();
        let phonemes = g2p_en::word_to_phonemes(word);
        let source = if in_cmu { "CMU" } else { "Neural" };
        println!("{:12} [{:6}]: {}", word, source, phonemes.join(" "));
    }

    // Additional test for problematic words
    let test_words = ["and", "The", "Economist", "Bankers", "Gazette", "Railway", "Monitor"];
    println!("\n=== Problem Words ===");
    for word in &test_words {
        let in_cmu = cmudict::lookup(word).is_some();
        let phonemes = g2p_en::word_to_phonemes(word);
        let source = if in_cmu { "CMU" } else { "Neural" };
        println!("{:12} [{:6}]: {}", word, source, phonemes.join(" "));
    }
}
