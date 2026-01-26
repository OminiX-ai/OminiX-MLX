//! Debug preprocessor output for problematic text

use gpt_sovits_mlx::inference::preprocess_text;

fn main() {
    let texts = [
        "银行",           // Simplified - should be yin2 hang2
        "銀行",           // Traditional - should be yin2 hang2
        "银行公报",       // Bank gazette
        "銀行公報",       // Traditional
    ];

    for text in &texts {
        println!("\n{}", "=".repeat(60));
        println!("Text: {}", text);

        let (_, phonemes, word2ph, _) = preprocess_text(text);

        println!("Phonemes: {:?}", phonemes);

        // Check for hang vs xing
        let has_hang = phonemes.iter().any(|p| p == "ang2");
        let has_xing = phonemes.iter().any(|p| p == "ing2");

        if has_hang {
            println!("✅ 行 = hang2 (correct for bank)");
        }
        if has_xing {
            println!("⚠️  行 = xing2 (WRONG for bank)");
        }
    }
}
