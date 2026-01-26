fn main() {
    use gpt_sovits_mlx::text::{preprocess_text, preprocessor::chinese_g2p, symbols::symbol_to_id};
    use gpt_sovits_mlx::inference::preprocess_text as inference_preprocess;

    // Test problematic segment
    println!("=== Testing 及《鐵路觀察》 ===");
    let text = "及《鐵路觀察》";
    let output = preprocess_text(text, None);
    println!("Text: {}", text);
    println!("Normalized: '{}'", output.text_normalized);
    println!("Phonemes: {:?}", output.phonemes);
    println!("word2ph: {:?}", output.word2ph);
    println!();

    // Test 观察 specifically
    println!("=== Testing 观察 G2P ===");
    let (phonemes, word2ph) = chinese_g2p("观察");
    println!("Text: 观察");
    println!("Phonemes: {:?}", phonemes);
    println!("Word2ph: {:?}", word2ph);
    println!("Expected: g + uan1, ch + a2\n");

    // Show phone IDs
    println!("Phone IDs:");
    for (i, ph) in phonemes.iter().enumerate() {
        println!("  [{}] '{}' -> {}", i, ph, symbol_to_id(ph));
    }
    println!();

    // Test full preprocessing for 观察
    println!("=== Full preprocess 观察 ===");
    let output = preprocess_text("观察", None);
    println!("Normalized: {}", output.text_normalized);
    println!("Phonemes: {:?}", output.phonemes);
    // Convert phonemes to IDs
    let phone_ids: Vec<i32> = output.phonemes.iter().map(|p| symbol_to_id(p)).collect();
    println!("Phone IDs: {:?}", phone_ids);
    println!();

    // Test with prepended period (like voice_clone.rs does for short text)
    println!("=== Full preprocess .观察 (with prepended period) ===");
    let output = preprocess_text(".观察", None);
    println!("Normalized: {}", output.text_normalized);
    println!("Phonemes: {:?}", output.phonemes);
    let phone_ids: Vec<i32> = output.phonemes.iter().map(|p| symbol_to_id(p)).collect();
    println!("Phone IDs: {:?}", phone_ids);
    println!("Expected: '.' should be ID 3, not SP (77)");
    println!();

    // Test with inference::preprocess_text (actual function used in voice_clone.rs)
    println!("=== inference::preprocess_text .观察 (NO BOS/EOS) ===");
    let (phoneme_ids, phonemes, _word2ph, text_normalized) = inference_preprocess(".观察");
    println!("Normalized: {}", text_normalized);
    println!("Phonemes: {:?}", phonemes);
    let flat = phoneme_ids.flatten(None, None).expect("Failed to flatten");
    let ids_vec: Vec<i32> = flat.as_slice::<i32>().to_vec();
    println!("Phone IDs: {:?}", ids_vec);
    println!("Python expects: [3, 156, 270, 125, 98]");
    println!();

    let text = r#"1845年，在英国"铁路狂热"时期，该报一度与《银行公报》（Bankers' Gazette）及《铁路观察》（Railway Monitor）合并，并将刊名改为《经济学人：商业周报、银行公报及铁路观察——政治化和文学化的大众报纸》（The Economist, Weekly Commercial Times, Bankers' Gazette, and Railway Monitor. A Political, Literary and General Newspaper）"#;

    let output = preprocess_text(text, None);

    println!("=== Input Text ===");
    println!("{}", text);

    println!("\n=== Normalized Text ({} chars) ===", output.text_normalized.len());
    println!("{}", output.text_normalized);

    println!("\n=== Phonemes ({}) ===", output.phonemes.len());
    for (i, chunk) in output.phonemes.chunks(20).enumerate() {
        println!("[{:3}] {}", i*20, chunk.join(" "));
    }

    // Check for "er" phonemes
    println!("\n=== 'er' phonemes (could be 而) ===");
    for (i, ph) in output.phonemes.iter().enumerate() {
        if ph.starts_with("er") || ph == "EE" {
            let start = i.saturating_sub(3);
            let end = (i + 4).min(output.phonemes.len());
            let context: Vec<_> = output.phonemes[start..end].iter().map(|s| s.as_str()).collect();
            println!("[{:3}] '{}' in: {}", i, ph, context.join(" "));
        }
    }

    println!("\n=== Language: {:?} ===", output.language);
}
