/// TDD test for E4B degenerate output bug.
///
/// Root causes found:
/// 1. Shared KV layers got wrong RoPE offset (cache already incremented by reference layer)
/// 2. compute_next didn't select last token, causing dimension accumulation
/// 3. layer_scalar was applied BEFORE PLE instead of AFTER (HF applies it last)
/// 4. Shared sliding layers skipped sliding window mask
///
/// This test generates a short response and checks for coherent output.
use std::{collections::HashSet, env, error::Error, path::PathBuf};

use gemma4_mlx::{
    load_model, load_tokenizer, Gemma4ChatTemplate, Generate, KVCache, EOS_TOKEN_IDS,
};
use mlx_rs::{
    ops::indexing::{IndexOp, NewAxis},
    Array,
};

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args: Vec<String> = env::args().collect();
    let model_dir = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/gemma-4-E4B-it"));

    eprintln!("[test_e4b_coherence] Loading model from {:?}", model_dir);
    let mut model = load_model(&model_dir)?;
    let tokenizer = load_tokenizer(&model_dir)?;
    let template = Gemma4ChatTemplate::load(&model_dir)?;

    assert!(
        model.args.num_kv_shared_layers > 0,
        "Expected E4B model with KV sharing"
    );
    eprintln!(
        "[test_e4b_coherence] Model: {} layers, {} shared, ple_dim={}",
        model.args.num_hidden_layers,
        model.args.num_kv_shared_layers,
        model.args.hidden_size_per_layer_input,
    );

    // Format with chat template for instruct model
    let prompt = "What is 2 + 2? Answer in one word.";
    let formatted = format!(
        "{}{}\n{}{}\n{}{}\n",
        template.tokens.bos,
        template.tokens.turn_start,
        "user\n",
        prompt,
        template.tokens.turn_end,
        template.tokens.turn_start,
    ) + "model\n";

    eprintln!("[test_e4b_coherence] Prompt: {prompt}");
    eprintln!("[test_e4b_coherence] Formatted: {:?}", &formatted[..formatted.len().min(100)]);

    let encoding = tokenizer.encode(formatted, false)?;
    let prompt_tokens = Array::from(encoding.get_ids()).index(NewAxis);
    eprintln!(
        "[test_e4b_coherence] Prompt tokens: {} tokens",
        encoding.get_ids().len()
    );

    let mut cache = Vec::<KVCache>::new();
    let generator = Generate::new(&mut model, &mut cache, 0.0, &prompt_tokens);

    let max_tokens = 64;
    let mut generated_ids: Vec<u32> = Vec::new();
    let mut generated_text = String::new();

    for token in generator.take(max_tokens) {
        let token = token?;
        let token_id = token.item::<u32>();
        generated_ids.push(token_id);
        if EOS_TOKEN_IDS.contains(&token_id) {
            break;
        }
        let piece = tokenizer.decode(&[token_id], true)?;
        generated_text.push_str(&piece);
    }

    eprintln!("[test_e4b_coherence] Generated {} tokens", generated_ids.len());
    eprintln!("[test_e4b_coherence] Token IDs: {:?}", &generated_ids[..generated_ids.len().min(20)]);
    eprintln!("[test_e4b_coherence] Text: {generated_text}");

    // ---- Assertions ----
    assert!(
        !generated_ids.is_empty(),
        "FAIL: No tokens generated"
    );

    let unique_tokens: HashSet<u32> = generated_ids.iter().copied().collect();
    assert!(
        unique_tokens.len() >= 2,
        "FAIL: Degenerate output - only {} unique tokens: {:?}",
        unique_tokens.len(), unique_tokens,
    );

    // Must not be dominated by token 101
    let count_101 = generated_ids.iter().filter(|&&id| id == 101).count();
    assert!(
        count_101 < generated_ids.len() / 2,
        "FAIL: Token 101 dominates: {count_101}/{}",
        generated_ids.len(),
    );

    // For "2+2" the answer should contain "four" or "4"
    let lower = generated_text.to_lowercase();
    let has_answer = lower.contains("four") || lower.contains("4");
    if has_answer {
        eprintln!("[test_e4b_coherence] PASS - correct answer detected");
    } else {
        eprintln!("[test_e4b_coherence] WARN - answer not detected in: {generated_text}");
    }

    eprintln!("[test_e4b_coherence] PASS - output is coherent");
    println!("{generated_text}");
    Ok(())
}
