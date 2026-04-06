//! Gemma4 chat runtime repro harness.
//!
//! Runs the same short prompt through three paths and dumps full diagnostics
//! so we can compare token IDs, stop behavior, and BOS handling.
//!
//! Usage:
//!   cargo run -p gemma4-mlx --example repro_chat [model_dir] [prompt]

use std::{env, error::Error, path::PathBuf};

use gemma4_mlx::{
    load_model, load_tokenizer, Gemma4ChatConfig, Gemma4ChatPipeline, Gemma4ChatTemplate,
    Gemma4Conversation, Gemma4Message, Generate, KVCache,
};
use mlx_rs::{
    ops::indexing::{IndexOp, NewAxis},
    Array,
};

const MAX_TOKENS: usize = 128;
const TEMPERATURE: f32 = 0.0;

/// EOS token IDs from generation_config.json
const EOS_TOKEN_IDS: &[u32] = &[1, 106, 50];

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args: Vec<String> = env::args().collect();
    let model_dir = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/gemma-4-26B-A4B-it"));
    let user_prompt = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "What is 2 + 3?".to_string());

    println!("=== Gemma4 Chat Runtime Repro ===");
    println!("model_dir: {}", model_dir.display());
    println!("user_prompt: {:?}", user_prompt);
    println!("max_tokens: {MAX_TOKENS}");
    println!("temperature: {TEMPERATURE}");
    println!("eos_token_ids: {EOS_TOKEN_IDS:?}");
    println!();

    // Load template to inspect special token strings
    let template = Gemma4ChatTemplate::load(&model_dir)?;
    println!("--- Special Tokens ---");
    println!("  bos: {:?}", template.tokens.bos);
    println!("  eos: {:?}", template.tokens.eos);
    println!("  turn_start: {:?}", template.tokens.turn_start);
    println!("  turn_end: {:?}", template.tokens.turn_end);
    println!("  channel_start: {:?}", template.tokens.channel_start);
    println!("  channel_end: {:?}", template.tokens.channel_end);
    println!();

    // Load tokenizer and resolve special token IDs
    let tokenizer = load_tokenizer(&model_dir)?;
    print_special_token_ids(&tokenizer, &template);

    println!("\n============================================================");
    println!("=== PATH 1: Raw (encode with add_special_tokens=true) ===");
    println!("============================================================\n");

    // Replicate what chat_gemma4.rs does: encode user text directly with add_special_tokens=true
    let raw_prompt_text = user_prompt.clone();

    println!("raw prompt text: {:?}", raw_prompt_text);
    let raw_encoding = tokenizer.encode(raw_prompt_text.as_str(), true)?;
    let raw_ids = raw_encoding.get_ids().to_vec();
    print_token_summary("raw", &raw_ids);

    {
        let mut model = load_model(&model_dir)?;
        let prompt_array = Array::from(raw_ids.as_slice()).index(NewAxis);
        let mut cache = Vec::<KVCache>::new();
        let generator = Generate::new(&mut model, &mut cache, TEMPERATURE, &prompt_array);

        run_generator("PATH 1 (raw)", generator, &tokenizer, &template)?;
    }

    println!("\n============================================================");
    println!("=== PATH 2: Template + encode(prompt, false) ===");
    println!("============================================================\n");

    // This is what generate_raw_text / the API path does
    let chat_messages = [
        Gemma4Message::user(&user_prompt),
    ];
    let rendered_prompt = template.render_prompt(&chat_messages, &[], true)?;

    println!("rendered prompt ({} chars):", rendered_prompt.len());
    println!("  first 200 chars: {:?}", &rendered_prompt[..rendered_prompt.len().min(200)]);
    println!("  last 100 chars:  {:?}", &rendered_prompt[rendered_prompt.len().saturating_sub(100)..]);
    println!();

    let chat_encoding = tokenizer.encode(rendered_prompt.as_str(), false)?;
    let chat_ids = chat_encoding.get_ids().to_vec();
    print_token_summary("chat(false)", &chat_ids);

    // Also encode with add_special_tokens=true for comparison
    let chat_encoding_true = tokenizer.encode(rendered_prompt.as_str(), true)?;
    let chat_ids_true = chat_encoding_true.get_ids().to_vec();
    print_token_summary("chat(true)", &chat_ids_true);

    // Check BOS presence
    println!();
    println!("  BOS token id=2 at position 0?");
    println!("    encode(prompt, false): {}", chat_ids.first() == Some(&2));
    println!("    encode(prompt, true):  {}", chat_ids_true.first() == Some(&2));
    if chat_ids != chat_ids_true {
        println!("  *** MISMATCH between encode(false) and encode(true) ***");
        println!("    encode(false) len: {}", chat_ids.len());
        println!("    encode(true)  len: {}", chat_ids_true.len());
        // Show first divergence
        for (i, (a, b)) in chat_ids.iter().zip(chat_ids_true.iter()).enumerate() {
            if a != b {
                println!("    first diff at index {i}: encode(false)={a}, encode(true)={b}");
                break;
            }
        }
    } else {
        println!("  encode(false) == encode(true): identical token IDs");
    }

    {
        let mut model = load_model(&model_dir)?;
        let prompt_array = Array::from(chat_ids.as_slice()).index(NewAxis);
        let mut cache = Vec::<KVCache>::new();
        let generator = Generate::new(&mut model, &mut cache, TEMPERATURE, &prompt_array);

        run_generator("PATH 2 (template+false)", generator, &tokenizer, &template)?;
    }

    println!("\n============================================================");
    println!("=== PATH 3: Gemma4ChatPipeline::chat() ===");
    println!("============================================================\n");

    {
        let mut pipeline = Gemma4ChatPipeline::load(
            &model_dir,
            Gemma4ChatConfig {
                temperature: TEMPERATURE,
                max_new_tokens: MAX_TOKENS,
                max_tool_iterations: 0,
            },
        )?;

        let mut conversation = Gemma4Conversation::new();
        conversation.add_user(&user_prompt);

        let response = pipeline.chat(&mut conversation)?;

        println!("  tokens_generated: {}", response.tokens_generated);
        let raw_trunc: String = response.raw_text.chars().take(500).collect();
        println!("  raw_text ({} chars): {:?}", response.raw_text.len(), raw_trunc);
        println!("  parsed text: {:?}", response.text);
        println!("  thinking: {:?}", response.thinking);
        println!("  tool_calls: {:?}", response.tool_calls);

        // Check for control token leaks in parsed text
        check_control_token_leaks("PATH 3 parsed text", &response.text, &template);
    }

    println!("\n=== Done ===");
    Ok(())
}

fn run_generator(
    label: &str,
    generator: Generate<'_, KVCache>,
    tokenizer: &tokenizers::Tokenizer,
    template: &Gemma4ChatTemplate,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut generated_ids: Vec<u32> = Vec::new();
    let mut stop_reason = "hit_cap";
    let mut stop_token_id: Option<u32> = None;

    for token in generator.take(MAX_TOKENS) {
        let token = token?;
        let token_id = token.item::<u32>();
        generated_ids.push(token_id);

        // Check token-level EOS
        if EOS_TOKEN_IDS.contains(&token_id) {
            stop_reason = "eos_token_id";
            stop_token_id = Some(token_id);
            break;
        }

        // Check string-level stop (mirrors chat.rs behavior)
        let decoded = tokenizer.decode(&generated_ids, false)?;
        if decoded.ends_with(&template.tokens.turn_end) {
            stop_reason = "string_turn_end";
            stop_token_id = Some(token_id);
            break;
        }
        if decoded.ends_with(&template.tokens.eos) {
            stop_reason = "string_eos";
            stop_token_id = Some(token_id);
            break;
        }
    }

    let full_decoded = tokenizer.decode(&generated_ids, false)?;

    println!("  [{label}] Generated {} tokens", generated_ids.len());
    println!("  [{label}] Stop reason: {stop_reason}");
    if let Some(sid) = stop_token_id {
        println!("  [{label}] Stop token id: {sid}");
    }
    println!("  [{label}] Token IDs (first 30): {:?}", &generated_ids[..generated_ids.len().min(30)]);
    if generated_ids.len() > 30 {
        println!("  [{label}] Token IDs (last 10): {:?}", &generated_ids[generated_ids.len().saturating_sub(10)..]);
    }
    println!("  [{label}] Decoded text ({} chars):", full_decoded.len());
    let truncated: String = full_decoded.chars().take(500).collect();
    println!("    {:?}", truncated);

    // Check for control token leaks
    check_control_token_leaks(label, &full_decoded, template);

    Ok(())
}

fn print_token_summary(label: &str, ids: &[u32]) {
    println!("  [{label}] {len} tokens", len = ids.len());
    println!("    first 20: {:?}", &ids[..ids.len().min(20)]);
    if ids.len() > 20 {
        println!("    last 10:  {:?}", &ids[ids.len().saturating_sub(10)..]);
    }
}

fn print_special_token_ids(tokenizer: &tokenizers::Tokenizer, template: &Gemma4ChatTemplate) {
    println!("--- Special Token ID Resolution ---");
    let special_strings = [
        ("bos", &template.tokens.bos),
        ("eos", &template.tokens.eos),
        ("turn_start", &template.tokens.turn_start),
        ("turn_end", &template.tokens.turn_end),
        ("channel_start", &template.tokens.channel_start),
        ("channel_end", &template.tokens.channel_end),
    ];

    for (name, token_str) in &special_strings {
        // Try encoding just this token both ways
        let enc_false = tokenizer.encode(token_str.as_str(), false).ok();
        let enc_true = tokenizer.encode(token_str.as_str(), true).ok();

        let ids_false = enc_false.as_ref().map(|e| e.get_ids().to_vec());
        let ids_true = enc_true.as_ref().map(|e| e.get_ids().to_vec());

        println!("  {name} = {token_str:?}");
        println!("    encode(false): {:?}", ids_false);
        println!("    encode(true):  {:?}", ids_true);

        // Also check the vocab directly
        if let Some(id) = tokenizer.token_to_id(token_str) {
            println!("    token_to_id:   {id}");
        } else {
            println!("    token_to_id:   NOT FOUND (not a single vocab entry)");
        }
    }
}

fn check_control_token_leaks(label: &str, text: &str, template: &Gemma4ChatTemplate) {
    let control_tokens = [
        ("channel_start", &template.tokens.channel_start),
        ("channel_end", &template.tokens.channel_end),
        ("turn_start", &template.tokens.turn_start),
        ("turn_end", &template.tokens.turn_end),
        ("tool_start", &template.tokens.tool_start),
        ("tool_end", &template.tokens.tool_end),
        ("tool_call_start", &template.tokens.tool_call_start),
        ("tool_call_end", &template.tokens.tool_call_end),
        ("bos", &template.tokens.bos),
        ("eos", &template.tokens.eos),
    ];

    let mut found_any = false;
    for (name, token_str) in &control_tokens {
        if text.contains(token_str.as_str()) {
            if !found_any {
                println!("  [{label}] CONTROL TOKEN LEAKS:");
                found_any = true;
            }
            println!("    - {name} ({token_str:?}) found in output");
        }
    }
    if !found_any {
        println!("  [{label}] No control token leaks detected");
    }
}
