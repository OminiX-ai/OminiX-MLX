//! Test multimodal embedding injection
//!
//! Run: cargo run --example test_multimodal --release

use funasr_qwen4b_mlx::sensevoice_encoder::{SenseVoiceEncoder, SenseVoiceEncoderConfig};
use funasr_qwen4b_mlx::adaptor::AudioAdaptorQwen4B;
use funasr_qwen4b_mlx::error::Result;
use mlx_rs::Array;
use mlx_rs::module::Module;
use mlx_rs::quantization::MaybeQuantized;
use mlx_rs::ops::indexing::{IndexOp, NewAxis};
use qwen3_mlx::{
    load_model, load_tokenizer, KVCache,
    AttentionInput, sample, create_attention_mask, AttentionMask,
};

fn main() -> Result<()> {
    println!("=== Multimodal Embedding Injection Test ===\n");

    // Paths
    let qwen_path = "models/Qwen3-4B";
    let sensevoice_path = "/Users/yuechen/.dora/models/funasr-nano/model.safetensors";
    let adaptor_path = "adaptor_phase2_final.safetensors";

    // Check Qwen3 model exists
    if !std::path::Path::new(qwen_path).exists() {
        println!("Qwen3-4B not found: {}", qwen_path);
        return Ok(());
    }

    // 1. Load models
    println!("1. Loading models...");

    let mut llm = load_model(qwen_path)
        .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?;
    let tokenizer = load_tokenizer(qwen_path)
        .map_err(|e| funasr_qwen4b_mlx::error::Error::Tokenizer(format!("{:?}", e)))?;
    println!("   Qwen3-4B loaded");

    let mut encoder = SenseVoiceEncoder::new(SenseVoiceEncoderConfig::default())?;
    if std::path::Path::new(sensevoice_path).exists() {
        encoder.load_weights(sensevoice_path)?;
        println!("   SenseVoice encoder loaded");
    }

    let mut adaptor = AudioAdaptorQwen4B::new()?;
    if std::path::Path::new(adaptor_path).exists() {
        adaptor.load_weights(adaptor_path)?;
        println!("   Adaptor loaded");
    }

    // 2. Simulate audio processing
    println!("\n2. Simulating audio pipeline...");
    let batch_size = 1i32;
    let seq_len = 50i32;  // ~0.5 second audio
    let lfr_dim = 560i32;

    // Simulate LFR output
    let lfr_input = Array::zeros::<f32>(&[batch_size, seq_len, lfr_dim])?;
    println!("   LFR input: {:?}", lfr_input.shape());

    // SenseVoice encoding
    let encoder_out = encoder.forward(&lfr_input)?;
    println!("   Encoder output: {:?}", encoder_out.shape());

    // Adaptor projection
    let audio_features = adaptor.forward(&encoder_out)?;
    println!("   Adapted features: {:?}", audio_features.shape());

    // 3. Create multimodal embeddings
    println!("\n3. Creating multimodal embeddings...");

    // Speech token IDs
    let start_of_speech: u32 = 151646;
    let end_of_speech: u32 = 151647;
    let im_end: u32 = 151645;
    let eos: u32 = 151643;

    // Prefix tokens: <|im_start|>user\n/no_think 语音转写成中文：<|startofspeech|>
    // Note: /no_think disables Qwen3's thinking mode for direct output
    let prefix = "语音转写成中文：";
    let prefix_encoding = tokenizer.encode(format!("<|im_start|>user\n/no_think {}", prefix).as_str(), false)
        .map_err(|e| funasr_qwen4b_mlx::error::Error::Tokenizer(format!("{}", e)))?;
    let mut prefix_tokens: Vec<u32> = prefix_encoding.get_ids().to_vec();
    prefix_tokens.push(start_of_speech);
    println!("   Prefix tokens: {} (including <|startofspeech|>)", prefix_tokens.len());

    // Suffix tokens: <|endofspeech|><|im_end|>\n<|im_start|>assistant\n
    let suffix = "<|im_end|>\n<|im_start|>assistant\n";
    let suffix_encoding = tokenizer.encode(suffix, false)
        .map_err(|e| funasr_qwen4b_mlx::error::Error::Tokenizer(format!("{}", e)))?;
    let mut suffix_tokens: Vec<u32> = vec![end_of_speech];
    suffix_tokens.extend(suffix_encoding.get_ids());
    println!("   Suffix tokens: {} (including <|endofspeech|>)", suffix_tokens.len());

    // Get text embeddings
    let prefix_ids = Array::from(prefix_tokens.as_slice()).index(NewAxis);
    let suffix_ids = Array::from(suffix_tokens.as_slice()).index(NewAxis);

    let prefix_embeds = match &mut llm.model.embed_tokens {
        MaybeQuantized::Original(embed) => embed.forward(&prefix_ids)
            .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?,
        MaybeQuantized::Quantized(embed) => embed.forward(&prefix_ids)
            .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?,
    };
    println!("   Prefix embeddings: {:?}", prefix_embeds.shape());

    let suffix_embeds = match &mut llm.model.embed_tokens {
        MaybeQuantized::Original(embed) => embed.forward(&suffix_ids)
            .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?,
        MaybeQuantized::Quantized(embed) => embed.forward(&suffix_ids)
            .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?,
    };
    println!("   Suffix embeddings: {:?}", suffix_embeds.shape());

    // Concatenate: [prefix_embeds, audio_features, suffix_embeds]
    let combined_embeds = mlx_rs::ops::concatenate_axis(
        &[&prefix_embeds, &audio_features, &suffix_embeds],
        1,  // axis=1 (sequence dimension)
    ).map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?;
    println!("   Combined embeddings: {:?}", combined_embeds.shape());

    // 4. Forward through transformer (prefill)
    println!("\n4. Running transformer prefill...");
    let mut cache: Vec<Option<KVCache>> = Vec::new();
    let mut h = combined_embeds;

    // Create attention mask
    let mask = match create_attention_mask(&h, &cache, Some(true))
        .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?
    {
        Some(AttentionMask::Array(m)) => Some(m),
        _ => None,
    };

    // Initialize cache
    cache = (0..llm.model.layers.len())
        .map(|_| Some(KVCache::default()))
        .collect();

    let start = std::time::Instant::now();

    // Forward through layers
    for (i, (layer, c)) in llm.model.layers.iter_mut().zip(cache.iter_mut()).enumerate() {
        let layer_input = AttentionInput {
            x: &h,
            mask: mask.as_ref(),
            cache: c.as_mut(),
        };
        h = layer.forward(layer_input)
            .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?;

        if i == 0 {
            println!("   Layer 0 output: {:?}", h.shape());
        }
    }

    // Final norm
    h = llm.model.norm.forward(&h)
        .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?;

    let prefill_time = start.elapsed();
    println!("   Final hidden state: {:?}", h.shape());
    println!("   Prefill time: {:.2?}", prefill_time);

    // 5. Get logits and sample first token
    println!("\n5. Sampling first token...");
    let last_hidden = h.index((.., -1, ..));

    let logits = match &mut llm.lm_head {
        Some(lm_head) => lm_head.forward(&last_hidden)
            .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?,
        None => {
            match &mut llm.model.embed_tokens {
                MaybeQuantized::Original(embed) => embed.as_linear(&last_hidden)
                    .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?,
                MaybeQuantized::Quantized(embed) => embed.as_linear(&last_hidden)
                    .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?,
            }
        }
    };
    println!("   Logits shape: {:?}", logits.shape());

    let y = sample(&logits, 0.0)
        .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?;
    let token_id: u32 = y.item();
    println!("   First token ID: {}", token_id);

    // Decode token
    let text = tokenizer.decode(&[token_id], true)
        .map_err(|e| funasr_qwen4b_mlx::error::Error::Tokenizer(format!("{}", e)))?;
    println!("   First token: '{}'", text);

    // 6. Generate more tokens
    println!("\n6. Generating tokens...");
    let max_tokens = 20;
    let mut generated_tokens: Vec<u32> = vec![];

    if token_id != eos && token_id != im_end {
        generated_tokens.push(token_id);
    }

    let mut current_token = y;
    let decode_start = std::time::Instant::now();

    for _ in 1..max_tokens {
        // Embed current token
        let y_ids = current_token.index(NewAxis);
        let y_embed = match &mut llm.model.embed_tokens {
            MaybeQuantized::Original(embed) => embed.forward(&y_ids)
                .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?,
            MaybeQuantized::Quantized(embed) => embed.forward(&y_ids)
                .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?,
        };

        // Forward through layers
        let mut h = y_embed;
        for (layer, c) in llm.model.layers.iter_mut().zip(cache.iter_mut()) {
            let layer_input = AttentionInput {
                x: &h,
                mask: None,  // Single token, no mask needed
                cache: c.as_mut(),
            };
            h = layer.forward(layer_input)
                .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?;
        }

        h = llm.model.norm.forward(&h)
            .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?;

        // Get logits and sample
        let logits = match &mut llm.lm_head {
            Some(lm_head) => lm_head.forward(&h.index((.., -1, ..)))
                .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?,
            None => {
                match &mut llm.model.embed_tokens {
                    MaybeQuantized::Original(embed) => embed.as_linear(&h.index((.., -1, ..)))
                        .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?,
                    MaybeQuantized::Quantized(embed) => embed.as_linear(&h.index((.., -1, ..)))
                        .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?,
                }
            }
        };

        current_token = sample(&logits, 0.0)
            .map_err(|e| funasr_qwen4b_mlx::error::Error::ModelLoad(format!("{:?}", e)))?;
        let token_id: u32 = current_token.item();

        if token_id == eos || token_id == im_end {
            break;
        }
        generated_tokens.push(token_id);
    }

    let decode_time = decode_start.elapsed();

    // Decode all tokens
    let output_text = tokenizer.decode(&generated_tokens, true)
        .map_err(|e| funasr_qwen4b_mlx::error::Error::Tokenizer(format!("{}", e)))?;

    println!("\n=== Results ===");
    println!("Generated {} tokens in {:.2?}", generated_tokens.len(), decode_time);
    if !generated_tokens.is_empty() {
        println!("Speed: {:.1} tok/s", generated_tokens.len() as f64 / decode_time.as_secs_f64());
    }
    println!("Output: {}", output_text);

    println!("\n=== Multimodal Embedding Injection Test Complete ===");
    println!("\nNote: With dummy audio (zeros), output may be garbage.");
    println!("Real transcription requires actual audio features from SenseVoice.");

    Ok(())
}
