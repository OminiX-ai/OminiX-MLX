//! Debug the token generation.

use funasr_nano_mlx::audio;
use funasr_nano_mlx::model::FunASRNano;
use mlx_rs::module::Module;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::transforms::eval;
use mlx_rs::Array;

fn main() {
    let model_dir = "./Fun-ASR-Nano-2512";
    let audio_path = "./Fun-ASR-Nano-2512/example/zh.wav";

    println!("Loading model from {}...", model_dir);
    let mut model = FunASRNano::load(model_dir).expect("Failed to load model");

    // Load and preprocess audio
    let (samples, sample_rate) = audio::load_wav(audio_path).expect("Failed to load audio");
    let samples = audio::resample(&samples, sample_rate, 16000).expect("Failed to resample");

    // Compute mel spectrogram with LFR
    let audio_config = audio::AudioConfig::default();
    let mel = audio::compute_mel_spectrogram(&samples, &audio_config).expect("Failed to compute mel");
    let mel_lfr = audio::apply_lfr(&mel, 7, 6).expect("Failed to apply LFR");

    // Encode and adapt
    let encoder_out = model.encoder.forward(&mel_lfr).expect("Failed to encode");
    let audio_features = model.adaptor.forward(&encoder_out).expect("Failed to adapt");
    eval([&audio_features]).expect("eval failed");

    let audio_len = audio_features.shape()[1];
    println!("Audio features: shape={:?}", audio_features.shape());

    // Build prompt tokens
    let prefix_tokens: Vec<i32> = vec![
        151644,  // <|im_start|>
        872,     // user
        198,     // \n
        14880,   // 请
        46670,   // 转
        23656,   // 录
        87752,   // 以下
        111268,  // 音频
        25,      // :
    ];
    let suffix_tokens: Vec<i32> = vec![
        151645,  // <|im_end|>
        198,     // \n
        151644,  // <|im_start|>
        77091,   // assistant
        198,     // \n
    ];

    let speech_start = 151646;
    let speech_end = 151647;

    // Build prompt with audio placeholder
    let mut prompt_tokens: Vec<i32> = Vec::new();
    prompt_tokens.extend_from_slice(&prefix_tokens);
    prompt_tokens.push(speech_start);
    for _ in 0..audio_len {
        prompt_tokens.push(0);
    }
    prompt_tokens.push(speech_end);
    prompt_tokens.extend_from_slice(&suffix_tokens);

    println!("Prompt length: {}", prompt_tokens.len());

    // Get embeddings
    let prompt_array = Array::from_slice(&prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let embeddings = model.llm.get_token_embeddings(&prompt_array).expect("Failed to get embeddings");
    eval([&embeddings]).expect("eval failed");

    println!("Text embeddings: shape={:?}", embeddings.shape());
    println!("Text embeddings range: [{:.4}, {:.4}]",
        embeddings.min(None).expect("min").item::<f32>(),
        embeddings.max(None).expect("max").item::<f32>());

    // Insert audio
    let audio_start = prefix_tokens.len() + 1;
    let audio_end = audio_start + audio_len as usize;

    let prefix_embed = embeddings.index((.., ..audio_start as i32, ..));
    let suffix_embed = embeddings.index((.., audio_end as i32.., ..));
    let h = mlx_rs::ops::concatenate_axis(&[&prefix_embed, &audio_features, &suffix_embed], 1)
        .expect("Failed to concatenate");
    eval([&h]).expect("eval failed");

    println!("Combined embeddings: shape={:?}", h.shape());
    println!("Combined range: [{:.4}, {:.4}]",
        h.min(None).expect("min").item::<f32>(),
        h.max(None).expect("max").item::<f32>());

    // First forward pass
    let mut cache = Vec::new();
    let logits = model.llm.forward_embeddings(&h, &mut cache).expect("Failed to forward");
    eval([&logits]).expect("eval failed");

    println!("\nLogits: shape={:?}", logits.shape());
    println!("Logits range: [{:.4}, {:.4}]",
        logits.min(None).expect("min").item::<f32>(),
        logits.max(None).expect("max").item::<f32>());

    // Get top-5 tokens
    let last_logits = logits.index((.., -1, ..));
    let probs = mlx_rs::ops::softmax_axis(&last_logits, -1, None).expect("softmax failed");
    eval([&probs]).expect("eval failed");

    // Find top tokens
    println!("\nTop predicted tokens:");
    for i in 0..5 {
        let argmax = mlx_rs::ops::indexing::argmax_axis(&last_logits, -1, false).expect("argmax failed");
        eval([&argmax]).expect("eval failed");
        let token_id = argmax.item::<i32>();
        let prob = probs.index((.., .., token_id)).item::<f32>();
        println!("  {}: token_id={}, prob={:.4}", i, token_id, prob);
        // Note: This just shows the same top token repeatedly since we don't modify logits
        break;
    }
}
