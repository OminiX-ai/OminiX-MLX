//! Talker model (28-layer Qwen3-style transformer) and Code Predictor (5-layer sub-talker).

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::{
    builder::Builder,
    module::{Module, ModuleParameters, Param},
    nn,
    ops::indexing::IndexOp,
    ops::zeros,
    quantization::MaybeQuantized,
    transforms::eval,
    Array,
};

use crate::config::{CodePredictorConfig, QuantizationConfig, TalkerConfig};
use crate::error::{Error, Result};
use crate::sampling::sample_logits;
use mlx_rs_core::cache::{KVCache, KeyValueCache};
use mlx_rs_core::utils::{scaled_dot_product_attention, SdpaMask};

// ============================================================================
// Attention with MRoPE
// ============================================================================

pub struct TalkerAttention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub rope: nn::Rope,

    pub q_proj: MaybeQuantized<nn::Linear>,
    pub k_proj: MaybeQuantized<nn::Linear>,
    pub v_proj: MaybeQuantized<nn::Linear>,
    pub o_proj: MaybeQuantized<nn::Linear>,
    pub q_norm: nn::RmsNorm,
    pub k_norm: nn::RmsNorm,
}

impl TalkerAttention {
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut KVCache,
    ) -> Result<Array> {
        let shape = x.shape();
        let b = shape[0];
        let l = shape[1];

        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        // Reshape to [B, L, heads, head_dim] then transpose to [B, heads, L, head_dim]
        let mut queries = self
            .q_norm
            .forward(
                &queries
                    .reshape(&[b, l, self.n_heads, -1])?
                    .transpose_axes(&[0, 2, 1, 3])?,
            )?;
        let mut keys = self
            .k_norm
            .forward(
                &keys
                    .reshape(&[b, l, self.n_kv_heads, -1])?
                    .transpose_axes(&[0, 2, 1, 3])?,
            )?;
        let values = values
            .reshape(&[b, l, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Apply RoPE (all 3 MRoPE dims use same position, equivalent to standard RoPE)
        let offset = cache.offset();
        let q_input = nn::RopeInputBuilder::new(&queries)
            .offset(offset)
            .build()
            .unwrap();
        queries = self.rope.forward(q_input)?;
        let k_input = nn::RopeInputBuilder::new(&keys)
            .offset(offset)
            .build()
            .unwrap();
        keys = self.rope.forward(k_input)?;

        // Update KV cache
        let (keys, values) = cache.update_and_fetch(keys, values)?;

        // Attention mask
        let sdpa_mask = match mask {
            Some(m) => Some(SdpaMask::Array(m)),
            None if l > 1 => Some(SdpaMask::Causal),
            None => None,
        };

        let output = scaled_dot_product_attention::<KVCache>(
            queries, keys, values, None, self.scale, sdpa_mask,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[b, l, -1])?;

        Ok(self.o_proj.forward(&output)?)
    }
}

// ============================================================================
// MLP (SwiGLU)
// ============================================================================

pub struct TalkerMlp {
    pub gate_proj: MaybeQuantized<nn::Linear>,
    pub up_proj: MaybeQuantized<nn::Linear>,
    pub down_proj: MaybeQuantized<nn::Linear>,
}

impl TalkerMlp {
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let gate = nn::silu(self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        Ok(self.down_proj.forward(&gate.multiply(up)?)?)
    }
}

// ============================================================================
// Transformer Block
// ============================================================================

pub struct TalkerBlock {
    pub self_attn: TalkerAttention,
    pub mlp: TalkerMlp,
    pub input_layernorm: nn::RmsNorm,
    pub post_attention_layernorm: nn::RmsNorm,
}

impl TalkerBlock {
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut KVCache,
    ) -> Result<Array> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&normed, mask, cache)?;
        let h = x.add(attn_out)?;
        let normed = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        Ok(h.add(mlp_out)?)
    }
}

// ============================================================================
// Text Projection (2-layer MLP)
// ============================================================================

pub struct TextProjection {
    pub fc1: MaybeQuantized<nn::Linear>,
    pub fc1_bias: Option<Array>,
    pub fc2: MaybeQuantized<nn::Linear>,
    pub fc2_bias: Option<Array>,
}

impl TextProjection {
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let mut h = self.fc1.forward(x)?;
        if let Some(bias) = &self.fc1_bias {
            h = h.add(bias)?;
        }
        h = nn::silu(h)?;
        h = self.fc2.forward(&h)?;
        if let Some(bias) = &self.fc2_bias {
            h = h.add(bias)?;
        }
        Ok(h)
    }
}

// ============================================================================
// Code Predictor
// ============================================================================

pub struct CodePredictorAttention {
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,

    pub q_proj: MaybeQuantized<nn::Linear>,
    pub k_proj: MaybeQuantized<nn::Linear>,
    pub v_proj: MaybeQuantized<nn::Linear>,
    pub o_proj: MaybeQuantized<nn::Linear>,
    pub q_norm: nn::RmsNorm,
    pub k_norm: nn::RmsNorm,
    pub rope: nn::Rope,
}

impl CodePredictorAttention {
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut KVCache,
    ) -> Result<Array> {
        let shape = x.shape();
        let b = shape[0];
        let l = shape[1];

        let queries = self.q_proj.forward(x)?;
        let keys = self.k_proj.forward(x)?;
        let values = self.v_proj.forward(x)?;

        let mut queries = self
            .q_norm
            .forward(
                &queries
                    .reshape(&[b, l, self.n_heads, -1])?
                    .transpose_axes(&[0, 2, 1, 3])?,
            )?;
        let mut keys = self
            .k_norm
            .forward(
                &keys
                    .reshape(&[b, l, self.n_kv_heads, -1])?
                    .transpose_axes(&[0, 2, 1, 3])?,
            )?;
        let values = values
            .reshape(&[b, l, self.n_kv_heads, -1])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // Standard RoPE for code predictor
        let offset = cache.offset();
        let q_input = nn::RopeInputBuilder::new(&queries)
            .offset(offset)
            .build()
            .unwrap(); // safe: Infallible error type
        queries = self.rope.forward(q_input)?;
        let k_input = nn::RopeInputBuilder::new(&keys)
            .offset(offset)
            .build()
            .unwrap(); // safe: Infallible error type
        keys = self.rope.forward(k_input)?;

        let (keys, values) = cache.update_and_fetch(keys, values)?;

        let sdpa_mask = match mask {
            Some(m) => Some(SdpaMask::Array(m)),
            None if l > 1 => Some(SdpaMask::Causal),
            None => None,
        };

        let output = scaled_dot_product_attention::<KVCache>(
            queries, keys, values, None, self.scale, sdpa_mask,
        )?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[b, l, -1])?;

        Ok(self.o_proj.forward(&output)?)
    }
}

pub struct CodePredictorBlock {
    pub self_attn: CodePredictorAttention,
    pub mlp: TalkerMlp,
    pub input_layernorm: nn::RmsNorm,
    pub post_attention_layernorm: nn::RmsNorm,
}

impl CodePredictorBlock {
    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut KVCache,
    ) -> Result<Array> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = self.self_attn.forward(&normed, mask, cache)?;
        let h = x.add(attn_out)?;
        let normed = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed)?;
        Ok(h.add(mlp_out)?)
    }
}

pub struct CodePredictor {
    pub layers: Vec<CodePredictorBlock>,
    pub norm: nn::RmsNorm,
    pub codec_embeddings: Vec<nn::Embedding>, // 15 embeddings for codebooks 1-15
    pub lm_heads: Vec<MaybeQuantized<nn::Linear>>, // 15 heads
    pub small_to_mtp_projection: MaybeQuantized<nn::Linear>,
    pub mtp_proj_bias: Option<Array>,
}

impl CodePredictor {
    /// Generate codebooks 1-15 autoregressively for one time step.
    /// Takes the talker's last hidden state and the codebook-0 embedding (both in talker dim).
    /// Always uses greedy decoding (argmax) for deterministic sub-code generation.
    pub fn generate_codes(
        &mut self,
        talker_hidden: &Array, // [B, 1, talker_hidden_size=2048]
        code0_embed: &Array,   // [B, 1, talker_hidden_size=2048]
    ) -> Result<Vec<u32>> {
        let mut codes = Vec::with_capacity(15);

        // Project both to code predictor dimension (2048 → 1024)
        let mut proj_hidden = self.small_to_mtp_projection.forward(talker_hidden)?;
        if let Some(bias) = &self.mtp_proj_bias {
            proj_hidden = proj_hidden.add(bias)?;
        }
        let mut proj_code0 = self.small_to_mtp_projection.forward(code0_embed)?;
        if let Some(bias) = &self.mtp_proj_bias {
            proj_code0 = proj_code0.add(bias)?;
        }

        // Concatenate to form 2-token prefill: [past_hidden, code0_embed]
        let prefill_input =
            mlx_rs::ops::concatenate_axis(&[&proj_hidden, &proj_code0], 1)?;
        eval(std::iter::once(&prefill_input))?;

        // Initialize KV caches for code predictor (fresh per time step)
        let mut caches: Vec<KVCache> = (0..self.layers.len())
            .map(|_| KVCache::new())
            .collect();

        // Prefill with 2 tokens
        let mut current_output = prefill_input;
        for (layer, cache) in self.layers.iter_mut().zip(caches.iter_mut()) {
            current_output = layer.forward(&current_output, None, cache)?;
        }
        eval(std::iter::once(&current_output))?;

        // Sample codebook 1 from last position (code0 position) logits
        let normed = self.norm.forward(&current_output)?;
        use mlx_rs::ops::indexing::IndexOp;
        let last_normed = normed.index((.., -1.., ..)); // [B, 1, 1024]
        let logits = self.lm_heads[0].forward(&last_normed)?;
        eval([&logits])?;
        // Greedy decoding (argmax) for code predictor
        let token = sample_logits(&logits, 0.0, 0, 1.0, 1.0, &[], None)?;
        codes.push(token);

        // Autoregressive for codebooks 2-15
        for g in 1..15 {
            // Embed previous token through codec_embeddings[g-1] → project to 1024
            let token_arr = Array::from_slice(&[codes[g - 1] as i32], &[1, 1]);
            let embed = self.codec_embeddings[g - 1].forward(&token_arr)?;
            let mut current_input = self.small_to_mtp_projection.forward(&embed)?;
            if let Some(bias) = &self.mtp_proj_bias {
                current_input = current_input.add(bias)?;
            }

            for (layer, cache) in self.layers.iter_mut().zip(caches.iter_mut()) {
                current_input = layer.forward(&current_input, None, cache)?;
            }
            eval(std::iter::once(&current_input))?;

            let normed = self.norm.forward(&current_input)?;
            let logits = self.lm_heads[g].forward(&normed)?;
            eval([&logits])?;
            // Greedy decoding (argmax) for code predictor
            let token = sample_logits(&logits, 0.0, 0, 1.0, 1.0, &[], None)?;
            codes.push(token);
        }

        Ok(codes)
    }
}

// ============================================================================
// Full Talker
// ============================================================================

pub struct Talker {
    pub config: TalkerConfig,
    pub tts_pad_token_id: u32,

    // Embeddings
    pub text_embedding: nn::Embedding,
    pub codec_embedding: nn::Embedding, // shared across all 16 groups

    // Projection
    pub text_projection: TextProjection,

    // Transformer
    pub layers: Vec<TalkerBlock>,
    pub norm: nn::RmsNorm,

    // Output head
    pub codec_head: MaybeQuantized<nn::Linear>,

    // Code predictor
    pub code_predictor: CodePredictor,

    // KV caches for the main talker
    pub caches: Vec<KVCache>,
}

impl Talker {
    /// Reset KV caches (call before each generation)
    pub fn reset_caches(&mut self) {
        self.caches = (0..self.layers.len())
            .map(|_| KVCache::new())
            .collect();
    }

    /// Forward one step of the talker.
    /// Returns (codebook_0_logits, last_hidden_state)
    pub fn forward_step(
        &mut self,
        input_embeds: &Array, // [B, L, hidden_size]
    ) -> Result<(Array, Array)> {
        let l = input_embeds.dim(1);
        let mut h = input_embeds.clone();

        // Create causal mask if needed
        let mask = if l > 1 {
            Some(mlx_rs_core::utils::create_causal_mask(
                l as i32,
                Some(self.caches[0].offset()),
                None,
                None,
            )?)
        } else {
            None
        };

        for (layer, cache) in self.layers.iter_mut().zip(self.caches.iter_mut()) {
            h = layer.forward(&h, mask.as_ref(), cache)?;
        }

        let normed = self.norm.forward(&h)?;
        let logits = self.codec_head.forward(&normed)?;

        // Return normed hidden (not pre-norm h) — code predictor needs post-norm hidden
        Ok((logits, normed))
    }

    /// Build text-only embedding (NO codec component).
    /// Used for role tokens (im_start, assistant, \n) at the start of generation.
    pub fn build_text_only_embedding(&mut self, text_token: u32) -> Result<Array> {
        let tok_arr = Array::from_slice(&[text_token as i32], &[1, 1]);
        let embed = self.text_embedding.forward(&tok_arr)?;
        let projected = self.text_projection.forward(&embed)?;
        // Ensure float32 for downstream sampling
        Ok(projected.as_dtype(mlx_rs::Dtype::Float32)?)
    }

    /// Build combined embedding for autoregressive generation.
    /// Uses talker's codec_embedding for codebook-0,
    /// code predictor's codec_embeddings for codebooks 1-15.
    pub fn build_generation_embedding(
        &mut self,
        text_token: Option<u32>,
        prev_codes: &[u32; 16],
    ) -> Result<Array> {
        let text_tok_id = text_token.unwrap_or(self.tts_pad_token_id);
        let tok_arr = Array::from_slice(&[text_tok_id as i32], &[1, 1]);
        let text_embed = {
            let embed = self.text_embedding.forward(&tok_arr)?;
            self.text_projection.forward(&embed)?
        };

        // Codebook 0: talker's codec_embedding
        let mut codec_embed = zeros::<f32>(&[1, 1, self.config.hidden_size])?;
        let code0_arr = Array::from_slice(&[prev_codes[0] as i32], &[1, 1]);
        codec_embed = codec_embed.add(self.codec_embedding.forward(&code0_arr)?)?;

        // Codebooks 1-15: code predictor's codec_embeddings
        for g in 0..15 {
            let code_arr = Array::from_slice(&[prev_codes[g + 1] as i32], &[1, 1]);
            let embed = self.code_predictor.codec_embeddings[g].forward(&code_arr)?;
            codec_embed = codec_embed.add(embed)?;
        }

        Ok(text_embed.add(codec_embed)?)
    }

    /// Build embedding for prefill positions.
    /// Uses a SINGLE codec token (not a sum of 16).
    /// text_token: text token ID (or None → uses tts_pad)
    /// codec_token: single codec control token
    pub fn build_prefill_embedding(
        &mut self,
        text_token: Option<u32>,
        codec_token: u32,
    ) -> Result<Array> {
        let text_tok_id = text_token.unwrap_or(self.tts_pad_token_id);
        let tok_arr = Array::from_slice(&[text_tok_id as i32], &[1, 1]);
        let text_embed = {
            let embed = self.text_embedding.forward(&tok_arr)?;
            self.text_projection.forward(&embed)?
        };

        let code_arr = Array::from_slice(&[codec_token as i32], &[1, 1]);
        let codec_embed = self.codec_embedding.forward(&code_arr)?;

        // Ensure float32 for consistent dtype through transformer
        Ok(text_embed.add(codec_embed)?.as_dtype(mlx_rs::Dtype::Float32)?)
    }

    /// Build batched prefill embedding for ALL positions at once.
    /// text_tokens: text token IDs for each position (use tts_pad_token_id for padding)
    /// codec_tokens: codec token IDs for each position
    /// no_codec_mask: positions where codec embedding should NOT be added (role tokens)
    pub fn build_batched_prefill_embedding(
        &mut self,
        text_tokens: &[u32],
        codec_tokens: &[u32],
        no_codec_positions: usize, // number of leading positions with no codec
    ) -> Result<Array> {
        assert_eq!(text_tokens.len(), codec_tokens.len());
        let seq_len = text_tokens.len();

        // Build text embeddings for all positions at once
        let text_ids: Vec<i32> = text_tokens.iter().map(|&t| t as i32).collect();
        let text_arr = Array::from_slice(&text_ids, &[1, seq_len as i32]);
        let text_embed = self.text_embedding.forward(&text_arr)?;
        let text_proj = self.text_projection.forward(&text_embed)?;

        // Build codec embeddings for all positions at once
        let codec_ids: Vec<i32> = codec_tokens.iter().map(|&c| c as i32).collect();
        let codec_arr = Array::from_slice(&codec_ids, &[1, seq_len as i32]);
        let codec_embed = self.codec_embedding.forward(&codec_arr)?;

        // For role positions (no codec), zero out the codec embedding
        if no_codec_positions > 0 {
            let codec_embed = codec_embed.as_dtype(mlx_rs::Dtype::Float32)?;
            eval([&text_proj, &codec_embed])?;
            let mut codec_vec: Vec<f32> = codec_embed.as_slice::<f32>().to_vec();
            let hidden = self.config.hidden_size as usize;
            // Zero out first no_codec_positions positions
            for i in 0..no_codec_positions * hidden {
                codec_vec[i] = 0.0;
            }
            let codec_zeroed = Array::from_slice(
                &codec_vec,
                &[1, seq_len as i32, hidden as i32],
            );
            Ok(text_proj.add(codec_zeroed)?.as_dtype(mlx_rs::Dtype::Float32)?)
        } else {
            Ok(text_proj.add(codec_embed)?.as_dtype(mlx_rs::Dtype::Float32)?)
        }
    }

    /// Build projected text embeddings for a batch of token IDs.
    /// Returns [1, seq_len, hidden_size] tensor.
    pub fn build_projected_text_embeddings(&mut self, token_ids: &[u32]) -> Result<Array> {
        let ids: Vec<i32> = token_ids.iter().map(|&t| t as i32).collect();
        let arr = Array::from_slice(&ids, &[1, ids.len() as i32]);
        let embed = self.text_embedding.forward(&arr)?;
        let projected = self.text_projection.forward(&embed)?;
        Ok(projected.as_dtype(mlx_rs::Dtype::Float32)?)
    }

    /// Build batched prefill embedding for VoiceDesign mode.
    /// Layout: [instruct_embed (N)] + [role_prefix (3)] + [codec_overlay (5)] + [first_text+bos (1)]
    /// instruct_tokens: tokenized instruct text (ChatML-wrapped)
    /// text_tokens: the text to synthesize (first token used in last position)
    /// codec_prefix: [think, think_bos, lang, think_eos] (4 tokens, no speaker)
    pub fn build_voice_design_prefill_embedding(
        &mut self,
        instruct_tokens: &[u32],
        text_tokens: &[u32],
        codec_prefix: &[u32],
        tts_config: &crate::config::Qwen3TtsConfig,
    ) -> Result<Array> {
        let hidden_size = self.config.hidden_size;
        let pad_id = tts_config.talker_config.codec_pad_id;
        let bos_id = tts_config.talker_config.codec_bos_id;

        // 1. Instruct embedding [1, N, hidden] — text projection only, no codec
        let instruct_embed = if instruct_tokens.is_empty() {
            zeros::<f32>(&[1, 0, hidden_size])?
        } else {
            self.build_projected_text_embeddings(instruct_tokens)?
        };

        // 2. Role prefix [1, 3, hidden] — text projection only, no codec
        let role_tokens = [
            tts_config.im_start_token_id,
            tts_config.assistant_token_id,
            198u32, // \n
        ];
        let role_embed = self.build_projected_text_embeddings(&role_tokens)?;

        // 3. Codec prefix overlay [1, 5, hidden]
        //    text: [tts_pad×4, tts_bos] projected
        //    codec: [think, think_bos, lang, think_eos, pad] embedded
        //    combined via addition
        let mut text_overlay_ids = vec![tts_config.tts_pad_token_id; 4];
        text_overlay_ids.push(tts_config.tts_bos_token_id);
        let text_overlay = self.build_projected_text_embeddings(&text_overlay_ids)?;

        let mut codec_ids: Vec<u32> = codec_prefix.to_vec(); // [think, think_bos, lang, think_eos]
        codec_ids.push(pad_id); // codec_pad at position 4
        let codec_ids_i32: Vec<i32> = codec_ids.iter().map(|&c| c as i32).collect();
        let codec_arr = Array::from_slice(&codec_ids_i32, &[1, 5]);
        let codec_embed = self.codec_embedding.forward(&codec_arr)?;
        let codec_overlay = text_overlay.add(codec_embed)?;

        // 4. First text + codec_bos [1, 1, hidden]
        let first_text = if text_tokens.is_empty() {
            tts_config.tts_pad_token_id
        } else {
            text_tokens[0]
        };
        let first_text_embed = self.build_projected_text_embeddings(&[first_text])?;
        let bos_arr = Array::from_slice(&[bos_id as i32], &[1, 1]);
        let bos_embed = self.codec_embedding.forward(&bos_arr)?;
        let first_combined = first_text_embed.add(bos_embed)?;

        // Concatenate all parts along sequence axis: [1, N+3+5+1, hidden]
        let all = mlx_rs::ops::concatenate_axis(
            &[&instruct_embed, &role_embed, &codec_overlay, &first_combined],
            1,
        )?;
        Ok(all.as_dtype(mlx_rs::Dtype::Float32)?)
    }

    /// Build batched prefill embedding for voice cloning (x_vector_only mode).
    /// Same as CustomVoice prefill but position 7 uses continuous speaker embedding
    /// instead of a discrete speaker token.
    ///
    /// Layout (10 positions):
    ///   Pos 0-2: role [im_start, assistant, \n] — text projection only
    ///   Pos 3-6: tts_pad + codec [think, think_bos, lang, think_eos]
    ///   Pos 7:   tts_pad + speaker_embedding (continuous, from ECAPA-TDNN)
    ///   Pos 8:   tts_bos + codec_pad
    ///   Pos 9:   first_text + codec_bos
    pub fn build_voice_clone_prefill_embedding(
        &mut self,
        text_tokens: &[u32],
        codec_prefix: &[u32],  // e.g., [nothink, think_bos, think_eos] (3) or [think, think_bos, lang, think_eos] (4)
        speaker_embedding: &Array, // [1, enc_dim] from ECAPA-TDNN
        tts_config: &crate::config::Qwen3TtsConfig,
    ) -> Result<Array> {
        let pad_id = tts_config.talker_config.codec_pad_id;
        let bos_id = tts_config.talker_config.codec_bos_id;
        let n_prefix = codec_prefix.len();

        // Pos 0-2: role tokens — text projection only [1, 3, hidden]
        let role_tokens = [
            tts_config.im_start_token_id,
            tts_config.assistant_token_id,
            198u32, // \n
        ];
        let role_embed = self.build_projected_text_embeddings(&role_tokens)?;

        // Pos 3..3+N: tts_pad + codec prefix (N tokens)
        let text_pad_n = vec![tts_config.tts_pad_token_id; n_prefix];
        let text_pad_n_embed = self.build_projected_text_embeddings(&text_pad_n)?;
        let codec_ids_i32: Vec<i32> = codec_prefix.iter().map(|&c| c as i32).collect();
        let codec_arr = Array::from_slice(&codec_ids_i32, &[1, n_prefix as i32]);
        let codec_embed = self.codec_embedding.forward(&codec_arr)?;
        let codec_overlay = text_pad_n_embed.add(codec_embed)?;

        // Pos 3+N: tts_pad + speaker_embedding (continuous)
        let spk_embed = speaker_embedding.reshape(&[1, 1, -1])?;
        let tts_pad_one = self.build_projected_text_embeddings(&[tts_config.tts_pad_token_id])?;
        let spk_pos = tts_pad_one.add(&spk_embed)?;

        // Pos 3+N+1: tts_bos + codec_pad [1, 1, hidden]
        let tts_bos_embed = self.build_projected_text_embeddings(&[tts_config.tts_bos_token_id])?;
        let pad_arr = Array::from_slice(&[pad_id as i32], &[1, 1]);
        let pad_embed = self.codec_embedding.forward(&pad_arr)?;
        let bos_pos = tts_bos_embed.add(pad_embed)?;

        // Pos 3+N+2: first_text + codec_bos [1, 1, hidden]
        let first_text = if text_tokens.is_empty() {
            tts_config.tts_pad_token_id
        } else {
            text_tokens[0]
        };
        let first_text_embed = self.build_projected_text_embeddings(&[first_text])?;
        let bos_arr = Array::from_slice(&[bos_id as i32], &[1, 1]);
        let bos_embed = self.codec_embedding.forward(&bos_arr)?;
        let first_pos = first_text_embed.add(bos_embed)?;

        // Concatenate: [1, 3+N+3, hidden]
        let all = mlx_rs::ops::concatenate_axis(
            &[&role_embed, &codec_overlay, &spk_pos, &bos_pos, &first_pos],
            1,
        )?;
        Ok(all.as_dtype(mlx_rs::Dtype::Float32)?)
    }

    /// Build ICL prefill embedding for voice cloning (9 positions, no first_text+bos).
    /// ICL mode excludes position 9 because all text goes into the ICL prompt instead.
    ///
    /// Layout (variable positions, matching Python's streaming prefill):
    ///   Pos 0-2:            role [im_start, assistant, \n] — text projection only
    ///   Pos 3..3+N:         tts_pad + codec_prefix (N = codec_prefix.len())
    ///   Pos 3+N:            tts_pad + speaker_embedding (continuous, from ECAPA-TDNN)
    ///   Pos 3+N+1:          tts_bos + codec_pad
    ///
    /// For nothink/auto-language (N=3): 8 total positions
    /// For explicit language (N=4): 9 total positions
    pub fn build_icl_prefill_embedding(
        &mut self,
        codec_prefix: &[u32],  // [nothink, think_bos, think_eos] (3) or [think, think_bos, lang, think_eos] (4)
        speaker_embedding: &Array, // [1, enc_dim] from ECAPA-TDNN
        tts_config: &crate::config::Qwen3TtsConfig,
    ) -> Result<Array> {
        let pad_id = tts_config.talker_config.codec_pad_id;
        let n_prefix = codec_prefix.len();

        // Pos 0-2: role tokens [1, 3, hidden]
        let role_tokens = [
            tts_config.im_start_token_id,
            tts_config.assistant_token_id,
            198u32,
        ];
        let role_embed = self.build_projected_text_embeddings(&role_tokens)?;

        // Pos 3..3+N: tts_pad + codec prefix [1, N, hidden]
        let text_pad_n = vec![tts_config.tts_pad_token_id; n_prefix];
        let text_pad_embed = self.build_projected_text_embeddings(&text_pad_n)?;
        let codec_ids_i32: Vec<i32> = codec_prefix.iter().map(|&c| c as i32).collect();
        let codec_arr = Array::from_slice(&codec_ids_i32, &[1, n_prefix as i32]);
        let codec_embed = self.codec_embedding.forward(&codec_arr)?;
        let codec_overlay = text_pad_embed.add(codec_embed)?;

        // Pos 3+N: tts_pad + speaker_embedding [1, 1, hidden]
        let spk_embed = speaker_embedding.reshape(&[1, 1, -1])?;
        let tts_pad_one = self.build_projected_text_embeddings(&[tts_config.tts_pad_token_id])?;
        let spk_pos = tts_pad_one.add(&spk_embed)?;

        // Pos 3+N+1: tts_bos + codec_pad [1, 1, hidden]
        let tts_bos_embed = self.build_projected_text_embeddings(&[tts_config.tts_bos_token_id])?;
        let pad_arr = Array::from_slice(&[pad_id as i32], &[1, 1]);
        let pad_embed = self.codec_embedding.forward(&pad_arr)?;
        let bos_pos = tts_bos_embed.add(pad_embed)?;

        // Concatenate: [1, 3+N+2, hidden]
        let all = mlx_rs::ops::concatenate_axis(
            &[&role_embed, &codec_overlay, &spk_pos, &bos_pos],
            1,
        )?;
        Ok(all.as_dtype(mlx_rs::Dtype::Float32)?)
    }

    /// Sum reference codec embeddings across all 16 codebook groups.
    /// For each frame: talker.codec_embedding(code[0]) + code_predictor.codec_embeddings[i-1](code[i]) for i=1..15.
    /// Input: reference frames as Vec<[u32; 16]>
    /// Output: [1, T_ref, hidden_size]
    pub fn sum_ref_codec_embeddings(&mut self, ref_codes: &[[u32; 16]]) -> Result<Array> {
        let t = ref_codes.len();

        // Build codes arrays for each codebook group
        // Group 0: talker.codec_embedding
        let codes_0: Vec<i32> = ref_codes.iter().map(|f| f[0] as i32).collect();
        let codes_0_arr = Array::from_slice(&codes_0, &[1, t as i32]);
        let mut total = self.codec_embedding.forward(&codes_0_arr)?; // [1, T, hidden]

        // Groups 1-15: code_predictor.codec_embeddings[i-1]
        for g in 0..15 {
            let codes_g: Vec<i32> = ref_codes.iter().map(|f| f[g + 1] as i32).collect();
            let codes_g_arr = Array::from_slice(&codes_g, &[1, t as i32]);
            let embed = self.code_predictor.codec_embeddings[g].forward(&codes_g_arr)?;
            total = total.add(embed)?;
        }

        Ok(total.as_dtype(mlx_rs::Dtype::Float32)?)
    }

    /// Build ICL prompt for voice cloning.
    ///
    /// Two modes matching the official Python implementation:
    /// - streaming (default): text and codec overlaid element-wise at each position
    /// - non_streaming: text block then codec block sequentially
    ///
    /// Returns (icl_embed, trailing_text_embed, trailing_len):
    /// - icl_embed: [1, icl_len, hidden] — the ICL block to feed through transformer
    /// - trailing_text_embed: [1, trailing_len, hidden] — leftover text for generation
    /// - trailing_len: number of trailing text tokens
    pub fn build_icl_prompt(
        &mut self,
        ref_text_ids: &[u32],      // tokenized reference text
        target_text_ids: &[u32],    // tokenized target text
        ref_codec_embed: &Array,    // [1, T_ref, hidden] from sum_ref_codec_embeddings
        tts_config: &crate::config::Qwen3TtsConfig,
        non_streaming_mode: bool,
    ) -> Result<(Array, Array, usize)> {
        let pad_id = tts_config.talker_config.codec_pad_id;
        let bos_id = tts_config.talker_config.codec_bos_id;

        // text_embed: [ref_text, target_text, tts_eos] projected [1, T1, hidden]
        let mut all_text_ids: Vec<u32> = Vec::new();
        all_text_ids.extend_from_slice(ref_text_ids);
        all_text_ids.extend_from_slice(target_text_ids);
        all_text_ids.push(tts_config.tts_eos_token_id);
        let text_embed = self.build_projected_text_embeddings(&all_text_ids)?;
        let text_lens = text_embed.dim(1) as usize;

        // codec_embed: [codec_bos, ref_codes_summed] [1, T2, hidden]
        let bos_arr = Array::from_slice(&[bos_id as i32], &[1, 1]);
        let bos_embed = self.codec_embedding.forward(&bos_arr)?;
        let codec_embed = mlx_rs::ops::concatenate_axis(
            &[&bos_embed, ref_codec_embed],
            1,
        )?;
        let codec_lens = codec_embed.dim(1) as usize;

        let tts_pad_embed = self.build_text_only_embedding(tts_config.tts_pad_token_id)?; // [1, 1, hidden]

        if non_streaming_mode {
            // Non-streaming: text block (text + codec_pad) then codec block (tts_pad + codec)
            let pad_arr = Array::from_slice(&[pad_id as i32], &[1, 1]);
            let pad_embed = self.codec_embedding.forward(&pad_arr)?;
            let pad_broadcast = mlx_rs::ops::broadcast_to(
                &pad_embed,
                &[1, text_lens as i32, pad_embed.dim(2) as i32],
            )?;
            let text_block = text_embed.add(&pad_broadcast)?;

            let tts_pad_broadcast = mlx_rs::ops::broadcast_to(
                &tts_pad_embed,
                &[1, codec_lens as i32, tts_pad_embed.dim(2) as i32],
            )?;
            let codec_block = tts_pad_broadcast.add(&codec_embed)?;

            let icl_embed = mlx_rs::ops::concatenate_axis(
                &[&text_block, &codec_block],
                1,
            )?;
            Ok((icl_embed, tts_pad_embed, 0))
        } else {
            // Streaming (default): text and codec overlaid element-wise
            if text_lens > codec_lens {
                // More text than codec: overlay first codec_lens positions, return surplus text
                let text_prefix = text_embed.index((.., ..codec_lens as i32, ..));
                let icl_embed = text_prefix.add(&codec_embed)?;
                let trailing = text_embed.index((.., codec_lens as i32.., ..));
                let trailing_len = text_lens - codec_lens;
                Ok((icl_embed, trailing, trailing_len))
            } else {
                // More codec than text (common case): pad text with tts_pad, overlay all
                let pad_count = codec_lens - text_lens;
                if pad_count > 0 {
                    let pad_broadcast = mlx_rs::ops::broadcast_to(
                        &tts_pad_embed,
                        &[1, pad_count as i32, tts_pad_embed.dim(2) as i32],
                    )?;
                    let text_padded = mlx_rs::ops::concatenate_axis(
                        &[&text_embed, &pad_broadcast],
                        1,
                    )?;
                    let icl_embed = text_padded.add(&codec_embed)?;
                    Ok((icl_embed, tts_pad_embed, 0))
                } else {
                    let icl_embed = text_embed.add(&codec_embed)?;
                    Ok((icl_embed, tts_pad_embed, 0))
                }
            }
        }
    }

    /// Build generation embedding with pre-computed text embedding.
    /// Codec: talker's codec_embedding for group 0, code_predictor's for groups 1-15.
    /// Text: provided as pre-computed projected embedding [1, 1, hidden].
    pub fn build_generation_embedding_with_text(
        &mut self,
        prev_codes: &[u32; 16],
        text_embed: &Array,
    ) -> Result<Array> {
        // Codebook 0: talker's codec_embedding
        let mut codec_embed = zeros::<f32>(&[1, 1, self.config.hidden_size])?;
        let code0_arr = Array::from_slice(&[prev_codes[0] as i32], &[1, 1]);
        codec_embed = codec_embed.add(self.codec_embedding.forward(&code0_arr)?)?;

        // Codebooks 1-15: code predictor's codec_embeddings
        for g in 0..15 {
            let code_arr = Array::from_slice(&[prev_codes[g + 1] as i32], &[1, 1]);
            let embed = self.code_predictor.codec_embeddings[g].forward(&code_arr)?;
            codec_embed = codec_embed.add(embed)?;
        }

        Ok(codec_embed.add(text_embed)?)
    }
}

// ============================================================================
// Weight Loading
// ============================================================================

fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Result<Array> {
    weights
        .get(key)
        .cloned()
        .ok_or_else(|| Error::WeightNotFound(key.to_string()))
}

fn get_weight_optional(weights: &HashMap<String, Array>, key: &str) -> Option<Array> {
    weights.get(key).cloned()
}

fn make_quantized_linear(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<nn::QuantizedLinear> {
    let weight = get_weight(weights, &format!("{prefix}.weight"))?;
    let scales = get_weight(weights, &format!("{prefix}.scales"))?;
    let biases = get_weight(weights, &format!("{prefix}.biases"))?;

    let inner = nn::Linear {
        weight: Param::new(weight),
        bias: Param::new(None),
    };

    let mut ql = nn::QuantizedLinear {
        group_size,
        bits,
        scales: Param::new(scales),
        biases: Param::new(biases),
        inner,
    };
    ql.freeze_parameters(true);
    Ok(ql)
}

fn make_quantized_linear_with_bias(
    weights: &HashMap<String, Array>,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<(nn::QuantizedLinear, Option<Array>)> {
    let ql = make_quantized_linear(weights, prefix, group_size, bits)?;
    let bias = get_weight_optional(weights, &format!("{prefix}.bias"));
    Ok((ql, bias))
}

fn make_linear(
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<nn::Linear> {
    let weight = get_weight(weights, &format!("{prefix}.weight"))?;
    let bias = get_weight_optional(weights, &format!("{prefix}.bias"));
    Ok(nn::Linear {
        weight: Param::new(weight),
        bias: Param::new(bias),
    })
}

fn make_linear_with_bias(
    weights: &HashMap<String, Array>,
    prefix: &str,
) -> Result<(nn::Linear, Option<Array>)> {
    // Return bias separately — do NOT include it in the nn::Linear too,
    // otherwise the caller applies bias twice (once inside Linear, once externally).
    // This matches make_quantized_linear which sets bias: None in the inner Linear.
    let weight = get_weight(weights, &format!("{prefix}.weight"))?;
    let bias = get_weight_optional(weights, &format!("{prefix}.bias"));
    let linear = nn::Linear {
        weight: Param::new(weight),
        bias: Param::new(None), // bias applied externally
    };
    Ok((linear, bias))
}

/// Load a linear layer as MaybeQuantized, auto-detecting from quant config.
fn load_maybe_quantized(
    weights: &HashMap<String, Array>,
    prefix: &str,
    quant: Option<&QuantizationConfig>,
) -> Result<MaybeQuantized<nn::Linear>> {
    if let Some(q) = quant {
        Ok(MaybeQuantized::Quantized(make_quantized_linear(weights, prefix, q.group_size, q.bits)?))
    } else {
        Ok(MaybeQuantized::Original(make_linear(weights, prefix)?))
    }
}

/// Load a linear layer with separate bias, auto-detecting quant.
fn load_maybe_quantized_with_bias(
    weights: &HashMap<String, Array>,
    prefix: &str,
    quant: Option<&QuantizationConfig>,
) -> Result<(MaybeQuantized<nn::Linear>, Option<Array>)> {
    if let Some(q) = quant {
        let (ql, bias) = make_quantized_linear_with_bias(weights, prefix, q.group_size, q.bits)?;
        Ok((MaybeQuantized::Quantized(ql), bias))
    } else {
        let (linear, bias) = make_linear_with_bias(weights, prefix)?;
        Ok((MaybeQuantized::Original(linear), bias))
    }
}

fn load_rms_norm(weights: &HashMap<String, Array>, prefix: &str, eps: f32) -> Result<nn::RmsNorm> {
    Ok(nn::RmsNorm {
        weight: Param::new(get_weight(weights, &format!("{prefix}.weight"))?),
        eps,
    })
}

fn load_talker_attention(
    weights: &HashMap<String, Array>,
    prefix: &str,
    config: &TalkerConfig,
    quant: Option<&QuantizationConfig>,
) -> Result<TalkerAttention> {
    // MRoPE with all 3 dims identical = standard RoPE for TTS
    // Qwen3 uses non-traditional (stride-based) RoPE, matching qwen3-mlx
    let rope = nn::RopeBuilder::new(config.head_dim)
        .base(config.rope_theta)
        .build()
        .map_err(|e| Error::Model(format!("RoPE build error: {e}")))?;

    Ok(TalkerAttention {
        n_heads: config.num_attention_heads,
        n_kv_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
        scale: (config.head_dim as f32).sqrt().recip(),
        rope,
        q_proj: load_maybe_quantized(weights, &format!("{prefix}.q_proj"), quant)?,
        k_proj: load_maybe_quantized(weights, &format!("{prefix}.k_proj"), quant)?,
        v_proj: load_maybe_quantized(weights, &format!("{prefix}.v_proj"), quant)?,
        o_proj: load_maybe_quantized(weights, &format!("{prefix}.o_proj"), quant)?,
        q_norm: load_rms_norm(weights, &format!("{prefix}.q_norm"), config.rms_norm_eps)?,
        k_norm: load_rms_norm(weights, &format!("{prefix}.k_norm"), config.rms_norm_eps)?,
    })
}

fn load_mlp(
    weights: &HashMap<String, Array>,
    prefix: &str,
    quant: Option<&QuantizationConfig>,
) -> Result<TalkerMlp> {
    Ok(TalkerMlp {
        gate_proj: load_maybe_quantized(weights, &format!("{prefix}.gate_proj"), quant)?,
        up_proj: load_maybe_quantized(weights, &format!("{prefix}.up_proj"), quant)?,
        down_proj: load_maybe_quantized(weights, &format!("{prefix}.down_proj"), quant)?,
    })
}

fn load_code_predictor_attention(
    weights: &HashMap<String, Array>,
    prefix: &str,
    config: &CodePredictorConfig,
    quant: Option<&QuantizationConfig>,
) -> Result<CodePredictorAttention> {
    let head_dim = config.head_dim;
    let rope = nn::RopeBuilder::new(head_dim)
        .base(config.rope_theta)
        .build()
        .map_err(|e| Error::Model(format!("RoPE build error: {e}")))?;

    Ok(CodePredictorAttention {
        n_heads: config.num_attention_heads,
        n_kv_heads: config.num_key_value_heads,
        head_dim,
        scale: (head_dim as f32).sqrt().recip(),
        q_proj: load_maybe_quantized(weights, &format!("{prefix}.q_proj"), quant)?,
        k_proj: load_maybe_quantized(weights, &format!("{prefix}.k_proj"), quant)?,
        v_proj: load_maybe_quantized(weights, &format!("{prefix}.v_proj"), quant)?,
        o_proj: load_maybe_quantized(weights, &format!("{prefix}.o_proj"), quant)?,
        q_norm: load_rms_norm(weights, &format!("{prefix}.q_norm"), config.rms_norm_eps())?,
        k_norm: load_rms_norm(weights, &format!("{prefix}.k_norm"), config.rms_norm_eps())?,
        rope,
    })
}

pub fn load_talker(model_dir: &Path, config: &TalkerConfig, quant: Option<&QuantizationConfig>, tts_pad_token_id: u32) -> Result<Talker> {
    // Load all weights
    let weights = load_all_weights(model_dir)?;

    // Text embedding (NOT quantized)
    let text_embedding = nn::Embedding {
        weight: Param::new(get_weight(&weights, "talker.model.text_embedding.weight")?),
    };

    // Codec embedding (NOT quantized, shared for all 16 groups)
    let codec_embedding = nn::Embedding {
        weight: Param::new(get_weight(&weights, "talker.model.codec_embedding.weight")?),
    };

    // Text projection
    let (fc1, fc1_bias) = load_maybe_quantized_with_bias(
        &weights,
        "talker.text_projection.linear_fc1",
        quant,
    )?;
    let (fc2, fc2_bias) = load_maybe_quantized_with_bias(
        &weights,
        "talker.text_projection.linear_fc2",
        quant,
    )?;
    let text_projection = TextProjection {
        fc1,
        fc1_bias,
        fc2,
        fc2_bias,
    };

    // Transformer layers
    let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
    for i in 0..config.num_hidden_layers {
        let prefix = format!("talker.model.layers.{i}");
        let attn = load_talker_attention(
            &weights,
            &format!("{prefix}.self_attn"),
            config,
            quant,
        )?;
        let mlp = load_mlp(&weights, &format!("{prefix}.mlp"), quant)?;
        let block = TalkerBlock {
            self_attn: attn,
            mlp,
            input_layernorm: load_rms_norm(
                &weights,
                &format!("{prefix}.input_layernorm"),
                config.rms_norm_eps,
            )?,
            post_attention_layernorm: load_rms_norm(
                &weights,
                &format!("{prefix}.post_attention_layernorm"),
                config.rms_norm_eps,
            )?,
        };
        layers.push(block);
    }

    let norm = load_rms_norm(&weights, "talker.model.norm", config.rms_norm_eps)?;
    let codec_head = load_maybe_quantized(&weights, "talker.codec_head", quant)?;

    // Code predictor
    let cp_config = &config.code_predictor_config;
    let mut cp_layers = Vec::with_capacity(cp_config.num_hidden_layers as usize);
    for i in 0..cp_config.num_hidden_layers {
        let prefix = format!("talker.code_predictor.model.layers.{i}");
        let attn = load_code_predictor_attention(
            &weights,
            &format!("{prefix}.self_attn"),
            cp_config,
            quant,
        )?;
        let mlp = load_mlp(&weights, &format!("{prefix}.mlp"), quant)?;
        let block = CodePredictorBlock {
            self_attn: attn,
            mlp,
            input_layernorm: load_rms_norm(
                &weights,
                &format!("{prefix}.input_layernorm"),
                cp_config.rms_norm_eps(),
            )?,
            post_attention_layernorm: load_rms_norm(
                &weights,
                &format!("{prefix}.post_attention_layernorm"),
                cp_config.rms_norm_eps(),
            )?,
        };
        cp_layers.push(block);
    }

    let cp_norm = load_rms_norm(
        &weights,
        "talker.code_predictor.model.norm",
        cp_config.rms_norm_eps(),
    )?;

    // Code predictor embeddings (15 for codebooks 1-15)
    let mut codec_embeddings = Vec::with_capacity(15);
    for i in 0..15 {
        let w = get_weight(
            &weights,
            &format!("talker.code_predictor.model.codec_embedding.{i}.weight"),
        )?;
        codec_embeddings.push(nn::Embedding {
            weight: Param::new(w),
        });
    }

    // Code predictor LM heads (15)
    let mut lm_heads = Vec::with_capacity(15);
    for i in 0..15 {
        lm_heads.push(load_maybe_quantized(
            &weights,
            &format!("talker.code_predictor.lm_head.{i}"),
            quant,
        )?);
    }

    let (mtp_proj, mtp_proj_bias) = load_maybe_quantized_with_bias(
        &weights,
        "talker.code_predictor.small_to_mtp_projection",
        quant,
    )?;

    let code_predictor = CodePredictor {
        layers: cp_layers,
        norm: cp_norm,
        codec_embeddings,
        lm_heads,
        small_to_mtp_projection: mtp_proj,
        mtp_proj_bias,
    };

    let num_layers = layers.len();
    Ok(Talker {
        config: config.clone(),
        tts_pad_token_id,
        text_embedding,
        codec_embedding,
        text_projection,
        layers,
        norm,
        codec_head,
        code_predictor,
        caches: (0..num_layers).map(|_| KVCache::new()).collect(),
    })
}

/// Load all weights from model.safetensors (or sharded index).
/// Public for speaker encoder loading.
pub fn load_all_weights(model_dir: &Path) -> Result<HashMap<String, Array>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let json = std::fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&json)?;
        let weight_map = index["weight_map"]
            .as_object()
            .ok_or_else(|| Error::Config("Invalid weight index".to_string()))?;

        let files: std::collections::HashSet<&str> =
            weight_map.values().filter_map(|v| v.as_str()).collect();

        let mut all_weights = HashMap::new();
        for file in files {
            let path = model_dir.join(file);
            let loaded = Array::load_safetensors(&path)?;
            all_weights.extend(loaded);
        }
        Ok(all_weights)
    } else {
        let path = model_dir.join("model.safetensors");
        Ok(Array::load_safetensors(&path)?)
    }
}
