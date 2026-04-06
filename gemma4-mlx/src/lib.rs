//! Gemma 4 text-only inference on Apple Silicon with MLX.

pub mod chat;
pub mod model;

pub use mlx_rs_core::{
    cache::{ConcatKeyValueCache, KVCache, KeyValueCache},
    error::{Error, Result},
    sampler::{DefaultSampler, Sampler},
    utils::{
        create_attention_mask, create_causal_mask, scaled_dot_product_attention, AttentionMask,
        SdpaMask,
    },
};

pub use chat::{
    Gemma4ChatConfig, Gemma4ChatPipeline, Gemma4ChatResponse, Gemma4ChatTemplate,
    Gemma4Conversation, Gemma4FunctionTool, Gemma4Message, Gemma4ParsedAssistantResponse,
    Gemma4Role, Gemma4SpecialTokens, Gemma4Tool, Gemma4ToolCall, Gemma4ToolRegistry,
    Gemma4ToolResult, Gemma4ToolSpec, EOS_TOKEN_IDS,
};

pub use model::{
    get_model_args, init_cache, load_model, load_model_with_overrides, load_tokenizer, Attention,
    AttentionInput, DecoderLayer, DecoderLayerInput, DenseMlp, Experts, Gemma4Config,
    Gemma4TextConfig, Generate, GenerateState, LanguageModel, Model, ModelInput, Router,
    UnscaledRmsNorm,
};
