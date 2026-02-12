use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_hidden_layers: i32,
    pub num_key_value_heads: i32,
    pub vocab_size: i32,
    pub head_dim: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: i32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,

    // Per-layer mixer types: "minicpm4" or "lightning-attn"
    pub mixer_types: Vec<String>,

    // Sparse attention config (for minicpm4 layers)
    #[serde(default)]
    pub sparse_config: Option<SparseConfig>,

    // muP scaling
    #[serde(default = "default_scale_emb")]
    pub scale_emb: f32,
    #[serde(default = "default_scale_depth")]
    pub scale_depth: f32,
    #[serde(default = "default_dim_model_base")]
    pub dim_model_base: i32,

    // RoPE per attention type
    #[serde(default)]
    pub attn_use_rope: bool,
    #[serde(default = "default_true")]
    pub lightning_use_rope: bool,

    // QK normalization
    #[serde(default)]
    pub qk_norm: bool,

    // Output gating
    #[serde(default)]
    pub use_output_gate: bool,
    #[serde(default)]
    pub use_output_norm: bool,
    #[serde(default)]
    pub attn_use_output_gate: bool,

    // Lightning attention parameters
    #[serde(default)]
    pub lightning_nh: Option<i32>,
    #[serde(default)]
    pub lightning_nkv: Option<i32>,
    #[serde(default)]
    pub lightning_head_dim: Option<i32>,
    #[serde(default = "default_lightning_scale")]
    pub lightning_scale: String,

    // Quantization (present if model is quantized)
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SparseConfig {
    #[serde(default = "default_32")]
    pub kernel_size: i32,
    #[serde(default = "default_16")]
    pub kernel_stride: i32,
    #[serde(default = "default_1")]
    pub init_blocks: i32,
    #[serde(default = "default_64")]
    pub block_size: i32,
    #[serde(default = "default_2048")]
    pub window_size: i32,
    #[serde(default = "default_64")]
    pub topk: i32,
    #[serde(default)]
    pub use_nope: bool,
    #[serde(default = "default_8192")]
    pub dense_len: i32,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct QuantizationConfig {
    #[serde(default = "default_64")]
    pub group_size: i32,
    #[serde(default = "default_4")]
    pub bits: i32,
}

impl ModelArgs {
    /// Number of lightning attention heads (defaults to num_attention_heads)
    pub fn lightning_num_heads(&self) -> i32 {
        self.lightning_nh.unwrap_or(self.num_attention_heads)
    }

    /// Number of lightning KV heads (defaults to num_key_value_heads)
    pub fn lightning_num_kv_heads(&self) -> i32 {
        self.lightning_nkv.unwrap_or(self.num_key_value_heads)
    }

    /// Lightning head dimension (defaults to head_dim)
    pub fn lightning_head_dim(&self) -> i32 {
        self.lightning_head_dim.unwrap_or(self.head_dim)
    }

    /// Lightning attention scale factor
    pub fn lightning_scale_value(&self) -> f32 {
        match self.lightning_scale.as_str() {
            "1/sqrt(d)" => (self.lightning_head_dim() as f32).sqrt().recip(),
            "1/d" => (self.lightning_head_dim() as f32).recip(),
            _ => 1.0,
        }
    }

    /// Residual scaling factor: scale_depth / sqrt(num_hidden_layers)
    pub fn residual_scale(&self) -> f32 {
        self.scale_depth / (self.num_hidden_layers as f32).sqrt()
    }

    /// Logits scaling denominator: hidden_size / dim_model_base
    pub fn logits_scale(&self) -> f32 {
        self.hidden_size as f32 / self.dim_model_base as f32
    }

    /// Check if a layer is sparse (minicpm4) vs lightning
    pub fn is_sparse_layer(&self, layer_idx: usize) -> bool {
        self.mixer_types
            .get(layer_idx)
            .map(|t| t == "minicpm4")
            .unwrap_or(false)
    }
}

fn default_max_position_embeddings() -> i32 { 524288 }
fn default_scale_emb() -> f32 { 1.0 }
fn default_scale_depth() -> f32 { 1.0 }
fn default_dim_model_base() -> i32 { 256 }
fn default_true() -> bool { true }
fn default_lightning_scale() -> String { "1/sqrt(d)".to_string() }
fn default_32() -> i32 { 32 }
fn default_16() -> i32 { 16 }
fn default_1() -> i32 { 1 }
fn default_64() -> i32 { 64 }
fn default_4() -> i32 { 4 }
fn default_2048() -> i32 { 2048 }
fn default_8192() -> i32 { 8192 }
