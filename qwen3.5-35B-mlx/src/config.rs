use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, Default)]
pub struct QuantizationConfig {
    #[serde(default = "default_group_size")]
    pub group_size: i32,
    #[serde(default = "default_bits")]
    pub bits: i32,
}

fn default_group_size() -> i32 {
    64
}
fn default_bits() -> i32 {
    4
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f32,
    #[serde(default)]
    pub rope_type: Option<String>,
}

fn default_rope_theta() -> f32 {
    10_000_000.0
}
fn default_partial_rotary_factor() -> f32 {
    0.25
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub hidden_size: i32,
    pub num_hidden_layers: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub intermediate_size: i32,
    pub rms_norm_eps: f32,
    pub vocab_size: i32,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: i32,
    pub layer_types: Vec<String>,

    // Linear attention (DeltaNet) config
    pub linear_num_key_heads: i32,
    pub linear_num_value_heads: i32,
    pub linear_key_head_dim: i32,
    pub linear_value_head_dim: i32,
    pub linear_conv_kernel_dim: i32,

    // RoPE
    pub rope_parameters: RopeParameters,

    // Output gate for full attention
    #[serde(default)]
    pub attn_output_gate: bool,

    // Quantization (sometimes inside text_config)
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

fn default_max_pos() -> i32 {
    262144
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub text_config: TextConfig,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    // Quantization may be at top level
    #[serde(default)]
    pub quantization: Option<QuantizationConfig>,
}

impl ModelArgs {
    /// Get the effective quantization config (top-level takes precedence)
    pub fn quantization(&self) -> Option<&QuantizationConfig> {
        self.quantization
            .as_ref()
            .or(self.text_config.quantization.as_ref())
    }
}
