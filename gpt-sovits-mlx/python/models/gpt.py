"""GPT model for GPT-SoVITS semantic token generation.

This is the core autoregressive model that generates semantic tokens
from phoneme sequences and audio conditioning.

Architecture:
- Phoneme embedding + position encoding
- Optional audio feature projection (from CNHubert)
- N transformer decoder blocks
- Output projection to semantic vocabulary

Based on GPT-SoVITS with optimizations for MLX.
"""

from typing import Optional, Tuple, List
import mlx.core as mx
import mlx.nn as nn

from python.models.config import GPTConfig
from python.models.attention import Attention, CrossAttention
from python.models.mlp import MLP
from python.models.cache import KVCache, create_kv_caches


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    More efficient than LayerNorm as it doesn't require mean computation.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((hidden_size,))

    def __call__(self, x: mx.array) -> mx.array:
        # RMS = sqrt(mean(x^2))
        rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.weight


class LayerNorm(nn.Module):
    """Standard Layer Normalization with weight and bias.

    Compatible with PyTorch LayerNorm format.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((hidden_size,))
        self.bias = mx.zeros((hidden_size,))

    def __call__(self, x: mx.array) -> mx.array:
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / mx.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias


class TransformerBlock(nn.Module):
    """Single transformer decoder block.

    Structure:
        x -> Norm -> SelfAttention -> + -> Norm -> MLP -> +
             └────────────────────────┘    └────────────┘

    Optionally includes cross-attention for audio conditioning.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        rope_theta: float = 10000.0,
        max_seq_len: int = 2048,
        norm_eps: float = 1e-6,
        use_cross_attention: bool = False,
        cross_attention_dim: Optional[int] = None,
        use_layernorm: bool = False,
        use_gelu: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.use_cross_attention = use_cross_attention

        # Choose normalization type
        Norm = LayerNorm if use_layernorm else RMSNorm

        # Pre-attention norm
        self.input_layernorm = Norm(hidden_size, eps=norm_eps)

        # Self attention (use bias when in original GPT-SoVITS mode)
        self.self_attn = Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
            bias=use_layernorm,  # Original GPT-SoVITS uses bias
        )

        # Optional cross-attention for conditioning
        if use_cross_attention:
            self.cross_attn_norm = Norm(hidden_size, eps=norm_eps)
            self.cross_attn = CrossAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                encoder_dim=cross_attention_dim or hidden_size,
            )

        # Pre-MLP norm
        self.post_attention_layernorm = Norm(hidden_size, eps=norm_eps)

        # MLP (supports GELU or SwiGLU)
        # Use bias when in GELU mode (original GPT-SoVITS style)
        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            use_gelu=use_gelu,
            bias=use_gelu,  # Original GPT-SoVITS uses bias
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        encoder_output: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[KVCache]]:
        """Forward pass.

        Args:
            x: Input tensor [batch, seq, hidden]
            mask: Optional attention mask
            cache: Optional KV cache
            encoder_output: Optional encoder output for cross-attention

        Returns:
            Tuple of (output tensor, updated cache)
        """
        # Self-attention with residual
        residual = x
        x = self.input_layernorm(x)
        x, cache = self.self_attn(x, mask=mask, cache=cache)
        x = residual + x

        # Cross-attention (if enabled and encoder output provided)
        if self.use_cross_attention and encoder_output is not None:
            residual = x
            x = self.cross_attn_norm(x)
            x = self.cross_attn(x, encoder_output)
            x = residual + x

        # MLP with residual
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x, cache


class GPTSoVITS(nn.Module):
    """GPT model for semantic token generation.

    Takes phoneme IDs and optional audio features, outputs semantic token logits.
    """

    def __init__(self, config: GPTConfig):
        """Initialize GPT model.

        Args:
            config: Model configuration
        """
        super().__init__()

        self.config = config

        # Token embeddings
        self.phoneme_embed = nn.Embedding(config.phoneme_vocab_size, config.hidden_size)
        self.semantic_embed = nn.Embedding(config.semantic_vocab_size, config.hidden_size)

        # BERT feature projection (RoBERTa 1024-dim -> hidden_size)
        # Note: This projects BERT text features, not CNHubert audio features
        # The original GPT-SoVITS uses BERT features for conditioning
        bert_dim = config.text_feature_dim if config.text_feature_dim else 1024
        self.bert_proj = nn.Linear(bert_dim, config.hidden_size)

        # Note: text_proj is not needed since bert_proj handles BERT/RoBERTa features
        self.text_proj = None

        # Get architecture options from config
        use_layernorm = getattr(config, 'use_layernorm', False)
        use_gelu = getattr(config, 'use_gelu', False)
        use_cross_attn = getattr(config, 'use_cross_attention', True)

        # Transformer blocks
        self.layers = [
            TransformerBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                intermediate_size=config.intermediate_size,
                rope_theta=config.rope_theta,
                max_seq_len=config.max_seq_len,
                norm_eps=config.rms_norm_eps,
                use_cross_attention=use_cross_attn,
                cross_attention_dim=config.hidden_size,
                use_layernorm=use_layernorm,
                use_gelu=use_gelu,
            )
            for _ in range(config.num_layers)
        ]

        # Output normalization
        Norm = LayerNorm if use_layernorm else RMSNorm
        self.norm = Norm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.semantic_vocab_size, bias=False)

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def __call__(
        self,
        phoneme_ids: mx.array,
        semantic_ids: mx.array,
        bert_features: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[List[KVCache]] = None,
    ) -> Tuple[mx.array, Optional[List[KVCache]]]:
        """Forward pass.

        Args:
            phoneme_ids: Phoneme token IDs [batch, phoneme_seq]
            semantic_ids: Semantic token IDs [batch, semantic_seq]
            bert_features: BERT/RoBERTa features [batch, text_seq, 1024]
            mask: Optional attention mask
            cache: Optional list of KV caches (one per layer)

        Returns:
            Tuple of (logits [batch, seq, vocab], updated caches)
        """
        batch_size = phoneme_ids.shape[0]

        # Embed inputs
        phoneme_emb = self.phoneme_embed(phoneme_ids)  # [batch, phoneme_seq, hidden]
        semantic_emb = self.semantic_embed(semantic_ids)  # [batch, semantic_seq, hidden]

        # Project BERT features to hidden_size
        bert_emb = self.bert_proj(bert_features)  # [batch, text_seq, hidden]

        # Combine embeddings
        # For generation: phoneme context + current semantic token
        # The decoder attends to phoneme+bert, generates semantic
        x = semantic_emb

        # Condition on phoneme and BERT features
        # Concatenate phoneme and BERT embeddings as context
        encoder_output = mx.concatenate([phoneme_emb, bert_emb], axis=1)

        # Run through transformer layers
        new_cache = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, layer_cache = layer(
                x,
                mask=mask,
                cache=layer_cache,
                encoder_output=encoder_output,
            )
            new_cache.append(layer_cache)

        # Output projection
        x = self.norm(x)
        logits = self.lm_head(x)

        return logits, new_cache if cache is not None else None

    def create_caches(self, dtype: mx.Dtype = mx.float32) -> List[KVCache]:
        """Create KV caches for all layers.

        Args:
            dtype: Data type for caches

        Returns:
            List of KVCache instances
        """
        return create_kv_caches(
            num_layers=self.num_layers,
            num_heads=self.config.num_heads,
            head_dim=self.config.head_dim,
            step=256,
            dtype=dtype,
        )


def load_gpt_model(
    weights_path: str,
    config_path: Optional[str] = None,
    config: Optional[GPTConfig] = None,
    strict: bool = False,
) -> GPTSoVITS:
    """Load GPT model from weights file.

    Args:
        weights_path: Path to safetensors weights file
        config_path: Optional path to config JSON
        config: Optional config object (takes precedence over config_path)
        strict: If True, raise error on missing/extra weights

    Returns:
        Loaded GPTSoVITS model
    """
    from safetensors import safe_open

    # Load or create config
    if config is None:
        if config_path is not None:
            config = GPTConfig.from_json(config_path)
        else:
            config = GPTConfig()

    # Create model
    model = GPTSoVITS(config)

    # Load weights
    weights = {}
    with safe_open(weights_path, framework="mlx") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)

    # Remap weight names from converted format to model format
    # The converter named BERT projection as audio_proj, but model uses bert_proj
    remap_keys = {
        "audio_proj.weight": "bert_proj.weight",
        "audio_proj.bias": "bert_proj.bias",
    }
    remapped_weights = {}
    for key, value in weights.items():
        new_key = remap_keys.get(key, key)
        remapped_weights[new_key] = value
    weights = remapped_weights

    # Convert flat dotted keys to hierarchical dict
    def unflatten_weights(flat_weights):
        """Convert flat dotted keys to nested dict/list structure.

        Examples:
            'audio_proj.weight' -> {'audio_proj': {'weight': ...}}
            'layers.0.self_attn.q_proj.weight' -> {'layers': [{..., 'self_attn': {'q_proj': {'weight': ...}}}]}
        """
        result = {}
        for key, value in flat_weights.items():
            parts = key.split(".")
            current = result

            for i, part in enumerate(parts[:-1]):
                next_part = parts[i + 1] if i + 1 < len(parts) else None

                # Check if next part is a digit (we're entering a list)
                if next_part is not None and next_part.isdigit():
                    # Create list if needed
                    if part not in current:
                        current[part] = []
                    current = current[part]
                elif part.isdigit():
                    # We're at a list index
                    idx = int(part)
                    while len(current) <= idx:
                        current.append({})
                    current = current[idx]
                else:
                    # Regular dict key
                    if part not in current:
                        current[part] = {}
                    current = current[part]

            # Set the final value
            final_key = parts[-1]
            current[final_key] = value

        return result

    # Get model parameter names recursively (flat)
    def get_param_names(params, prefix=""):
        names = set()
        if isinstance(params, list):
            for i, item in enumerate(params):
                names.update(get_param_names(item, f"{prefix}.{i}" if prefix else str(i)))
        elif isinstance(params, dict):
            for k, v in params.items():
                name = f"{prefix}.{k}" if prefix else k
                if isinstance(v, (dict, list)):
                    names.update(get_param_names(v, name))
                else:
                    names.add(name)
        return names

    model_params = get_param_names(model.parameters())

    # Filter weights to only include those the model has
    filtered_weights = {}
    skipped = []
    for key, value in weights.items():
        if key in model_params:
            filtered_weights[key] = value
        else:
            skipped.append(key)

    if skipped:
        print(f"  Skipped {len(skipped)} unknown params: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")

    # Check for missing parameters
    loaded_keys = set(filtered_weights.keys())
    missing = model_params - loaded_keys
    if missing and strict:
        raise ValueError(f"Missing parameters: {missing}")
    elif missing:
        print(f"  Missing {len(missing)} params (using defaults): {list(missing)[:5]}{'...' if len(missing) > 5 else ''}")

    # Convert to hierarchical structure and apply
    if filtered_weights:
        hierarchical = unflatten_weights(filtered_weights)
        model.update(hierarchical)

    return model
