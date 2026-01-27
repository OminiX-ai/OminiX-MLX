"""Configuration classes for GPT-SoVITS models."""

from dataclasses import dataclass, field
from typing import Optional
import json
from pathlib import Path


@dataclass
class GPTConfig:
    """Configuration for GPT semantic token generator.

    Based on GPT-SoVITS architecture:
    - 12-layer transformer decoder
    - 512 hidden dimension
    - 8 attention heads
    - Phoneme input → Semantic token output
    """

    # Model dimensions
    hidden_size: int = 512
    num_layers: int = 12
    num_heads: int = 8
    head_dim: int = 64  # hidden_size // num_heads
    intermediate_size: int = 2048  # 4 * hidden_size

    # Vocabulary
    phoneme_vocab_size: int = 512
    semantic_vocab_size: int = 1025  # 1024 codes + 1 EOS

    # Position encoding
    max_seq_len: int = 1024
    rope_theta: float = 10000.0

    # Audio conditioning
    audio_feature_dim: int = 768  # CNHubert output dimension

    # Normalization
    rms_norm_eps: float = 1e-6

    # Optional text features (RoBERTa)
    text_feature_dim: Optional[int] = 1024
    use_text_features: bool = False

    # Architecture options (for compatibility with original GPT-SoVITS)
    use_layernorm: bool = False  # Use LayerNorm instead of RMSNorm
    use_gelu: bool = False       # Use GELU instead of SwiGLU
    use_cross_attention: bool = True  # Use cross-attention (False for concat mode)

    def __post_init__(self):
        assert self.hidden_size % self.num_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
        self.head_dim = self.hidden_size // self.num_heads

    @classmethod
    def from_json(cls, path: str | Path) -> "GPTConfig":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_json(self, path: str | Path) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)


@dataclass
class VocoderConfig:
    """Configuration for SoVITS vocoder.

    Based on SoVITS architecture:
    - Duration predictor (flow-based)
    - RVQ decoder (8 codebooks)
    - Upsampler (transposed convolutions)
    """

    # Input
    semantic_dim: int = 512
    audio_feature_dim: int = 768

    # Duration predictor
    duration_channels: int = 256
    duration_kernel_size: int = 3
    duration_num_flows: int = 4

    # RVQ
    num_codebooks: int = 8
    codebook_size: int = 1024
    codebook_dim: int = 256

    # Upsampler
    upsample_rates: tuple = (8, 4, 2, 2)  # Total: 128x
    upsample_kernel_sizes: tuple = (16, 8, 4, 4)
    upsample_channels: int = 256
    resblock_kernel_sizes: tuple = (3, 7, 11)
    resblock_dilation_sizes: tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5))

    # Output
    sample_rate: int = 32000

    @classmethod
    def from_json(cls, path: str | Path) -> "VocoderConfig":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SynthesisConfig:
    """Configuration for TTS synthesis."""

    # Voice
    voice_name: str = "Doubao"
    language: str = "zh"

    # Sampling parameters
    temperature: float = 0.8
    top_k: int = 3
    top_p: float = 0.95
    repetition_penalty: float = 1.0

    # Generation limits
    max_semantic_tokens: int = 500
    min_semantic_tokens: int = 10

    # Audio
    speed_factor: float = 1.0
    sample_rate: int = 32000

    # Streaming
    streaming: bool = False
    chunk_size: int = 4096  # samples per chunk

    # Hardware
    use_ane: bool = True  # Use ANE for encoders
    use_compile: bool = True  # Use mx.compile


@dataclass
class ModelPaths:
    """Paths to model files."""

    # Encoders (CoreML)
    cnhubert_path: Optional[str] = None
    roberta_path: Optional[str] = None

    # GPT (MLX)
    gpt_weights_path: Optional[str] = None
    gpt_config_path: Optional[str] = None

    # Vocoder (MLX)
    vocoder_weights_path: Optional[str] = None
    vocoder_config_path: Optional[str] = None

    # Voice-specific
    reference_audio_path: Optional[str] = None

    @classmethod
    def from_model_dir(cls, model_dir: str | Path, voice_name: str = "Doubao") -> "ModelPaths":
        """Create paths from a model directory structure."""
        model_dir = Path(model_dir)
        voice_dir = model_dir / "voices" / voice_name

        return cls(
            cnhubert_path=str(model_dir / "encoders" / "cnhubert_ane.mlpackage"),
            roberta_path=str(model_dir / "encoders" / "roberta_ane.mlpackage"),
            gpt_weights_path=str(voice_dir / "gpt.safetensors"),
            gpt_config_path=str(voice_dir / "gpt_config.json"),
            vocoder_weights_path=str(voice_dir / "sovits.safetensors"),
            vocoder_config_path=str(voice_dir / "sovits_config.json"),
            reference_audio_path=str(voice_dir / "reference.wav"),
        )
