"""SoVITS Vocoder for audio synthesis.

Combines all vocoder components:
- Semantic to acoustic mapping (RVQ)
- Duration prediction and length regulation
- Upsampling to audio waveform

This is the second stage of GPT-SoVITS: semantic tokens -> audio.
"""

from typing import Optional, Tuple
from dataclasses import dataclass
import mlx.core as mx
import mlx.nn as nn

from python.models.config import VocoderConfig
from python.models.vq import ResidualVectorQuantizer, SemanticToAcoustic
from python.models.duration import (
    DurationPredictor,
    StochasticDurationPredictor,
    LengthRegulator,
)
from python.models.upsampler import Upsampler, SimplifiedUpsampler


class MRTE(nn.Module):
    """Multi-Resolution Temporal Encoding.

    Captures temporal patterns at multiple scales using dilated convolutions.
    Used to condition the vocoder on reference audio style.

    Uses MLX's channels-last format: [batch, time, channels]
    """

    def __init__(
        self,
        channels: int = 256,
        hidden_channels: int = 256,
        kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        num_layers: int = 3,
    ):
        super().__init__()

        self.channels = channels
        self.hidden_channels = hidden_channels

        # Multi-scale convolutions with same padding to preserve time dimension
        self.convs = []
        for kernel_size in kernel_sizes:
            # Use same padding: (kernel_size - 1) // 2 for symmetric padding
            padding = (kernel_size - 1) // 2
            conv_layers = []
            for i in range(num_layers):
                in_ch = channels if i == 0 else hidden_channels
                conv_layers.append(
                    nn.Conv1d(in_ch, hidden_channels, kernel_size, padding=padding)
                )
            self.convs.append(conv_layers)

        # Output projection
        self.proj = nn.Conv1d(hidden_channels * len(kernel_sizes), channels, 1)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply MRTE.

        Args:
            x: Input features [batch, time, channels] (channels-last)

        Returns:
            Temporally encoded features [batch, time, channels]
        """
        outputs = []
        original_time = x.shape[1]

        for conv_layers in self.convs:
            h = x
            for conv in conv_layers:
                h = conv(h)
                h = nn.leaky_relu(h, negative_slope=0.1)
            # Ensure time dimension matches original (truncate if needed)
            if h.shape[1] != original_time:
                h = h[:, :original_time, :]
            outputs.append(h)

        # Concatenate multi-scale outputs along channel dimension
        concat = mx.concatenate(outputs, axis=-1)

        # Project back to original channels
        return self.proj(concat)


class SoVITSVocoder(nn.Module):
    """SoVITS Vocoder: Semantic tokens -> Audio waveform.

    Pipeline:
    1. Decode semantic tokens via RVQ
    2. Predict durations for each token
    3. Expand features according to durations
    4. Apply MRTE for style conditioning
    5. Upsample to audio waveform

    Uses MLX's channels-last format: [batch, time, channels]
    """

    def __init__(self, config: VocoderConfig):
        """Initialize vocoder.

        Args:
            config: Vocoder configuration
        """
        super().__init__()

        self.config = config

        # Semantic to acoustic (RVQ)
        self.semantic_decoder = SemanticToAcoustic(
            semantic_dim=config.semantic_dim,
            acoustic_dim=config.codebook_dim,
            num_codebooks=config.num_codebooks,
            codebook_size=config.codebook_size,
        )

        # Duration prediction
        self.duration_predictor = DurationPredictor(
            in_channels=config.codebook_dim,
            hidden_channels=config.duration_channels,
            kernel_size=config.duration_kernel_size,
        )

        # Length regulator
        self.length_regulator = LengthRegulator()

        # Audio feature projection (for style conditioning from reference audio)
        self.audio_feature_proj = nn.Linear(config.audio_feature_dim, config.codebook_dim)

        # MRTE for style conditioning
        self.mrte = MRTE(
            channels=config.codebook_dim,
            hidden_channels=config.codebook_dim,
        )

        # Audio projection (to upsampler input dim)
        self.audio_proj = nn.Conv1d(
            config.codebook_dim,
            config.upsample_channels,
            1,
        )

        # Upsampler
        self.upsampler = SimplifiedUpsampler(
            in_channels=config.upsample_channels,
            hidden_channels=config.upsample_channels,
            out_channels=1,
            upsample_factor=int(config.sample_rate / 50),  # 50Hz features to sample_rate
        )

    def __call__(
        self,
        semantic_tokens: mx.array,
        audio_features: Optional[mx.array] = None,
        durations: Optional[mx.array] = None,
        speed_factor: float = 1.0,
    ) -> mx.array:
        """Synthesize audio from semantic tokens.

        Args:
            semantic_tokens: Semantic token indices [batch, seq]
            audio_features: Optional reference audio features for style [batch, time, feat_dim]
            durations: Optional pre-computed durations [batch, seq]
            speed_factor: Speed adjustment (1.0 = normal, <1 = slower, >1 = faster)

        Returns:
            Audio waveform [batch, samples, 1]
        """
        batch_size, seq_len = semantic_tokens.shape

        # 1. Decode semantic tokens to acoustic features [batch, seq, dim]
        acoustic = self.semantic_decoder(semantic_tokens)  # [batch, seq, dim]
        # acoustic is already in channels-last format

        # 2. Predict durations if not provided
        if durations is None:
            log_durations = self.duration_predictor(acoustic)  # [batch, seq]
            durations = mx.exp(log_durations)

        # Apply speed factor
        durations = durations / speed_factor

        # 3. Expand features according to durations
        expanded, out_lens = self.length_regulator(acoustic, durations)
        # expanded: [batch, expanded_time, dim] (channels-last)

        # 4. Apply MRTE for temporal encoding
        # If we have reference audio, use it for style
        if audio_features is not None:
            # Project audio features to codebook dimension
            audio_feat_proj = self.audio_feature_proj(audio_features)  # [batch, time, codebook_dim]
            # Apply MRTE for style encoding
            style = self.mrte(audio_feat_proj)
            # Interpolate style to match expanded length
            if style.shape[1] != expanded.shape[1]:
                # Simple repetition to match length
                ratio = expanded.shape[1] / style.shape[1]
                style = mx.repeat(style, int(ratio) + 1, axis=1)[:, :expanded.shape[1], :]
            expanded = expanded + 0.1 * style  # Add style with small weight

        # 5. Project to upsampler input dimension
        x = self.audio_proj(expanded)  # [batch, time, upsample_channels]
        x = nn.leaky_relu(x, negative_slope=0.1)

        # 6. Upsample to audio
        audio = self.upsampler(x)  # [batch, samples, 1]

        return audio

    def forward_with_durations(
        self,
        semantic_tokens: mx.array,
        durations: mx.array,
        audio_features: Optional[mx.array] = None,
    ) -> mx.array:
        """Synthesize with explicit durations (for training/fine-tuning).

        Args:
            semantic_tokens: Semantic tokens [batch, seq]
            durations: Ground-truth durations [batch, seq]
            audio_features: Optional reference features

        Returns:
            Audio waveform
        """
        return self(
            semantic_tokens,
            audio_features=audio_features,
            durations=durations,
        )


def load_vocoder(
    weights_path: str,
    config_path: Optional[str] = None,
    config: Optional[VocoderConfig] = None,
) -> SoVITSVocoder:
    """Load vocoder from weights file.

    Args:
        weights_path: Path to safetensors weights
        config_path: Optional path to config JSON
        config: Optional config object

    Returns:
        Loaded SoVITSVocoder
    """
    from safetensors import safe_open

    # Load or create config
    if config is None:
        if config_path is not None:
            config = VocoderConfig.from_json(config_path)
        else:
            config = VocoderConfig()

    # Create model
    model = SoVITSVocoder(config)

    # Load weights
    weights = {}
    with safe_open(weights_path, framework="mlx") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)

    model.update(weights)

    return model


@dataclass
class VocoderOutput:
    """Output from vocoder synthesis."""

    audio: mx.array  # [batch, 1, samples]
    durations: mx.array  # [batch, seq]
    expanded_len: int  # Length after expansion


def synthesize_audio(
    vocoder: SoVITSVocoder,
    semantic_tokens: mx.array,
    audio_features: Optional[mx.array] = None,
    speed_factor: float = 1.0,
    sample_rate: int = 32000,
) -> VocoderOutput:
    """High-level function to synthesize audio from semantic tokens.

    Args:
        vocoder: Vocoder model
        semantic_tokens: Semantic tokens from GPT
        audio_features: Reference audio features (for style) [batch, time, feat_dim]
        speed_factor: Speed adjustment
        sample_rate: Target sample rate

    Returns:
        VocoderOutput with audio and metadata
    """
    # Get acoustic features for duration prediction [batch, seq, dim]
    acoustic = vocoder.semantic_decoder(semantic_tokens)
    # acoustic is already in channels-last format

    # Predict durations
    log_durations = vocoder.duration_predictor(acoustic)
    durations = mx.exp(log_durations) / speed_factor

    # Synthesize
    audio = vocoder(
        semantic_tokens,
        audio_features=audio_features,
        durations=durations,
        speed_factor=1.0,  # Already applied to durations
    )

    return VocoderOutput(
        audio=audio,
        durations=durations,
        expanded_len=audio.shape[1],  # channels-last: [batch, samples, 1]
    )
