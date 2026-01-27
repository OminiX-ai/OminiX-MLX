"""CoreML encoder wrappers for GPT-SoVITS.

Provides wrappers for:
- CNHubert: Audio feature extraction (ANE accelerated)
- RoBERTa: Text feature extraction (ANE accelerated)

These models run on the Apple Neural Engine (ANE) for maximum efficiency,
with zero-copy transfer to MLX via unified memory.
"""

from typing import Optional, Protocol
from pathlib import Path
import numpy as np

try:
    import coremltools as ct
    HAS_COREML = True
except ImportError:
    HAS_COREML = False

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


class AudioEncoder(Protocol):
    """Protocol for audio encoders."""

    def encode(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Encode audio to features."""
        ...


class TextEncoder(Protocol):
    """Protocol for text encoders."""

    def encode(self, token_ids: np.ndarray) -> np.ndarray:
        """Encode token IDs to features."""
        ...


class CoreMLCNHubert:
    """CNHubert audio encoder using CoreML (ANE accelerated).

    Extracts 768-dimensional features from audio at ~50Hz.
    Optimized to run on Apple Neural Engine for ~5ms latency.
    """

    def __init__(
        self,
        model_path: str | Path,
        compute_units: str = "ALL",  # "ALL", "CPU_AND_GPU", "CPU_AND_NE", "CPU_ONLY"
    ):
        """Initialize CNHubert encoder.

        Args:
            model_path: Path to .mlpackage or .mlmodelc
            compute_units: CoreML compute units to use
        """
        if not HAS_COREML:
            raise ImportError("coremltools is required for CoreML encoders")

        self.model_path = Path(model_path)

        # Map compute units string to CoreML enum
        compute_map = {
            "ALL": ct.ComputeUnit.ALL,
            "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
            "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
            "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        }
        units = compute_map.get(compute_units, ct.ComputeUnit.ALL)

        # Load model
        self.model = ct.models.MLModel(
            str(self.model_path),
            compute_units=units,
        )

        # Get input/output names from spec
        spec = self.model.get_spec()
        self.input_name = spec.description.input[0].name
        self.output_name = spec.description.output[0].name

    def encode(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """Encode audio waveform to features.

        Args:
            audio: Audio waveform [samples] or [1, samples] at 16kHz
            sample_rate: Sample rate (must be 16000)

        Returns:
            Features array [1, 768, time] or [1, time, 768]
        """
        if sample_rate != 16000:
            raise ValueError(f"CNHubert expects 16kHz audio, got {sample_rate}Hz")

        # Ensure correct shape
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Run inference
        output = self.model.predict({self.input_name: audio})
        features = output[self.output_name]

        return features

    def encode_to_mlx(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> "mx.array":
        """Encode audio and return MLX array (zero-copy if possible).

        Args:
            audio: Audio waveform
            sample_rate: Sample rate

        Returns:
            MLX array of features
        """
        if not HAS_MLX:
            raise ImportError("mlx is required for MLX output")

        features = self.encode(audio, sample_rate)

        # Convert to MLX array
        # On unified memory, this should be zero-copy
        return mx.array(features)


class CoreMLRoBERTa:
    """RoBERTa text encoder using CoreML (ANE accelerated).

    Extracts 1024-dimensional features from token sequences.
    Optimized for Apple Neural Engine.
    """

    def __init__(
        self,
        model_path: str | Path,
        compute_units: str = "ALL",
        max_length: int = 512,
    ):
        """Initialize RoBERTa encoder.

        Args:
            model_path: Path to .mlpackage or .mlmodelc
            compute_units: CoreML compute units
            max_length: Maximum sequence length
        """
        if not HAS_COREML:
            raise ImportError("coremltools is required for CoreML encoders")

        self.model_path = Path(model_path)
        self.max_length = max_length

        compute_map = {
            "ALL": ct.ComputeUnit.ALL,
            "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
            "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
            "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        }
        units = compute_map.get(compute_units, ct.ComputeUnit.ALL)

        self.model = ct.models.MLModel(
            str(self.model_path),
            compute_units=units,
        )

        spec = self.model.get_spec()
        self.input_name = spec.description.input[0].name
        self.output_name = spec.description.output[0].name

        # Try to load tokenizer for text encoding
        self._tokenizer = None
        try:
            from transformers import BertTokenizer
            # Try to load the tokenizer from the model directory or a default
            model_dir = self.model_path.parent
            tokenizer_path = model_dir / "tokenizer"
            if tokenizer_path.exists():
                self._tokenizer = BertTokenizer.from_pretrained(str(tokenizer_path))
            else:
                # Try default Chinese BERT tokenizer
                self._tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
        except Exception:
            pass  # Tokenizer not available, will require token IDs

    def encode(self, token_ids: np.ndarray) -> np.ndarray:
        """Encode token IDs to features.

        Args:
            token_ids: Token IDs [batch, seq_len] or [seq_len]

        Returns:
            Features array [batch, seq_len, 1024]
        """
        if token_ids.ndim == 1:
            token_ids = token_ids.reshape(1, -1)

        # Ensure int32
        if token_ids.dtype != np.int32:
            token_ids = token_ids.astype(np.int32)

        # Truncate if needed
        if token_ids.shape[1] > self.max_length:
            token_ids = token_ids[:, :self.max_length]

        output = self.model.predict({self.input_name: token_ids})
        features = output[self.output_name]

        return features

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text string directly to features.

        Args:
            text: Input text string

        Returns:
            Features array [1, seq_len, 1024]
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not available. Use encode() with token IDs.")

        # Tokenize
        encoded = self._tokenizer(
            text,
            return_tensors="np",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        token_ids = encoded["input_ids"]

        return self.encode(token_ids)

    def encode_to_mlx(self, text_or_ids) -> "mx.array":
        """Encode text or tokens and return MLX array.

        Args:
            text_or_ids: Either text string or token IDs numpy array

        Returns:
            MLX array of features [1, seq_len, 1024]
        """
        if not HAS_MLX:
            raise ImportError("mlx is required for MLX output")

        if isinstance(text_or_ids, str):
            features = self.encode_text(text_or_ids)
        else:
            features = self.encode(text_or_ids)

        return mx.array(features)


class MLXCNHubert:
    """CNHubert encoder implemented in pure MLX (GPU).

    Fallback when CoreML is not available or ANE is not desired.
    Uses MLX for GPU acceleration via Metal.
    """

    def __init__(self, weights_path: str | Path):
        """Initialize MLX-based CNHubert.

        Args:
            weights_path: Path to safetensors weights
        """
        if not HAS_MLX:
            raise ImportError("mlx is required for MLX encoders")

        # TODO: Implement full CNHubert in MLX
        # This would port the Whisper encoder architecture
        raise NotImplementedError(
            "MLX CNHubert not yet implemented. Use CoreML version."
        )

    def encode(self, audio: np.ndarray, sample_rate: int = 16000) -> "mx.array":
        raise NotImplementedError


class DummyAudioEncoder:
    """Dummy encoder for testing without models."""

    def __init__(self, feature_dim: int = 768, feature_rate: float = 50.0):
        self.feature_dim = feature_dim
        self.feature_rate = feature_rate  # Features per second

    def encode(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Generate random features for testing."""
        duration = len(audio) / sample_rate
        num_frames = int(duration * self.feature_rate)
        return np.random.randn(1, num_frames, self.feature_dim).astype(np.float32)

    def encode_to_mlx(self, audio: np.ndarray, sample_rate: int = 16000) -> "mx.array":
        if not HAS_MLX:
            raise ImportError("mlx is required")
        return mx.array(self.encode(audio, sample_rate))


class DummyTextEncoder:
    """Dummy text encoder for testing."""

    def __init__(self, feature_dim: int = 1024):
        self.feature_dim = feature_dim

    def encode(self, token_ids: np.ndarray) -> np.ndarray:
        """Generate random features for testing."""
        if token_ids.ndim == 1:
            token_ids = token_ids.reshape(1, -1)
        seq_len = token_ids.shape[1]
        return np.random.randn(1, seq_len, self.feature_dim).astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        """Generate features based on text length."""
        seq_len = max(1, len(text))  # Rough approximation
        return np.random.randn(1, seq_len, self.feature_dim).astype(np.float32)

    def encode_to_mlx(self, text_or_ids) -> "mx.array":
        """Encode text or tokens and return MLX array."""
        if not HAS_MLX:
            raise ImportError("mlx is required")
        if isinstance(text_or_ids, str):
            return mx.array(self.encode_text(text_or_ids))
        return mx.array(self.encode(text_or_ids))


def load_audio_encoder(
    model_path: Optional[str] = None,
    use_coreml: bool = True,
    compute_units: str = "ALL",
) -> AudioEncoder:
    """Load audio encoder with automatic backend selection.

    Args:
        model_path: Path to model (None for dummy)
        use_coreml: Whether to use CoreML (ANE)
        compute_units: CoreML compute units

    Returns:
        Audio encoder instance
    """
    if model_path is None:
        return DummyAudioEncoder()

    if use_coreml and HAS_COREML:
        return CoreMLCNHubert(model_path, compute_units=compute_units)
    else:
        return MLXCNHubert(model_path)


def load_text_encoder(
    model_path: Optional[str] = None,
    use_coreml: bool = True,
    compute_units: str = "ALL",
) -> TextEncoder:
    """Load text encoder with automatic backend selection.

    Args:
        model_path: Path to model (None for dummy)
        use_coreml: Whether to use CoreML
        compute_units: CoreML compute units

    Returns:
        Text encoder instance
    """
    if model_path is None:
        return DummyTextEncoder()

    if use_coreml and HAS_COREML:
        return CoreMLRoBERTa(model_path, compute_units=compute_units)
    else:
        raise NotImplementedError("MLX RoBERTa not implemented")
