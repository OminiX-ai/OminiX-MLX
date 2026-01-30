"""High-level TTS engine for GPT-SoVITS MLX.

Provides a simple API for text-to-speech synthesis using the hybrid
CoreML + MLX architecture.

Example:
    engine = GPTSoVITSEngine(model_dir="/path/to/models")
    engine.load_voice("Doubao")

    result = engine.synthesize("你好世界")
    # result.audio is numpy array at 32kHz

    # Or with streaming
    for chunk in engine.synthesize_streaming("你好世界"):
        play_audio(chunk)
"""

from typing import Optional, Iterator, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import time
import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from python.models.config import GPTConfig, VocoderConfig, SynthesisConfig, ModelPaths
from python.models.gpt import GPTSoVITS, load_gpt_model
from python.models.vocoder import SoVITSVocoder, load_vocoder
from python.generate import generate_semantic_tokens, GenerationConfig
from python.encoders import load_audio_encoder, load_text_encoder
from python.text.preprocessor import TextPreprocessor, PreprocessorOutput


@dataclass
class SynthesisResult:
    """Result of TTS synthesis."""

    audio: np.ndarray  # Float32 waveform
    sample_rate: int  # 32000
    duration: float  # Seconds

    # Debug info
    semantic_tokens: Optional[np.ndarray] = None
    phonemes: Optional[list] = None  # Phoneme sequence
    timing: Optional[Dict[str, float]] = None


class GPTSoVITSEngine:
    """High-level TTS engine using hybrid CoreML + MLX.

    This is the main entry point for GPT-SoVITS synthesis on Apple Silicon.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        use_ane: bool = True,
        use_compile: bool = True,
        device: str = "gpu",
    ):
        """Initialize the TTS engine.

        Args:
            model_dir: Path to model directory
            use_ane: Use ANE for encoders via CoreML
            use_compile: Use mx.compile for optimization
            device: Device for MLX ("gpu" or "cpu")
        """
        if not HAS_MLX:
            raise ImportError("mlx is required for GPTSoVITSEngine")

        self.model_dir = Path(model_dir) if model_dir else None
        self.use_ane = use_ane
        self.use_compile = use_compile
        self.device = device

        # Models (loaded lazily)
        self._audio_encoder = None
        self._text_encoder = None
        self._gpt_model = None
        self._vocoder = None

        # Text preprocessor
        self._text_preprocessor = TextPreprocessor()

        # Current voice
        self._current_voice: Optional[str] = None
        self._voice_config: Optional[Dict[str, Any]] = None

        # Cached features
        self._cached_bert_features: Optional[mx.array] = None  # BERT features for GPT
        self._cached_audio_features: Optional[mx.array] = None  # CNHubert features for vocoder

    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._gpt_model is not None

    def load_voice(self, voice_name: str) -> None:
        """Load a voice model.

        Args:
            voice_name: Name of the voice to load
        """
        if self.model_dir is None:
            raise ValueError("model_dir must be set to load voices")

        paths = ModelPaths.from_model_dir(self.model_dir, voice_name)

        print(f"Loading voice: {voice_name}")
        start = time.perf_counter()

        # Load audio encoder (CoreML/ANE)
        if paths.cnhubert_path and Path(paths.cnhubert_path).exists():
            compute_units = "ALL" if self.use_ane else "CPU_AND_GPU"
            try:
                self._audio_encoder = load_audio_encoder(
                    paths.cnhubert_path,
                    use_coreml=True,
                    compute_units=compute_units,
                )
                print(f"  Audio encoder loaded: {paths.cnhubert_path}")
            except (ImportError, NotImplementedError) as e:
                # Fall back to dummy encoder if CoreML not available
                self._audio_encoder = load_audio_encoder(None)
                print(f"  Using dummy audio encoder (CoreML not available: {e})")
        else:
            # Use dummy encoder for testing
            self._audio_encoder = load_audio_encoder(None)
            print("  Using dummy audio encoder (no model found)")

        # Load text encoder (optional)
        if paths.roberta_path and Path(paths.roberta_path).exists():
            try:
                self._text_encoder = load_text_encoder(
                    paths.roberta_path,
                    use_coreml=True,
                    compute_units="ALL" if self.use_ane else "CPU_AND_GPU",
                )
                print(f"  Text encoder loaded: {paths.roberta_path}")
            except (ImportError, NotImplementedError) as e:
                # Fall back to dummy encoder if CoreML not available
                from python.encoders import DummyTextEncoder
                self._text_encoder = DummyTextEncoder()
                print(f"  Using dummy text encoder (CoreML not available: {e})")
        else:
            from python.encoders import DummyTextEncoder
            self._text_encoder = DummyTextEncoder()
            print("  Using dummy text encoder (no model found)")

        # Load GPT model (MLX)
        if paths.gpt_weights_path and Path(paths.gpt_weights_path).exists():
            config_path = paths.gpt_config_path if Path(paths.gpt_config_path).exists() else None
            self._gpt_model = load_gpt_model(
                paths.gpt_weights_path,
                config_path=config_path,
            )
            print(f"  GPT model loaded: {paths.gpt_weights_path}")

            # Optionally compile the model
            if self.use_compile:
                # TODO: Apply mx.compile to forward pass
                pass
        else:
            raise FileNotFoundError(f"GPT weights not found: {paths.gpt_weights_path}")

        # Load vocoder (MLX)
        if paths.vocoder_weights_path and Path(paths.vocoder_weights_path).exists():
            vocoder_config = None
            if paths.vocoder_config_path and Path(paths.vocoder_config_path).exists():
                vocoder_config = VocoderConfig.from_json(paths.vocoder_config_path)
            self._vocoder = load_vocoder(
                paths.vocoder_weights_path,
                config=vocoder_config,
            )
            print(f"  Vocoder loaded: {paths.vocoder_weights_path}")
        else:
            # Create default vocoder for testing
            self._vocoder = SoVITSVocoder(VocoderConfig())
            print("  Using default vocoder (no weights found)")

        # Cache reference audio features
        if paths.reference_audio_path and Path(paths.reference_audio_path).exists():
            self._cache_reference_audio(paths.reference_audio_path)

        self._current_voice = voice_name
        elapsed = time.perf_counter() - start
        print(f"Voice loaded in {elapsed:.2f}s")

    def _cache_reference_audio(self, audio_path: str) -> None:
        """Cache reference audio features for the current voice."""
        import wave

        # Load audio file
        with wave.open(audio_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            audio_bytes = wf.readframes(n_frames)

        # Convert to float32
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            # Simple resampling (should use proper resampler in production)
            ratio = 16000 / sample_rate
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length).astype(int)
            audio = audio[indices]

        # Extract features
        features = self._audio_encoder.encode_to_mlx(audio, sample_rate=16000)
        self._cached_audio_features = features
        print(f"  Reference audio cached: {features.shape}")

    def synthesize(
        self,
        text: str,
        config: Optional[SynthesisConfig] = None,
    ) -> SynthesisResult:
        """Synthesize speech from text.

        Args:
            text: Input text to synthesize
            config: Synthesis configuration

        Returns:
            SynthesisResult with audio waveform
        """
        if not self.is_loaded:
            raise RuntimeError("No voice loaded. Call load_voice() first.")

        if config is None:
            config = SynthesisConfig()

        timing = {}
        start_total = time.perf_counter()

        # Step 1: Text processing (phonemization)
        start = time.perf_counter()
        phoneme_ids, preproc_output = self._text_to_phonemes(text, config.language)
        timing["text_processing"] = time.perf_counter() - start

        # Step 2: Extract BERT features from input text
        start = time.perf_counter()
        bert_features = self._extract_bert_features(text)
        timing["bert_encoding"] = time.perf_counter() - start

        # Step 3: Generate semantic tokens (GPT)
        start = time.perf_counter()
        gen_config = GenerationConfig(
            max_tokens=config.max_semantic_tokens,
            min_tokens=config.min_semantic_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
        )

        output = generate_semantic_tokens(
            self._gpt_model,
            phoneme_ids,
            bert_features,
            config=gen_config,
        )
        semantic_tokens = output.tokens
        timing["gpt_generation"] = time.perf_counter() - start

        # Step 4: Vocoder synthesis
        # Use cached CNHubert audio features for vocoder conditioning
        start = time.perf_counter()
        audio = self._vocode(semantic_tokens, self._cached_audio_features, config)
        timing["vocoder"] = time.perf_counter() - start

        timing["total"] = time.perf_counter() - start_total

        return SynthesisResult(
            audio=np.array(audio),
            sample_rate=config.sample_rate,
            duration=len(audio) / config.sample_rate,
            semantic_tokens=np.array(semantic_tokens),
            phonemes=preproc_output.phonemes,
            timing=timing,
        )

    def synthesize_streaming(
        self,
        text: str,
        config: Optional[SynthesisConfig] = None,
        chunk_samples: int = 4096,
    ) -> Iterator[np.ndarray]:
        """Synthesize with streaming output.

        Yields audio chunks as they become available.

        Args:
            text: Input text
            config: Synthesis configuration
            chunk_samples: Samples per chunk

        Yields:
            Audio chunks as numpy arrays
        """
        # For now, fall back to non-streaming
        # TODO: Implement true streaming with incremental vocoding
        result = self.synthesize(text, config)

        # Yield in chunks
        audio = result.audio
        for i in range(0, len(audio), chunk_samples):
            yield audio[i:i + chunk_samples]

    def _text_to_phonemes(self, text: str, language: str = "zh") -> Tuple[mx.array, PreprocessorOutput]:
        """Convert text to phoneme IDs.

        Args:
            text: Input text
            language: Language code

        Returns:
            Tuple of (phoneme_ids as MLX array [1, seq_len], preprocessor output)
        """
        # Use the text preprocessor for proper phonemization
        output = self._text_preprocessor.preprocess(text, language=language)
        phoneme_ids = mx.array([output.phoneme_ids], dtype=mx.int32)
        return phoneme_ids, output

    def _extract_bert_features(self, text: str) -> mx.array:
        """Extract BERT features from text using RoBERTa encoder.

        Args:
            text: Input text

        Returns:
            BERT features as MLX array [1, seq_len, 1024]
        """
        if self._text_encoder is not None:
            # Use actual RoBERTa encoder
            features = self._text_encoder.encode_to_mlx(text)
            return features
        else:
            # Fallback: create dummy features matching expected shape
            # Use a reasonable sequence length based on text length
            seq_len = max(1, len(text) // 2)  # Rough estimate
            return mx.zeros((1, seq_len, 1024), dtype=mx.float32)

    def _vocode(
        self,
        semantic_tokens: mx.array,
        audio_features: mx.array,
        config: SynthesisConfig,
    ) -> mx.array:
        """Convert semantic tokens to audio waveform.

        Args:
            semantic_tokens: Semantic token IDs [1, seq_len]
            audio_features: Reference audio features [1, time, 768]
            config: Synthesis config

        Returns:
            Audio waveform [samples]
        """
        if self._vocoder is None:
            # Fallback: generate silence based on token count
            num_tokens = semantic_tokens.shape[1] if semantic_tokens.ndim > 1 else len(semantic_tokens)
            duration_samples = int(num_tokens * 0.02 * config.sample_rate)
            return mx.zeros((duration_samples,), dtype=mx.float32)

        # Use the vocoder to synthesize audio
        # semantic_tokens: [1, seq_len]
        # audio_features: [1, time, 768]
        audio = self._vocoder(
            semantic_tokens,
            audio_features=audio_features,
            speed_factor=config.speed_factor,
        )
        mx.eval(audio)

        # audio is [batch, samples, 1], squeeze to [samples]
        audio = audio.squeeze()

        return audio

    def warmup(self) -> None:
        """Warmup models for optimal first-inference latency."""
        if not self.is_loaded:
            return

        print("Warming up models...")

        # Run dummy inference
        dummy_phonemes = mx.array([[0, 1, 2, 3, 4]], dtype=mx.int32)
        dummy_bert = mx.zeros((1, 10, 1024), dtype=mx.float32)  # BERT features

        # Warmup GPT
        caches = self._gpt_model.create_caches()
        _ = self._gpt_model(
            dummy_phonemes,
            mx.array([[0]], dtype=mx.int32),
            dummy_bert,
            cache=caches,
        )
        mx.eval(_)

        print("Warmup complete")

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "voice": self._current_voice,
            "is_loaded": self.is_loaded,
            "use_ane": self.use_ane,
            "use_compile": self.use_compile,
            "device": self.device,
            "has_audio_encoder": self._audio_encoder is not None,
            "has_text_encoder": self._text_encoder is not None,
            "has_vocoder": self._vocoder is not None,
            "has_cached_audio": self._cached_audio_features is not None,  # For vocoder
        }


# High-level API functions (matching dora-primespeech TTS.run() interface)

def synthesize(
    text: str,
    model_dir: str,
    voice_name: str = "Doubao",
    language: str = "zh",
    speed: float = 1.0,
    output_path: Optional[str] = None,
    use_ane: bool = True,
) -> SynthesisResult:
    """High-level function to synthesize speech from text.

    Similar to dora-primespeech TTS.run() interface.

    Args:
        text: Text to synthesize
        model_dir: Path to model directory
        voice_name: Voice to use
        language: Language code ("zh", "en", "auto")
        speed: Speed factor (1.0 = normal)
        output_path: Optional path to save WAV file
        use_ane: Use Apple Neural Engine for encoders

    Returns:
        SynthesisResult with audio and metadata
    """
    # Create engine
    engine = GPTSoVITSEngine(
        model_dir=model_dir,
        use_ane=use_ane,
    )

    # Load voice
    engine.load_voice(voice_name)

    # Configure synthesis
    config = SynthesisConfig(
        voice_name=voice_name,
        language=language,
        speed_factor=speed,
    )

    # Synthesize
    result = engine.synthesize(text, config=config)

    # Save if requested
    if output_path:
        save_wav(result.audio, result.sample_rate, output_path)

    return result


def save_wav(audio: np.ndarray, sample_rate: int, path: str) -> None:
    """Save audio to WAV file.

    Args:
        audio: Float32 audio waveform
        sample_rate: Sample rate
        path: Output path
    """
    import wave

    # Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    print(f"Saved audio to {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPT-SoVITS MLX TTS Engine")
    parser.add_argument("text", nargs="?", default="你好，世界！", help="Text to synthesize")
    parser.add_argument("--model-dir", "-m", default="~/.dora/models/primespeech/gpt-sovits-mlx", help="Model directory")
    parser.add_argument("--voice", "-v", default="Doubao", help="Voice name")
    parser.add_argument("--language", "-l", default="zh", help="Language (zh, en, auto)")
    parser.add_argument("--speed", "-s", type=float, default=1.0, help="Speed factor")
    parser.add_argument("--output", "-o", help="Output WAV file")
    parser.add_argument("--no-ane", action="store_true", help="Disable ANE")

    args = parser.parse_args()

    print(f"GPT-SoVITS MLX TTS Engine")
    print(f"  Text: {args.text}")
    print(f"  Voice: {args.voice}")
    print(f"  Language: {args.language}")
    print()

    try:
        result = synthesize(
            args.text,
            model_dir=args.model_dir,
            voice_name=args.voice,
            language=args.language,
            speed=args.speed,
            output_path=args.output,
            use_ane=not args.no_ane,
        )

        print(f"\nSynthesis complete:")
        print(f"  Duration: {result.duration:.2f}s")
        print(f"  Sample rate: {result.sample_rate}Hz")
        print(f"  Phonemes: {result.phonemes}")
        print(f"  Semantic tokens: {len(result.semantic_tokens)}")
        if result.timing:
            print(f"  Timing:")
            for k, v in result.timing.items():
                print(f"    {k}: {v*1000:.1f}ms")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the model directory contains the required files.")
        print("Expected structure:")
        print("  {model_dir}/encoders/cnhubert_ane.mlpackage")
        print("  {model_dir}/voices/{voice}/gpt.safetensors")
        print("  {model_dir}/voices/{voice}/reference.wav")
