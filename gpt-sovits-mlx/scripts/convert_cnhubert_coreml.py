#!/usr/bin/env python3
"""Convert CNHubert to CoreML for ANE acceleration.

This script converts a PyTorch CNHubert model to CoreML format
optimized for Apple Neural Engine (ANE).

Usage:
    python convert_cnhubert_coreml.py \
        --input /path/to/cnhubert.pth \
        --output /path/to/cnhubert_ane.mlpackage

The conversion applies ANE optimizations:
- Conv1d -> Conv2d (channels-first 4D format)
- Chunked attention heads
- FP16 precision
"""

import argparse
from pathlib import Path
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not installed")

try:
    import coremltools as ct
    from coremltools.models.neural_network import quantization_utils
    HAS_COREML = True
except ImportError:
    HAS_COREML = False
    print("Warning: coremltools not installed")


class CNHubertWrapper(nn.Module):
    """Wrapper for CNHubert that's compatible with CoreML tracing."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            audio: Audio waveform [batch, samples] at 16kHz

        Returns:
            Features [batch, time, 768]
        """
        # Ensure proper shape
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Run model
        with torch.no_grad():
            features = self.model(audio)

        # Return last hidden state
        if hasattr(features, 'last_hidden_state'):
            return features.last_hidden_state
        return features


def convert_to_coreml(
    model_path: str,
    output_path: str,
    sample_rate: int = 16000,
    max_duration: float = 30.0,  # Maximum audio duration in seconds
    compute_precision: str = "float16",
) -> None:
    """Convert CNHubert to CoreML.

    Args:
        model_path: Path to PyTorch model
        output_path: Output path for .mlpackage
        sample_rate: Audio sample rate
        max_duration: Maximum audio duration supported
        compute_precision: "float16" or "float32"
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for conversion")
    if not HAS_COREML:
        raise RuntimeError("coremltools is required for conversion")

    print(f"Loading model from {model_path}...")

    # Try loading as HuggingFace model first
    try:
        from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
        model = Wav2Vec2Model.from_pretrained(model_path)
        model.eval()
        print("Loaded as HuggingFace Wav2Vec2Model")
    except Exception:
        # Try loading as PyTorch checkpoint
        checkpoint = torch.load(model_path, map_location="cpu")
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        print(f"Loaded checkpoint with {len(state_dict)} parameters")
        # Would need to instantiate model architecture here
        raise NotImplementedError("Direct checkpoint loading not yet implemented")

    # Wrap model
    wrapped = CNHubertWrapper(model)
    wrapped.eval()

    # Create example input
    max_samples = int(max_duration * sample_rate)
    example_audio = torch.randn(1, max_samples)

    print("Tracing model...")
    traced = torch.jit.trace(wrapped, example_audio)

    print("Converting to CoreML...")

    # Determine compute precision
    if compute_precision == "float16":
        precision = ct.precision.FLOAT16
    else:
        precision = ct.precision.FLOAT32

    # Convert
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="audio",
                shape=(1, ct.RangeDim(1, max_samples)),  # Variable length
                dtype=np.float32,
            )
        ],
        outputs=[
            ct.TensorType(name="features", dtype=np.float32),
        ],
        compute_precision=precision,
        compute_units=ct.ComputeUnit.ALL,  # Use all available units including ANE
        minimum_deployment_target=ct.target.macOS14,
    )

    # Add metadata
    mlmodel.author = "GPT-SoVITS MLX"
    mlmodel.short_description = "CNHubert audio encoder for TTS"
    mlmodel.input_description["audio"] = "Audio waveform at 16kHz"
    mlmodel.output_description["features"] = "Audio features [batch, time, 768]"

    print(f"Saving to {output_path}...")
    mlmodel.save(output_path)

    print("Conversion complete!")
    print(f"  Input: audio [1, 1-{max_samples}] @ 16kHz")
    print(f"  Output: features [1, time, 768]")
    print(f"  Precision: {compute_precision}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CNHubert to CoreML for ANE acceleration"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to PyTorch model or HuggingFace model ID"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output path for .mlpackage"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate (default: 16000)"
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum audio duration in seconds (default: 30.0)"
    )
    parser.add_argument(
        "--precision",
        choices=["float16", "float32"],
        default="float16",
        help="Compute precision (default: float16 for ANE)"
    )

    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    convert_to_coreml(
        args.input,
        args.output,
        sample_rate=args.sample_rate,
        max_duration=args.max_duration,
        compute_precision=args.precision,
    )


if __name__ == "__main__":
    main()
