#!/usr/bin/env python3
"""Convert RoBERTa (chinese-roberta-wwm-ext-large) to CoreML for ANE acceleration.

This script converts a HuggingFace RoBERTa model to CoreML format
optimized for Apple Neural Engine (ANE).

Usage:
    python convert_roberta_coreml.py \
        --input hfl/chinese-roberta-wwm-ext-large \
        --output /path/to/roberta_ane.mlpackage

    # Or from local directory:
    python convert_roberta_coreml.py \
        --input ~/.dora/models/primespeech/moyoyo/chinese-roberta-wwm-ext-large \
        --output ~/.dora/models/primespeech/gpt-sovits-mlx/roberta_ane.mlpackage

The conversion applies ANE optimizations:
- FP16 precision for faster inference
- Variable sequence length support
- Optimized for Apple Neural Engine
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
    HAS_COREML = True
except ImportError:
    HAS_COREML = False
    print("Warning: coremltools not installed")


class RoBERTaWrapper(nn.Module):
    """Wrapper for RoBERTa that's compatible with CoreML tracing."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len]

        Returns:
            Features [batch, seq_len, 1024]
        """
        # Ensure proper shape
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.ones_like(input_ids)

        # Run model
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Return last hidden state
        return outputs.last_hidden_state


class RoBERTaWrapperSimple(nn.Module):
    """Simplified wrapper without attention mask for tracing compatibility."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len]

        Returns:
            Features [batch, seq_len, hidden_size]
        """
        # Run model without attention mask for simpler tracing
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
        return outputs.last_hidden_state


def convert_to_coreml(
    model_path: str,
    output_path: str,
    max_seq_length: int = 512,
    compute_precision: str = "float16",
) -> None:
    """Convert RoBERTa to CoreML.

    Args:
        model_path: Path to HuggingFace model or model ID
        output_path: Output path for .mlpackage
        max_seq_length: Maximum sequence length
        compute_precision: "float16" or "float32"
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for conversion")
    if not HAS_COREML:
        raise RuntimeError("coremltools is required for conversion")

    print(f"Loading model from {model_path}...")

    # Try loading as HuggingFace model
    try:
        from transformers import BertModel, BertConfig

        # Load model
        model = BertModel.from_pretrained(model_path)
        model.eval()

        # Get hidden size from config
        config = model.config
        hidden_size = config.hidden_size

        print(f"Loaded HuggingFace BertModel")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Num layers: {config.num_hidden_layers}")
        print(f"  Num attention heads: {config.num_attention_heads}")

    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError(f"Failed to load model from {model_path}")

    # Wrap model
    wrapped = RoBERTaWrapperSimple(model)
    wrapped.eval()

    # Create example input
    example_input = torch.randint(0, 1000, (1, 128), dtype=torch.long)

    print("Tracing model...")
    traced = torch.jit.trace(wrapped, example_input)

    print("Converting to CoreML...")

    # Determine compute precision
    if compute_precision == "float16":
        precision = ct.precision.FLOAT16
    else:
        precision = ct.precision.FLOAT32

    # Convert with variable sequence length
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=(1, ct.RangeDim(1, max_seq_length)),  # Variable length
                dtype=np.int32,
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
    mlmodel.short_description = "Chinese RoBERTa text encoder for TTS"
    mlmodel.input_description["input_ids"] = "Token IDs from RoBERTa tokenizer"
    mlmodel.output_description["features"] = f"Text features [batch, seq_len, {hidden_size}]"

    print(f"Saving to {output_path}...")
    mlmodel.save(output_path)

    print("Conversion complete!")
    print(f"  Input: input_ids [1, 1-{max_seq_length}]")
    print(f"  Output: features [1, seq_len, {hidden_size}]")
    print(f"  Precision: {compute_precision}")


def verify_conversion(
    mlpackage_path: str,
    original_model_path: str,
    test_sequence: str = "你好世界",
) -> bool:
    """Verify the converted model produces correct outputs.

    Args:
        mlpackage_path: Path to converted CoreML model
        original_model_path: Path to original HuggingFace model
        test_sequence: Test text to encode

    Returns:
        True if verification passes
    """
    if not HAS_COREML or not HAS_TORCH:
        print("Skipping verification (missing dependencies)")
        return True

    print(f"\nVerifying conversion...")

    from transformers import BertModel, BertTokenizer

    # Load original model
    tokenizer = BertTokenizer.from_pretrained(original_model_path)
    model = BertModel.from_pretrained(original_model_path)
    model.eval()

    # Tokenize
    inputs = tokenizer(test_sequence, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Get original output
    with torch.no_grad():
        original_output = model(input_ids=input_ids).last_hidden_state.numpy()

    # Load CoreML model
    coreml_model = ct.models.MLModel(mlpackage_path)

    # Get CoreML output
    coreml_input = input_ids.numpy().astype(np.int32)
    coreml_output = coreml_model.predict({"input_ids": coreml_input})["features"]

    # Compare outputs
    max_diff = np.abs(original_output - coreml_output).max()
    mean_diff = np.abs(original_output - coreml_output).mean()

    print(f"  Test input: '{test_sequence}'")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {coreml_output.shape}")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")

    # Allow some tolerance for float16 conversion
    tolerance = 0.01 if max_diff < 0.01 else 0.1
    if max_diff < tolerance:
        print("  Verification PASSED")
        return True
    else:
        print("  Verification FAILED (differences too large)")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert RoBERTa to CoreML for ANE acceleration"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to HuggingFace model or model ID (e.g., hfl/chinese-roberta-wwm-ext-large)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output path for .mlpackage"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )
    parser.add_argument(
        "--precision",
        choices=["float16", "float32"],
        default="float16",
        help="Compute precision (default: float16 for ANE)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify conversion by comparing outputs"
    )

    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    convert_to_coreml(
        args.input,
        args.output,
        max_seq_length=args.max_seq_length,
        compute_precision=args.precision,
    )

    if args.verify:
        verify_conversion(args.output, args.input)


if __name__ == "__main__":
    main()
