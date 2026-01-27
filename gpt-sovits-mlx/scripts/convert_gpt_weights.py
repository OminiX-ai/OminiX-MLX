#!/usr/bin/env python3
"""Convert GPT-SoVITS PyTorch weights to MLX safetensors format.

This script converts the GPT model weights from the original PyTorch
checkpoint format to safetensors for use with MLX.

Usage:
    python convert_gpt_weights.py --input /path/to/gpt.ckpt --output /path/to/gpt.safetensors

The script handles:
- Weight name mapping between PyTorch and MLX conventions
- Splitting combined QKV projections into separate Q, K, V
- Config extraction and saving

Original GPT-SoVITS weight format:
- model.bert_proj.weight/bias
- model.ar_text_embedding.word_embeddings.weight
- model.ar_audio_embedding.word_embeddings.weight
- model.ar_text_position.alpha
- model.ar_audio_position.alpha
- model.h.layers.{i}.self_attn.in_proj_weight/bias (combined QKV)
- model.h.layers.{i}.self_attn.out_proj.weight/bias
- model.h.layers.{i}.linear1.weight/bias (FFN)
- model.h.layers.{i}.linear2.weight/bias
- model.h.layers.{i}.norm1.weight/bias (LayerNorm)
- model.h.layers.{i}.norm2.weight/bias
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

# Try to import torch, but make it optional for environments without it
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: torch not installed. Some features may not work.")

from safetensors.numpy import save_file


def split_qkv(combined_weight: np.ndarray, combined_bias: Optional[np.ndarray] = None
              ) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]]]:
    """Split combined QKV projection into separate Q, K, V.

    Args:
        combined_weight: Combined [3*hidden, hidden] weight matrix
        combined_bias: Combined [3*hidden] bias vector

    Returns:
        Tuple of (weight_dict, bias_dict) with q, k, v keys
    """
    total_dim = combined_weight.shape[0]
    hidden_dim = total_dim // 3

    weights = {
        "q": combined_weight[:hidden_dim, :],
        "k": combined_weight[hidden_dim:2*hidden_dim, :],
        "v": combined_weight[2*hidden_dim:, :],
    }

    biases = None
    if combined_bias is not None:
        biases = {
            "q": combined_bias[:hidden_dim],
            "k": combined_bias[hidden_dim:2*hidden_dim],
            "v": combined_bias[2*hidden_dim:],
        }

    return weights, biases


def convert_gpt_sovits_weights(
    state_dict: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """Convert GPT-SoVITS weights to MLX format.

    Args:
        state_dict: Original PyTorch state dict
        config: Model configuration

    Returns:
        Converted weights dict for MLX
    """
    converted = {}
    num_layers = config.get("num_layers", 24)

    for name, tensor in state_dict.items():
        # Convert tensor to numpy
        if hasattr(tensor, 'numpy'):
            np_tensor = tensor.numpy()
        elif hasattr(tensor, 'cpu'):
            np_tensor = tensor.cpu().numpy()
        else:
            np_tensor = np.array(tensor)

        # Ensure float32
        if np_tensor.dtype == np.float64:
            np_tensor = np_tensor.astype(np.float32)

        # Map weight names
        if name == "model.bert_proj.weight":
            # BERT/HuBERT feature projection
            converted["audio_proj.weight"] = np_tensor

        elif name == "model.bert_proj.bias":
            converted["audio_proj.bias"] = np_tensor

        elif name == "model.ar_text_embedding.word_embeddings.weight":
            # Phoneme/text embeddings
            converted["phoneme_embed.weight"] = np_tensor

        elif name == "model.ar_audio_embedding.word_embeddings.weight":
            # Semantic/audio token embeddings
            converted["semantic_embed.weight"] = np_tensor

        elif name == "model.ar_text_position.alpha":
            # Positional encoding scale for text
            converted["text_pos_alpha"] = np_tensor

        elif name == "model.ar_audio_position.alpha":
            # Positional encoding scale for audio
            converted["audio_pos_alpha"] = np_tensor

        elif "model.h.layers." in name:
            # Parse layer index
            parts = name.split(".")
            layer_idx = int(parts[3])

            if layer_idx >= num_layers:
                continue

            suffix = ".".join(parts[4:])

            if suffix == "self_attn.in_proj_weight":
                # Split combined QKV
                weights, _ = split_qkv(np_tensor)
                converted[f"layers.{layer_idx}.self_attn.q_proj.weight"] = weights["q"]
                converted[f"layers.{layer_idx}.self_attn.k_proj.weight"] = weights["k"]
                converted[f"layers.{layer_idx}.self_attn.v_proj.weight"] = weights["v"]

            elif suffix == "self_attn.in_proj_bias":
                # Split combined QKV bias
                _, biases = split_qkv(
                    np.zeros((np_tensor.shape[0], 1)),  # Dummy for shape
                    np_tensor
                )
                converted[f"layers.{layer_idx}.self_attn.q_proj.bias"] = biases["q"]
                converted[f"layers.{layer_idx}.self_attn.k_proj.bias"] = biases["k"]
                converted[f"layers.{layer_idx}.self_attn.v_proj.bias"] = biases["v"]

            elif suffix == "self_attn.out_proj.weight":
                converted[f"layers.{layer_idx}.self_attn.o_proj.weight"] = np_tensor

            elif suffix == "self_attn.out_proj.bias":
                converted[f"layers.{layer_idx}.self_attn.o_proj.bias"] = np_tensor

            elif suffix == "linear1.weight":
                # FFN first layer -> gate_proj (for SwiGLU, but original uses GELU)
                # We'll store as gate_proj and handle in model
                converted[f"layers.{layer_idx}.mlp.gate_proj.weight"] = np_tensor

            elif suffix == "linear1.bias":
                converted[f"layers.{layer_idx}.mlp.gate_proj.bias"] = np_tensor

            elif suffix == "linear2.weight":
                # FFN second layer -> down_proj
                converted[f"layers.{layer_idx}.mlp.down_proj.weight"] = np_tensor

            elif suffix == "linear2.bias":
                converted[f"layers.{layer_idx}.mlp.down_proj.bias"] = np_tensor

            elif suffix == "norm1.weight":
                converted[f"layers.{layer_idx}.input_layernorm.weight"] = np_tensor

            elif suffix == "norm1.bias":
                converted[f"layers.{layer_idx}.input_layernorm.bias"] = np_tensor

            elif suffix == "norm2.weight":
                converted[f"layers.{layer_idx}.post_attention_layernorm.weight"] = np_tensor

            elif suffix == "norm2.bias":
                converted[f"layers.{layer_idx}.post_attention_layernorm.bias"] = np_tensor

        elif name == "model.ar_predict_layer.weight":
            # Output prediction layer
            converted["lm_head.weight"] = np_tensor

        elif name == "model.ar_predict_layer.bias":
            converted["lm_head.bias"] = np_tensor

    return converted


def extract_config_from_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model config from GPT-SoVITS checkpoint.

    Args:
        checkpoint: Loaded checkpoint dict

    Returns:
        Model configuration
    """
    # GPT-SoVITS stores config in 'config' key
    if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
        orig_config = checkpoint['config']
        model_config = orig_config.get('model', {})

        return {
            "hidden_size": model_config.get("embedding_dim", 512),
            "num_layers": model_config.get("n_layer", 24),
            "num_heads": model_config.get("head", 16),
            "intermediate_size": model_config.get("linear_units", 2048),
            "phoneme_vocab_size": model_config.get("phoneme_vocab_size", 732),
            "semantic_vocab_size": model_config.get("vocab_size", 1025),
            "audio_feature_dim": 768,  # CNHubert output
            "text_feature_dim": 1024,  # BERT output
            "dropout": model_config.get("dropout", 0.0),
            "max_seq_len": 1024,
            "eos_token": model_config.get("EOS", 1024),
            # Original uses LayerNorm, not RMSNorm
            "use_layernorm": True,
            # Original uses GELU, not SwiGLU
            "use_gelu": True,
        }

    # Fallback: analyze weight shapes
    return {
        "hidden_size": 512,
        "num_layers": 24,
        "num_heads": 16,
        "intermediate_size": 2048,
        "phoneme_vocab_size": 732,
        "semantic_vocab_size": 1025,
        "audio_feature_dim": 768,
        "text_feature_dim": 1024,
        "dropout": 0.0,
        "max_seq_len": 1024,
        "eos_token": 1024,
        "use_layernorm": True,
        "use_gelu": True,
    }


def convert_checkpoint(
    input_path: str,
    output_path: str,
    config_output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert PyTorch checkpoint to MLX safetensors.

    Args:
        input_path: Path to PyTorch checkpoint
        output_path: Path for output safetensors file
        config_output_path: Optional path for config JSON

    Returns:
        Extracted configuration
    """
    if not HAS_TORCH:
        raise RuntimeError("torch is required for checkpoint conversion")

    print(f"Loading checkpoint from {input_path}...")
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)

    # Extract config
    config = extract_config_from_checkpoint(checkpoint)
    print(f"Extracted config:")
    print(json.dumps(config, indent=2))

    # Get state dict
    if "weight" in checkpoint:
        state_dict = checkpoint["weight"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    print(f"\nOriginal checkpoint has {len(state_dict)} parameters")

    # Convert weights
    converted = convert_gpt_sovits_weights(state_dict, config)

    print(f"\nConverted {len(converted)} weights:")
    for name, tensor in list(converted.items())[:10]:
        print(f"  {name}: {tensor.shape} {tensor.dtype}")
    if len(converted) > 10:
        print(f"  ... and {len(converted) - 10} more")

    # Save safetensors
    print(f"\nSaving to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, output_path)

    # Save config
    if config_output_path:
        print(f"Saving config to {config_output_path}...")
        with open(config_output_path, "w") as f:
            json.dump(config, f, indent=2)

    print(f"\nConversion complete!")
    print(f"  Output: {output_path}")
    if config_output_path:
        print(f"  Config: {config_output_path}")

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Convert GPT-SoVITS PyTorch weights to MLX safetensors"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to PyTorch checkpoint (.ckpt or .pth)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output path for safetensors file"
    )
    parser.add_argument(
        "--config", "-c",
        help="Output path for config JSON (optional)"
    )

    args = parser.parse_args()

    # Set default config path
    config_path = args.config
    if config_path is None:
        output_path = Path(args.output)
        config_path = str(output_path.parent / f"{output_path.stem}_config.json")

    convert_checkpoint(args.input, args.output, config_path)


if __name__ == "__main__":
    main()
