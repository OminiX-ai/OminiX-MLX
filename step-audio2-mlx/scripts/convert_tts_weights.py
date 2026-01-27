#!/usr/bin/env python3
"""Convert Step-Audio 2 TTS weights from PyTorch to safetensors format.

This script converts:
- flow.pt -> flow.safetensors (CosyVoice2 flow decoder)
- hift.pt -> hifigan.safetensors (HiFT vocoder)

Usage:
    python convert_tts_weights.py <token2wav_dir> <output_dir>

Example:
    python convert_tts_weights.py ./Step-Audio-2-mini/token2wav ./Step-Audio-2-mini/tts_mlx
"""

import argparse
import os
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import save_file


def convert_flow_weights(flow_path: Path) -> Dict[str, torch.Tensor]:
    """Convert flow.pt to safetensors-compatible dict.

    The flow model has:
    - input_embedding: Codebook [6561, 512]
    - spk_embed_affine_layer: Speaker embedding projection
    - encoder: UpsampleConformerEncoderV2
    - decoder: Not in flow.pt (separate CFM decoder)
    """
    print(f"Loading {flow_path}...")
    state_dict = torch.load(flow_path, map_location='cpu', weights_only=False)

    converted = {}

    for key, tensor in state_dict.items():
        # Ensure tensor is contiguous and float32
        if tensor.dtype == torch.float16:
            tensor = tensor.float()
        tensor = tensor.contiguous()

        # Rename keys for cleaner structure
        new_key = key

        # Map to our module structure
        # input_embedding.weight -> codebook.embeddings.weight
        if key == 'input_embedding.weight':
            new_key = 'codebook.embeddings.weight'

        # spk_embed_affine_layer -> speaker_proj
        elif key.startswith('spk_embed_affine_layer'):
            new_key = key.replace('spk_embed_affine_layer', 'speaker_proj')

        # encoder.embed -> encoder.input_proj
        elif key.startswith('encoder.embed'):
            new_key = key.replace('encoder.embed', 'encoder.input_proj')

        # encoder.encoders.N -> encoder.layers.N
        elif 'encoder.encoders.' in key:
            new_key = key.replace('encoder.encoders.', 'encoder.layers.')
            # self_attn.linear_q -> self_attn.q_proj
            new_key = new_key.replace('linear_q', 'q_proj')
            new_key = new_key.replace('linear_k', 'k_proj')
            new_key = new_key.replace('linear_v', 'v_proj')
            new_key = new_key.replace('linear_out', 'out_proj')
            new_key = new_key.replace('linear_pos', 'pos_proj')
            # feed_forward.w_1 -> ffn.up_proj
            new_key = new_key.replace('feed_forward.w_1', 'ffn.up_proj')
            new_key = new_key.replace('feed_forward.w_2', 'ffn.down_proj')
            new_key = new_key.replace('norm_ff', 'ffn_norm')

        # Keep other keys as-is with flow. prefix
        else:
            new_key = f'flow.{key}'

        converted[new_key] = tensor

    print(f"  Converted {len(converted)} tensors")
    return converted


def convert_hift_weights(hift_path: Path) -> Dict[str, torch.Tensor]:
    """Convert hift.pt to safetensors-compatible dict.

    HiFT uses weight parameterization:
    - parametrizations.weight.original0: scale [out, 1, 1]
    - parametrizations.weight.original1: weight [out, in, kernel]

    We reconstruct the actual weight as: weight = original0 * original1 / ||original1||
    """
    print(f"Loading {hift_path}...")
    state_dict = torch.load(hift_path, map_location='cpu', weights_only=False)

    converted = {}

    # Group parameterized weights
    param_groups = {}
    regular_weights = {}

    for key, tensor in state_dict.items():
        if '.parametrizations.weight.' in key:
            # Extract base key and param type
            base_key = key.replace('.parametrizations.weight.original0', '').replace('.parametrizations.weight.original1', '')
            param_type = 'original0' if 'original0' in key else 'original1'

            if base_key not in param_groups:
                param_groups[base_key] = {}
            param_groups[base_key][param_type] = tensor
        else:
            regular_weights[key] = tensor

    # Reconstruct parameterized weights (weight normalization)
    for base_key, params in param_groups.items():
        if 'original0' in params and 'original1' in params:
            g = params['original0']  # scale
            v = params['original1']  # direction

            # Weight normalization: w = g * v / ||v||
            # Normalize over all dims except output channel
            v_norm = v.view(v.size(0), -1).norm(dim=1, keepdim=True)
            v_norm = v_norm.view(v.size(0), *([1] * (v.dim() - 1)))
            weight = g * v / (v_norm + 1e-8)

            # Convert to float32 and contiguous
            weight = weight.float().contiguous()

            # Clean up key name
            new_key = f'hifigan.{base_key}.weight'
            converted[new_key] = weight

    # Add regular weights
    for key, tensor in regular_weights.items():
        if tensor.dtype == torch.float16:
            tensor = tensor.float()
        tensor = tensor.contiguous()

        new_key = f'hifigan.{key}'
        converted[new_key] = tensor

    print(f"  Converted {len(converted)} tensors")
    return converted


def main():
    parser = argparse.ArgumentParser(description='Convert TTS weights to safetensors')
    parser.add_argument('token2wav_dir', type=Path, help='Path to token2wav directory')
    parser.add_argument('output_dir', type=Path, help='Output directory for safetensors')
    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Convert flow.pt
    flow_path = args.token2wav_dir / 'flow.pt'
    if flow_path.exists():
        flow_weights = convert_flow_weights(flow_path)
        flow_output = args.output_dir / 'flow.safetensors'
        save_file(flow_weights, str(flow_output))
        print(f"  Saved to {flow_output}")

        # Print sample keys
        print("\n  Sample flow keys:")
        for k in list(flow_weights.keys())[:10]:
            print(f"    {k}: {flow_weights[k].shape}")
    else:
        print(f"Warning: {flow_path} not found")

    # Convert hift.pt
    hift_path = args.token2wav_dir / 'hift.pt'
    if hift_path.exists():
        hift_weights = convert_hift_weights(hift_path)
        hift_output = args.output_dir / 'hifigan.safetensors'
        save_file(hift_weights, str(hift_output))
        print(f"  Saved to {hift_output}")

        # Print sample keys
        print("\n  Sample hifigan keys:")
        for k in list(hift_weights.keys())[:10]:
            print(f"    {k}: {hift_weights[k].shape}")
    else:
        print(f"Warning: {hift_path} not found")

    print("\nConversion complete!")


if __name__ == '__main__':
    main()
