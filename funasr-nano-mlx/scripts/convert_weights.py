#!/usr/bin/env python3
"""
Convert Fun-ASR-Nano PyTorch weights to safetensors format.

Usage:
    python convert_weights.py <model_dir>

This script converts model.pt to model.safetensors with weight name mapping
compatible with the Rust implementation.
"""

import sys
import torch
from pathlib import Path
from safetensors.torch import save_file
from collections import OrderedDict


def map_weight_name(pt_name: str) -> str:
    """Map PyTorch weight names to Rust-compatible names."""
    # Remove the 'state_dict.' prefix if present
    name = pt_name

    # Map component prefixes
    mappings = [
        # Audio encoder
        ("audio_encoder.encoders0.", "encoder.encoders0."),
        ("audio_encoder.encoders.", "encoder.encoders."),
        ("audio_encoder.tp_encoders.", "encoder.tp_encoders."),
        ("audio_encoder.after_norm.", "encoder.after_norm."),
        ("audio_encoder.tp_norm.", "encoder.tp_norm."),

        # Audio adaptor
        ("audio_adaptor.linear1.", "adaptor.linear1."),
        ("audio_adaptor.linear2.", "adaptor.linear2."),
        ("audio_adaptor.blocks.", "adaptor.blocks."),

        # LLM (Qwen3)
        ("llm.model.embed_tokens.", "llm.embed_tokens."),
        ("llm.model.layers.", "llm.layers."),
        ("llm.model.norm.", "llm.norm."),
        ("llm.lm_head.", "llm.lm_head."),
    ]

    for pt_prefix, rust_prefix in mappings:
        if name.startswith(pt_prefix):
            name = rust_prefix + name[len(pt_prefix):]
            break

    # Map layer component names
    layer_mappings = [
        # Encoder attention
        (".self_attn.linear_q_k_v.", ".attn.qkv."),
        (".self_attn.linear_out.", ".attn.out."),
        (".self_attn.fsmn_block.", ".attn.fsmn."),

        # Encoder FFN
        (".feed_forward.w_1.", ".ffn.w1."),
        (".feed_forward.w_2.", ".ffn.w2."),

        # Encoder norms
        (".norm1.", ".norm1."),
        (".norm2.", ".norm2."),

        # Adaptor attention (separate Q/K/V)
        (".self_attn.linear_q.", ".attn.q."),
        (".self_attn.linear_k.", ".attn.k."),
        (".self_attn.linear_v.", ".attn.v."),

        # LLM attention
        (".self_attn.q_proj.", ".attn.q_proj."),
        (".self_attn.k_proj.", ".attn.k_proj."),
        (".self_attn.v_proj.", ".attn.v_proj."),
        (".self_attn.o_proj.", ".attn.o_proj."),
        (".self_attn.q_norm.", ".attn.q_norm."),
        (".self_attn.k_norm.", ".attn.k_norm."),

        # LLM MLP
        (".mlp.gate_proj.", ".mlp.gate_proj."),
        (".mlp.up_proj.", ".mlp.up_proj."),
        (".mlp.down_proj.", ".mlp.down_proj."),

        # LLM norms
        (".input_layernorm.", ".input_layernorm."),
        (".post_attention_layernorm.", ".post_attention_layernorm."),
    ]

    for pt_suffix, rust_suffix in layer_mappings:
        name = name.replace(pt_suffix, rust_suffix)

    return name


def convert_weights(model_dir: Path):
    """Convert PyTorch weights to safetensors."""
    model_path = model_dir / "model.pt"
    output_path = model_dir / "model.safetensors"

    print(f"Loading {model_path}...")
    data = torch.load(model_path, map_location="cpu", weights_only=False)

    if "state_dict" in data:
        state_dict = data["state_dict"]
    else:
        state_dict = data

    print(f"Found {len(state_dict)} weights")

    # Convert weights
    converted = OrderedDict()
    skipped = []

    for pt_name, tensor in state_dict.items():
        rust_name = map_weight_name(pt_name)

        # Convert to float32 if bfloat16 (safetensors doesn't support bf16)
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(torch.float16)

        converted[rust_name] = tensor.contiguous()

        if pt_name != rust_name:
            print(f"  {pt_name} -> {rust_name}")

    print(f"\nConverted {len(converted)} weights")
    if skipped:
        print(f"Skipped {len(skipped)} weights: {skipped}")

    # Save as safetensors
    print(f"\nSaving to {output_path}...")
    save_file(converted, str(output_path))
    print("Done!")

    # Print summary
    total_params = sum(t.numel() for t in converted.values())
    print(f"\nTotal parameters: {total_params / 1e6:.1f}M")

    # Group by component
    components = {}
    for name, tensor in converted.items():
        comp = name.split(".")[0]
        if comp not in components:
            components[comp] = 0
        components[comp] += tensor.numel()

    print("\nBy component:")
    for comp, params in sorted(components.items()):
        print(f"  {comp}: {params / 1e6:.1f}M")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model_dir>")
        print(f"Example: {sys.argv[0]} ./Fun-ASR-Nano-2512")
        sys.exit(1)

    model_dir = Path(sys.argv[1])
    if not model_dir.exists():
        print(f"Error: {model_dir} does not exist")
        sys.exit(1)

    convert_weights(model_dir)


if __name__ == "__main__":
    main()
