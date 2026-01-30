#!/usr/bin/env python3
"""
Convert GPT-SoVITS preprocessed data to Rust training format.

Input format (GPT-SoVITS):
  /exp_dir/
  ├── 2-name2text.txt      # filename<TAB>phonemes<TAB>word2ph<TAB>text
  ├── 3-bert/*.wav.pt      # BERT features [1024, seq_len] or [seq_len, 1024]
  ├── 6-name2semantic.tsv  # filename<TAB>space-separated tokens

Output format (Rust training):
  /output_dir/
  ├── metadata.json        # {"num_samples": N, "samples": [...]}
  ├── phoneme_ids/*.npy    # int32 [seq_len]
  ├── bert_features/*.npy  # float32 [1024, seq_len]
  ├── semantic_ids/*.npy   # int32 [seq_len]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add GPT-SoVITS to path for symbol table
MOYOYO_ROOT = "/Users/yuechen/home/OminiX-MLX/gpt-sovits-clone-mlx/MoYoYo.tts"
sys.path.insert(0, MOYOYO_ROOT)
sys.path.insert(0, os.path.join(MOYOYO_ROOT, "GPT_SoVITS"))

from text import symbols2


def get_symbol_to_id():
    """Get symbol to ID mapping for v2."""
    return {s: i for i, s in enumerate(symbols2.symbols)}


def parse_name2text(path: Path):
    """Parse 2-name2text.txt file.

    Format: filename<TAB>phonemes<TAB>word2ph<TAB>normalized_text
    """
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 4:
                print(f"Warning: Skipping malformed line: {line[:50]}...")
                continue

            filename, phonemes_str, word2ph_str, text = parts
            # Remove .wav extension for sample ID
            sample_id = filename.replace('.wav', '')

            # Split phonemes
            phonemes = phonemes_str.split(' ')

            samples.append({
                'id': sample_id,
                'filename': filename,
                'phonemes': phonemes,
                'text': text
            })

    return samples


def parse_semantic_tsv(path: Path):
    """Parse 6-name2semantic.tsv file.

    Format: filename<TAB>space-separated semantic tokens
    """
    semantics = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                print(f"Warning: Skipping malformed semantic line: {line[:50]}...")
                continue

            filename, tokens_str = parts
            sample_id = filename.replace('.wav', '')
            tokens = [int(t) for t in tokens_str.split(' ')]
            semantics[sample_id] = tokens

    return semantics


def convert_bert_feature(pt_path: Path) -> np.ndarray:
    """Load BERT feature from .pt file and convert to numpy.

    Returns: float32 array with shape [1024, seq_len]
    """
    data = torch.load(pt_path, map_location='cpu', weights_only=False)

    # Handle different tensor formats
    if isinstance(data, torch.Tensor):
        arr = data.numpy()
    else:
        arr = np.array(data)

    # Ensure float32
    arr = arr.astype(np.float32)

    # Ensure shape is [1024, seq_len]
    if arr.ndim == 1:
        arr = arr.reshape(1024, -1)
    elif arr.shape[0] != 1024 and arr.shape[-1] == 1024:
        # Shape is [seq_len, 1024], transpose
        arr = arr.T

    return arr


def convert_dataset(input_dir: Path, output_dir: Path):
    """Convert GPT-SoVITS preprocessed data to training format."""

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'phoneme_ids').mkdir(exist_ok=True)
    (output_dir / 'bert_features').mkdir(exist_ok=True)
    (output_dir / 'semantic_ids').mkdir(exist_ok=True)

    # Get symbol to ID mapping
    symbol_to_id = get_symbol_to_id()
    print(f"Symbol vocabulary size: {len(symbol_to_id)}")

    # Parse input files
    name2text_path = input_dir / '2-name2text.txt'
    semantic_path = input_dir / '6-name2semantic.tsv'
    bert_dir = input_dir / '3-bert'

    if not name2text_path.exists():
        raise FileNotFoundError(f"2-name2text.txt not found in {input_dir}")
    if not semantic_path.exists():
        raise FileNotFoundError(f"6-name2semantic.tsv not found in {input_dir}")
    if not bert_dir.exists():
        raise FileNotFoundError(f"3-bert directory not found in {input_dir}")

    # Parse text data
    print("Parsing 2-name2text.txt...")
    text_samples = parse_name2text(name2text_path)
    print(f"  Found {len(text_samples)} samples")

    # Parse semantic data
    print("Parsing 6-name2semantic.tsv...")
    semantic_data = parse_semantic_tsv(semantic_path)
    print(f"  Found {len(semantic_data)} samples")

    # Convert each sample
    metadata_samples = []
    success_count = 0

    for sample in text_samples:
        sample_id = sample['id']

        # Check if we have all data for this sample
        if sample_id not in semantic_data:
            print(f"Warning: No semantic data for {sample_id}, skipping")
            continue

        bert_path = bert_dir / f"{sample_id}.wav.pt"
        if not bert_path.exists():
            # Try without .wav
            bert_path = bert_dir / f"{sample_id}.pt"
            if not bert_path.exists():
                print(f"Warning: No BERT features for {sample_id}, skipping")
                continue

        try:
            # Convert phonemes to IDs
            phoneme_ids = []
            for phone in sample['phonemes']:
                if phone in symbol_to_id:
                    phoneme_ids.append(symbol_to_id[phone])
                else:
                    print(f"Warning: Unknown phoneme '{phone}' in {sample_id}")
                    # Use UNK token
                    phoneme_ids.append(symbol_to_id.get('UNK', 0))

            phoneme_ids = np.array(phoneme_ids, dtype=np.int32)

            # Load and convert BERT features
            bert_features = convert_bert_feature(bert_path)

            # Get semantic IDs
            semantic_ids = np.array(semantic_data[sample_id], dtype=np.int32)

            # Save numpy files
            np.save(output_dir / 'phoneme_ids' / f"{sample_id}.npy", phoneme_ids)
            np.save(output_dir / 'bert_features' / f"{sample_id}.npy", bert_features)
            np.save(output_dir / 'semantic_ids' / f"{sample_id}.npy", semantic_ids)

            # Add to metadata
            metadata_samples.append({
                'id': sample_id,
                'audio_path': str(input_dir / 'slicer_opt' / f"{sample_id}.wav"),
                'transcript': sample['text'],
                'phoneme_len': len(phoneme_ids),
                'semantic_len': len(semantic_ids)
            })

            success_count += 1

        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            continue

    # Write metadata
    metadata = {
        'num_samples': len(metadata_samples),
        'samples': metadata_samples
    }

    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nConversion complete!")
    print(f"  Converted: {success_count} samples")
    print(f"  Output: {output_dir}")
    print(f"  Metadata: {output_dir / 'metadata.json'}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert GPT-SoVITS preprocessed data to Rust training format'
    )
    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Input directory (GPT-SoVITS experiment directory)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        help='Output directory for converted data'
    )

    args = parser.parse_args()

    convert_dataset(args.input, args.output)


if __name__ == '__main__':
    main()
