#!/usr/bin/env python3
"""One-stop setup script for gpt-sovits-mlx model weights.

Downloads pre-trained models from HuggingFace and converts them to
MLX-compatible safetensors format.

Usage:
    python scripts/setup_models.py

Downloads ~2GB and produces all required model files in:
    ~/.dora/models/primespeech/gpt-sovits-mlx/
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path.home() / ".dora" / "models" / "primespeech" / "gpt-sovits-mlx"

HF_REPO_GPT_SOVITS = "lj1995/GPT-SoVITS"
GPT_CKPT_FILE = "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
SOVITS_PTH_FILE = "gsv-v2final-pretrained/s2G2333k.pth"

HF_REPO_HUBERT = "TencentGameMate/chinese-hubert-base"
HF_REPO_BERT = "hfl/chinese-roberta-wwm-ext-large"

REQUIRED_PACKAGES = ["torch", "safetensors", "transformers", "huggingface_hub"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dependencies():
    """Check (and install if missing) required Python packages."""
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        pip_base = [sys.executable, "-m", "pip", "install"]

        # Install torch separately with CPU-only index to save disk/time
        if "torch" in missing:
            subprocess.check_call(pip_base + [
                "torch", "--index-url", "https://download.pytorch.org/whl/cpu",
            ])
            missing.remove("torch")

        # Install remaining packages from default PyPI
        if missing:
            subprocess.check_call(pip_base + missing)

        print()


def download_hf_file(repo_id: str, filename: str, cache_dir: Path) -> Path:
    """Download a single file from HuggingFace Hub and return its local path."""
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=str(cache_dir)))


def download_hf_model(repo_id: str, cache_dir: Path, allow_patterns=None) -> Path:
    """Download a full model repo (or subset) and return its local snapshot path."""
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(
        repo_id=repo_id,
        cache_dir=str(cache_dir),
        allow_patterns=allow_patterns,
    ))


# ---------------------------------------------------------------------------
# Conversion: GPT T2S (.ckpt -> safetensors)
# ---------------------------------------------------------------------------

def convert_gpt(ckpt_path: Path, output_path: Path):
    """Convert GPT-SoVITS T2S checkpoint to safetensors.

    Reuses the same logic as convert_gpt_weights.py:
    - Split combined QKV projections into separate Q, K, V
    - Remap PyTorch naming to MLX naming
    """
    import numpy as np
    import torch
    from safetensors.numpy import save_file

    print(f"  Loading GPT checkpoint: {ckpt_path}")
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # Extract config for num_layers
    config = {}
    if "config" in checkpoint and isinstance(checkpoint["config"], dict):
        model_cfg = checkpoint["config"].get("model", {})
        config["num_layers"] = model_cfg.get("n_layer", 24)
        config["hidden_size"] = model_cfg.get("embedding_dim", 512)
    else:
        config["num_layers"] = 24
        config["hidden_size"] = 512

    # Get state dict
    if "weight" in checkpoint:
        state_dict = checkpoint["weight"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    num_layers = config["num_layers"]
    converted = {}

    def to_np(t):
        arr = t.cpu().numpy() if hasattr(t, "cpu") else np.array(t)
        return arr.astype(np.float32) if arr.dtype == np.float64 else arr

    def split_qkv(w, b=None):
        d = w.shape[0] // 3
        ws = {"q": w[:d], "k": w[d:2*d], "v": w[2*d:]}
        bs = None
        if b is not None:
            bs = {"q": b[:d], "k": b[d:2*d], "v": b[2*d:]}
        return ws, bs

    for name, tensor in state_dict.items():
        t = to_np(tensor)

        if name == "model.bert_proj.weight":
            converted["audio_proj.weight"] = t
        elif name == "model.bert_proj.bias":
            converted["audio_proj.bias"] = t
        elif name == "model.ar_text_embedding.word_embeddings.weight":
            converted["phoneme_embed.weight"] = t
        elif name == "model.ar_audio_embedding.word_embeddings.weight":
            converted["semantic_embed.weight"] = t
        elif name == "model.ar_text_position.alpha":
            converted["text_pos_alpha"] = t
        elif name == "model.ar_audio_position.alpha":
            converted["audio_pos_alpha"] = t
        elif name == "model.ar_predict_layer.weight":
            converted["lm_head.weight"] = t
        elif name == "model.ar_predict_layer.bias":
            converted["lm_head.bias"] = t
        elif "model.h.layers." in name:
            parts = name.split(".")
            idx = int(parts[3])
            if idx >= num_layers:
                continue
            suffix = ".".join(parts[4:])

            if suffix == "self_attn.in_proj_weight":
                ws, _ = split_qkv(t)
                for k in ("q", "k", "v"):
                    converted[f"layers.{idx}.self_attn.{k}_proj.weight"] = ws[k]
            elif suffix == "self_attn.in_proj_bias":
                _, bs = split_qkv(np.zeros((t.shape[0], 1), dtype=t.dtype), t)
                for k in ("q", "k", "v"):
                    converted[f"layers.{idx}.self_attn.{k}_proj.bias"] = bs[k]
            elif suffix == "self_attn.out_proj.weight":
                converted[f"layers.{idx}.self_attn.o_proj.weight"] = t
            elif suffix == "self_attn.out_proj.bias":
                converted[f"layers.{idx}.self_attn.o_proj.bias"] = t
            elif suffix == "linear1.weight":
                converted[f"layers.{idx}.mlp.gate_proj.weight"] = t
            elif suffix == "linear1.bias":
                converted[f"layers.{idx}.mlp.gate_proj.bias"] = t
            elif suffix == "linear2.weight":
                converted[f"layers.{idx}.mlp.down_proj.weight"] = t
            elif suffix == "linear2.bias":
                converted[f"layers.{idx}.mlp.down_proj.bias"] = t
            elif suffix == "norm1.weight":
                converted[f"layers.{idx}.input_layernorm.weight"] = t
            elif suffix == "norm1.bias":
                converted[f"layers.{idx}.input_layernorm.bias"] = t
            elif suffix == "norm2.weight":
                converted[f"layers.{idx}.post_attention_layernorm.weight"] = t
            elif suffix == "norm2.bias":
                converted[f"layers.{idx}.post_attention_layernorm.bias"] = t

    print(f"  Converted {len(converted)} tensors")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(output_path))
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Conversion: SoVITS VITS (.pth -> safetensors)
# ---------------------------------------------------------------------------

def convert_sovits(pth_path: Path, output_path: Path):
    """Convert SoVITS VITS checkpoint to safetensors.

    The PyTorch checkpoint contains a 'weight' key with the state dict.
    Key names (ssl_proj.*, enc_p.*, flow.*, dec.*, quantizer.*, ref_enc.*)
    already match what the Rust code expects, so we just extract and save.
    """
    import numpy as np
    import torch
    from safetensors.numpy import save_file

    print(f"  Loading SoVITS checkpoint: {pth_path}")
    checkpoint = torch.load(str(pth_path), map_location="cpu", weights_only=False)

    if "weight" in checkpoint:
        state_dict = checkpoint["weight"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    converted = {}
    for name, tensor in state_dict.items():
        arr = tensor.cpu().numpy() if hasattr(tensor, "cpu") else np.array(tensor)
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        converted[name] = arr

    print(f"  Converted {len(converted)} tensors")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(output_path))
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Conversion: HuBERT (pytorch_model.bin -> safetensors)
# ---------------------------------------------------------------------------

def convert_hubert(model_dir: Path, output_path: Path):
    """Convert HuBERT PyTorch weights to safetensors.

    The Rust code expects keys WITHOUT the 'hubert.' prefix, e.g.:
    - feature_extractor.conv_layers.{i}.conv.weight
    - feature_projection.layer_norm.weight
    - encoder.layers.{i}.attention.q_proj.weight
    """
    import numpy as np
    import torch
    from safetensors.numpy import save_file

    bin_path = model_dir / "pytorch_model.bin"
    print(f"  Loading HuBERT weights: {bin_path}")
    state_dict = torch.load(str(bin_path), map_location="cpu", weights_only=False)

    converted = {}
    for name, tensor in state_dict.items():
        arr = tensor.cpu().numpy() if hasattr(tensor, "cpu") else np.array(tensor)
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)

        # Strip 'hubert.' prefix — Rust expects keys like
        # feature_extractor.*, feature_projection.*, encoder.layers.*
        if name.startswith("hubert."):
            new_name = name[len("hubert."):]
        else:
            new_name = name

        converted[new_name] = arr

    print(f"  Converted {len(converted)} tensors")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(output_path))
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Conversion: BERT (pytorch_model.bin -> safetensors)
# ---------------------------------------------------------------------------

def convert_bert(model_dir: Path, output_path: Path):
    """Convert Chinese RoBERTa BERT weights to safetensors.

    The Rust code accepts both naming conventions:
    - New: embeddings.word_embeddings.weight, encoder.layers.{i}.*
    - Old: bert.embeddings.word_embeddings.weight, bert.encoder.layer.{i}.*

    We keep the original HuggingFace naming (with 'bert.' prefix) since
    the Rust loader has fallback support for it.
    """
    import numpy as np
    import torch
    from safetensors.numpy import save_file

    bin_path = model_dir / "pytorch_model.bin"
    print(f"  Loading BERT weights: {bin_path}")
    state_dict = torch.load(str(bin_path), map_location="cpu", weights_only=False)

    converted = {}
    for name, tensor in state_dict.items():
        arr = tensor.cpu().numpy() if hasattr(tensor, "cpu") else np.array(tensor)
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        converted[name] = arr

    print(f"  Converted {len(converted)} tensors")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(output_path))
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Copy tokenizer
# ---------------------------------------------------------------------------

def copy_tokenizer(model_dir: Path, output_dir: Path):
    """Copy tokenizer.json from HuggingFace download to output directory."""
    src = model_dir / "tokenizer.json"
    dst = output_dir / "chinese-roberta-tokenizer" / "tokenizer.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))
    print(f"  Copied tokenizer: {dst}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("gpt-sovits-mlx Model Setup")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}\n")

    # Step 0: Dependencies
    print("[0/5] Checking Python dependencies...")
    ensure_dependencies()
    print()

    cache_dir = OUTPUT_DIR / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download GPT-SoVITS checkpoints
    print("[1/5] Downloading GPT-SoVITS pretrained checkpoints...")
    gpt_ckpt = download_hf_file(HF_REPO_GPT_SOVITS, GPT_CKPT_FILE, cache_dir)
    sovits_pth = download_hf_file(HF_REPO_GPT_SOVITS, SOVITS_PTH_FILE, cache_dir)
    print()

    # Step 2: Download HuBERT
    print("[2/5] Downloading chinese-hubert-base...")
    hubert_dir = download_hf_model(
        HF_REPO_HUBERT, cache_dir,
        allow_patterns=["pytorch_model.bin", "config.json"],
    )
    print()

    # Step 3: Download BERT
    print("[3/5] Downloading chinese-roberta-wwm-ext-large...")
    bert_dir = download_hf_model(
        HF_REPO_BERT, cache_dir,
        allow_patterns=["pytorch_model.bin", "config.json", "tokenizer.json"],
    )
    print()

    # Step 4: Convert all models
    print("[4/5] Converting models to safetensors...")

    print("\n  [GPT T2S]")
    convert_gpt(gpt_ckpt, OUTPUT_DIR / "doubao_mixed_gpt_new.safetensors")

    print("\n  [SoVITS VITS]")
    convert_sovits(sovits_pth, OUTPUT_DIR / "doubao_mixed_sovits_new.safetensors")

    print("\n  [HuBERT]")
    convert_hubert(hubert_dir, OUTPUT_DIR / "hubert.safetensors")

    print("\n  [BERT]")
    convert_bert(bert_dir, OUTPUT_DIR / "bert.safetensors")

    print("\n  [Tokenizer]")
    copy_tokenizer(bert_dir, OUTPUT_DIR)

    print()

    # Step 5: Cleanup cache (optional — keep HF cache for re-runs)
    print("[5/5] Verifying output files...")
    expected_files = [
        "doubao_mixed_gpt_new.safetensors",
        "doubao_mixed_sovits_new.safetensors",
        "hubert.safetensors",
        "bert.safetensors",
        "chinese-roberta-tokenizer/tokenizer.json",
    ]

    all_ok = True
    for fname in expected_files:
        fpath = OUTPUT_DIR / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            print(f"  OK  {fname} ({size_mb:.1f} MB)")
        else:
            print(f"  MISSING  {fname}")
            all_ok = False

    print()
    if all_ok:
        print("Setup complete! All model files are ready.")
        print(f"\nModel directory: {OUTPUT_DIR}")
        print("\nNote: ONNX VITS (vits.onnx) is not included — the Rust code")
        print("will automatically use MLX VITS as fallback. You can also pass")
        print("--mlx-vits explicitly to silence the fallback message.")
        print("\nTo test:")
        print('  cargo run --release --example voice_clone -- \\')
        print('    --mlx-vits --ref /path/to/reference.wav "你好世界"')
    else:
        print("WARNING: Some files are missing. Check the output above for errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
