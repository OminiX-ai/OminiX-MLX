# Voice Cloning Finetuning Guide

This guide covers the complete workflow for finetuning GPT-SoVITS voice models and using them with the Rust MLX TTS pipeline.

> **ONNX Export Options**:
> - **Python Export** (recommended): Use `scripts/export_finetuned_onnx.py` for reliable ONNX export from `.pth` files
> - **Pure Rust Patching**: Use `cargo run --example export_vits_onnx` to patch base ONNX with `.safetensors` weights (206 weights patched, 7 skipped)

## Overview

The GPT-SoVITS TTS system has two main components:
1. **T2S (Text-to-Semantic)**: Converts text + reference audio to semantic tokens
2. **SoVITS (VITS)**: Converts semantic tokens to audio waveform

For voice cloning, you typically finetune the **SoVITS** model on your target voice data.

## Prerequisites

### Python Environment
```bash
# GPT-SoVITS / PrimeSpeech installation
cd ~/home/mofa-studio/node-hub/dora-primespeech/dora_primespeech/moyoyo_tts

# Required packages
pip install torch torchaudio onnx
```

### Rust Environment
```bash
cd ~/home/OminiX-MLX/gpt-sovits-mlx
cargo build --release --example voice_clone
```

### Model Files
- Pretrained GPT (T2S): `~/.dora/models/primespeech/moyoyo/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt`
- Pretrained SoVITS: `~/.dora/models/primespeech/moyoyo/SoVITS_weights/` (e.g., `doubao-mixed.pth`)

## Step 1: Prepare Training Data

### Audio Requirements
- **Format**: WAV, 16-bit PCM
- **Sample Rate**: 32kHz (will be resampled if different)
- **Duration**: 1-10 seconds per clip, total 1-60 minutes recommended
- **Quality**: Clean audio, minimal background noise

### Directory Structure
```
/tmp/fewshot_training/
├── raw_audio/
│   ├── clip_001.wav
│   ├── clip_002.wav
│   └── ...
└── transcripts.txt  # Optional: text transcriptions
```

### Preprocessing with GPT-SoVITS

```bash
cd ~/home/mofa-studio/node-hub/dora-primespeech/dora_primespeech/moyoyo_tts

# Run preprocessing pipeline
python GPT_SoVITS/prepare_datasets/1-get-text.py \
    --inp_text /tmp/fewshot_training/transcripts.txt \
    --inp_wav_dir /tmp/fewshot_training/raw_audio \
    --exp_name fewshot \
    --gpu_numbers 0

# Extract semantic tokens
python GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py \
    --exp_name fewshot

# Extract SSL features
python GPT_SoVITS/prepare_datasets/3-get-semantic.py \
    --exp_name fewshot
```

## Step 2: Finetune Training

### Option A: Python Training (Recommended for Quality)

```bash
cd ~/home/mofa-studio/node-hub/dora-primespeech/dora_primespeech/moyoyo_tts

# Train SoVITS
python GPT_SoVITS/s2_train.py \
    --exp_name fewshot \
    --pretrained_s2G path/to/pretrained_sovits.pth \
    --batch_size 4 \
    --total_epoch 8 \
    --save_every_epoch 2

# Output: logs/fewshot/SoVITS_weights/*.pth
```

### Option B: Rust MLX Training

```bash
cd ~/home/OminiX-MLX/gpt-sovits-mlx

# Prepare training data in Rust format
python scripts/prepare_training_data.py \
    --input /tmp/fewshot_training \
    --output /tmp/fewshot_rust_data

# Train with Rust MLX
cargo run --release --example train_vits -- \
    --data-dir /tmp/fewshot_rust_data \
    --pretrained ~/.dora/models/primespeech/gpt-sovits-mlx/vits_pretrained.safetensors \
    --output /tmp/vits_finetuned.safetensors \
    --lr 0.0001 \
    --batch-size 2 \
    --max-steps 1000

# Convert Rust weights to PyTorch format for ONNX export
python scripts/convert_vits_mlx_to_pytorch.py \
    --input /tmp/vits_finetuned.safetensors \
    --output ~/.dora/models/primespeech/moyoyo/SoVITS_weights/rust_finetuned.pth
```

## Step 3: Convert to ONNX

This is the **critical step** for using finetuned weights with the Rust TTS pipeline.

### Export Script

Create `/tmp/export_finetuned_onnx.py`:

```python
#!/usr/bin/env python3
"""Export finetuned VITS to ONNX using models_onnx (required for Rust pipeline)."""

import sys
import os
import torch

# Add GPT-SoVITS path
MOYOYO_ROOT = "/Users/yuechen/home/mofa-studio/node-hub/dora-primespeech/dora_primespeech/moyoyo_tts"
sys.path.insert(0, MOYOYO_ROOT)
os.chdir(MOYOYO_ROOT)

from module.models_onnx import SynthesizerTrn

class DictToAttrRecursive(dict):
    """Convert nested dict to attribute-accessible object."""
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)


def export_finetuned_to_onnx(weights_path: str, output_path: str):
    """Export finetuned VITS model to ONNX.

    IMPORTANT: Must use models_onnx.SynthesizerTrn, NOT models.SynthesizerTrn.
    The models_onnx version has an ONNX-compatible forward() method.
    """

    print(f"Loading weights from: {weights_path}")
    dict_s2 = torch.load(weights_path, map_location="cpu", weights_only=False)

    hps = dict_s2["config"]
    weight = dict_s2["weight"]

    # Determine version from weight shape
    if weight['enc_p.text_embedding.weight'].shape[0] == 322:
        hps["model"]["version"] = "v1"
    else:
        hps["model"]["version"] = "v2"

    print(f"Detected version: {hps['model']['version']}")

    # Convert to attribute-accessible config (required by models_onnx)
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"

    print(f"Config: filter_length={hps.data.filter_length}, hop_length={hps.data.hop_length}")

    # Build ONNX-compatible model
    # CRITICAL: Use models_onnx.SynthesizerTrn, not models.SynthesizerTrn
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    vq_model.eval()

    # Load weights
    missing, unexpected = vq_model.load_state_dict(weight, strict=False)
    print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    # Create dummy inputs
    batch_size = 1
    codes_len = 100
    text_len = 50
    refer_len = 256

    dummy_codes = torch.randint(0, 1024, (batch_size, 1, codes_len))
    dummy_text = torch.randint(0, 732 if hps.model.version == "v2" else 322, (batch_size, text_len))
    dummy_refer = torch.randn(batch_size, 704, refer_len)
    dummy_noise_scale = torch.tensor(0.5)

    print(f"Exporting to: {output_path}")

    torch.onnx.export(
        vq_model,
        (dummy_codes, dummy_text, dummy_refer, dummy_noise_scale),
        output_path,
        input_names=["codes", "text", "refer", "noise_scale"],
        output_names=["audio"],
        dynamic_axes={
            "codes": {2: "T_codes"},
            "text": {1: "T_text"},
            "refer": {2: "T_refer"},
            "audio": {0: "audio_dim_0", 2: "T_audio"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,  # IMPORTANT: Use legacy exporter for compatibility
    )

    print(f"Export successful!")

    # Verify
    import onnx
    model_onnx = onnx.load(output_path)
    onnx.checker.check_model(model_onnx)
    print(f"ONNX model verified!")

    # Check file size (should be ~155 MB for v2)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"Output size: {size_mb:.1f} MB")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to finetuned .pth weights")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    args = parser.parse_args()

    export_finetuned_to_onnx(args.input, args.output)
```

### Run Export

```bash
cd ~/home/mofa-studio/node-hub/dora-primespeech/dora_primespeech
PYTHONPATH=".:moyoyo_tts" python /tmp/export_finetuned_onnx.py \
    --input ~/.dora/models/primespeech/moyoyo/SoVITS_weights/finetuned_fewshot.pth \
    --output ~/.dora/models/primespeech/gpt-sovits-mlx/marc_vits.onnx
```

Expected output:
```
Loading weights from: ...finetuned_fewshot.pth
Detected version: v2
Config: filter_length=2048, hop_length=640
Missing keys: 6, Unexpected keys: 103
Exporting to: ...marc_vits.onnx
Export successful!
ONNX model verified!
Output size: 155.1 MB
```

### Common Export Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: moyoyo_tts` | Wrong PYTHONPATH | Run from `dora_primespeech` dir with `PYTHONPATH=".:moyoyo_tts"` |
| `GuardOnDataDependentSymNode` error | PyTorch 2.x dynamo export | Add `dynamo=False` to `torch.onnx.export()` |
| Output size ~162 MB instead of ~155 MB | Used `models.SynthesizerTrn` | Use `models_onnx.SynthesizerTrn` instead |
| Audio has wrong pitch/sounds different | Wrong model class | Must use `models_onnx.SynthesizerTrn` |

## Step 4: Use with Rust TTS Pipeline

### Method 1: Command Line

```bash
cd ~/home/OminiX-MLX/gpt-sovits-mlx

# With explicit ONNX path
cargo run --example voice_clone --release -- \
    "今天天气真不错" \
    --ref ~/.dora/models/primespeech/moyoyo/ref_audios/fewshot_ref.wav \
    --ref-text "一个是clawdbot可以替代人做事情的agent" \
    --vits-onnx ~/.dora/models/primespeech/gpt-sovits-mlx/marc_vits.onnx

# Or use voice preset (after adding to voice_clone.rs)
cargo run --example voice_clone --release -- "今天天气真不错" --voice marc
```

### Method 2: Add Voice Preset

Edit `examples/voice_clone.rs`:

```rust
// Add constants
const MARC_REF_AUDIO: &str = "/path/to/ref_audios/fewshot_ref.wav";
const MARC_REF_TEXT: &str = "一个是clawdbot可以替代人做事情的agent";
const MARC_VITS_ONNX: &str = "/path/to/gpt-sovits-mlx/marc_vits.onnx";

// Add to voice matching
"marc" => {
    if ref_audio.is_none() { ref_audio = Some(MARC_REF_AUDIO.to_string()); }
    if ref_text.is_none() { ref_text = Some(MARC_REF_TEXT.to_string()); }
    if vits_onnx_model.is_none() { vits_onnx_model = Some(MARC_VITS_ONNX.to_string()); }
}
```

### Method 3: PrimeSpeech Integration

Add to `dora_primespeech/config.py`:

```python
VOICE_CONFIGS = {
    # ... existing voices ...
    "Marc": {
        "repository": None,  # Local finetuned model
        "gpt_weights": "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
        "sovits_weights": "SoVITS_weights/finetuned_fewshot.pth",
        "reference_audio": "ref_audios/fewshot_ref.wav",
        "prompt_text": "一个是clawdbot可以替代人做事情的agent",
        "text_lang": "zh",
        "prompt_lang": "zh",
        "speed_factor": 1.0,
    },
}
```

## File Locations Summary

| File | Location | Description |
|------|----------|-------------|
| Pretrained T2S | `~/.dora/models/primespeech/moyoyo/gsv-v2final-pretrained/*.ckpt` | GPT model for text-to-semantic |
| Pretrained SoVITS | `~/.dora/models/primespeech/moyoyo/SoVITS_weights/*.pth` | VITS vocoder |
| Finetuned SoVITS | `~/.dora/models/primespeech/moyoyo/SoVITS_weights/finetuned_*.pth` | Your trained weights |
| ONNX Models | `~/.dora/models/primespeech/gpt-sovits-mlx/*.onnx` | ONNX for Rust pipeline |
| Reference Audio | `~/.dora/models/primespeech/moyoyo/ref_audios/*.wav` | Voice reference clips |

## Troubleshooting

### Voice sounds wrong (different pitch/timbre)
- **Cause**: Used wrong model class for ONNX export
- **Fix**: Use `models_onnx.SynthesizerTrn`, not `models.SynthesizerTrn`

### ONNX export fails with dynamo error
- **Cause**: PyTorch 2.x uses new export by default
- **Fix**: Add `dynamo=False` to `torch.onnx.export()`

### Missing keys warning during export
- **Expected**: 6 missing keys (speaker embedding layers) and ~103 unexpected keys (encoder_q layers) is normal
- **These are inference-only differences and don't affect audio quality**

### Audio is silent or very quiet
- **Check**: Reference audio quality and duration
- **Check**: Ensure noise_scale=0.5 (default)

## Testing Finetuned Weights

This section covers how to test finetuned weights in both Python (PrimeSpeech) and Rust pipelines.

### File Locations

After training, you should have these files:

| Source | Format | Location |
|--------|--------|----------|
| Python training | `.pth` | `~/.dora/models/primespeech/moyoyo/SoVITS_weights/finetuned_fewshot.pth` |
| Rust MLX training | `.safetensors` | `/tmp/vits_finetuned.generator.safetensors` |
| Rust → PyTorch converted | `.pth` | `~/.dora/models/primespeech/moyoyo/SoVITS_weights/rust_finetuned.pth` |

### Method 1: Test in PrimeSpeech (Python)

Use PrimeSpeech to test finetuned weights directly. Requires `mofa-studio` conda environment.

```bash
# Activate the correct conda environment
conda activate mofa-studio

# Run test script
python3 << 'EOF'
import sys
import os
sys.path.insert(0, os.path.expanduser('~/home/mofa-studio/node-hub/dora-primespeech'))
os.environ['PRIMESPEECH_MODEL_DIR'] = os.path.expanduser('~/.dora/models/primespeech')

from dora_primespeech.moyoyo_tts_wrapper_streaming_fix import StreamingMoYoYoTTSWrapper

# Configure voice with finetuned weights
voice_config = {
    'gpt_weights': 'gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt',
    'sovits_weights': 'SoVITS_weights/rust_finetuned.pth',  # or finetuned_fewshot.pth for Python-trained
    'reference_audio': 'ref_audios/fewshot_ref.wav',
    'prompt_text': '一个是clawdbot可以替代人做事情的agent',
}

wrapper = StreamingMoYoYoTTSWrapper(
    voice="Marc",
    device="mps",  # or "cpu" or "cuda"
    enable_streaming=False,
    voice_config=voice_config
)

sr, audio = wrapper.synthesize('今天天气真不错', language='zh', speed=1.0)

import soundfile as sf
sf.write('/tmp/test_primespeech.wav', audio, sr)
print(f'Saved {len(audio)} samples at {sr}Hz to /tmp/test_primespeech.wav')
EOF

# Play the result
afplay /tmp/test_primespeech.wav
```

### Method 2: Test in Rust ONNX Pipeline

#### Option A: Python ONNX Export (Recommended for Production)

Export finetuned weights to ONNX using Python, then use in Rust:

```bash
# 1. Export to ONNX (requires Python)
cd ~/home/mofa-studio/node-hub/dora-primespeech/dora_primespeech
conda activate mofa-studio
PYTHONPATH=".:moyoyo_tts" python ~/home/OminiX-MLX/gpt-sovits-mlx/scripts/export_finetuned_onnx.py \
    --input ~/.dora/models/primespeech/moyoyo/SoVITS_weights/rust_finetuned.pth \
    --output ~/.dora/models/primespeech/gpt-sovits-mlx/rust_finetuned_vits.onnx

# 2. Test with Rust pipeline
cd ~/home/OminiX-MLX/gpt-sovits-mlx
cargo run --example voice_clone --release -- "今天天气真不错" \
    --vits-onnx ~/.dora/models/primespeech/gpt-sovits-mlx/rust_finetuned_vits.onnx \
    --ref ~/.dora/models/primespeech/moyoyo/ref_audios/fewshot_ref.wav \
    --ref-text "一个是clawdbot可以替代人做事情的agent"
```

#### Option B: Pure Rust ONNX Patching

Patch the base ONNX model with finetuned weights directly in Rust (no Python required):

```bash
cd ~/home/OminiX-MLX/gpt-sovits-mlx

# 1. Patch base ONNX with finetuned safetensors weights
cargo run --release --example export_vits_onnx -- \
    --base ~/.dora/models/primespeech/gpt-sovits-mlx/vits.onnx \
    --weights /tmp/vits_fixed_1e5.generator.safetensors \
    --output /tmp/patched_vits.onnx

# Expected output:
# Patched: 206
# Skipped: 7
# Output size: 155.2 MB

# 2. Test with Rust pipeline
cargo run --example voice_clone --release -- "今天天气真不错" \
    --vits-onnx /tmp/patched_vits.onnx \
    --ref ~/.dora/models/primespeech/moyoyo/ref_audios/fewshot_ref.wav \
    --ref-text "一个是clawdbot可以替代人做事情的agent"
```

### Method 3: Use Pre-configured Voice Preset

If you've added the voice to `examples/voice_clone.rs`:

```bash
cd ~/home/OminiX-MLX/gpt-sovits-mlx

# Using marc voice preset (uses marc_vits.onnx by default)
cargo run --example voice_clone --release -- "今天天气真不错" --voice marc

# Override with different ONNX
cargo run --example voice_clone --release -- "今天天气真不错" \
    --voice marc \
    --vits-onnx /tmp/patched_vits.onnx
```

### Comparing Outputs

To verify finetuned weights work correctly, compare outputs from different pipelines:

```bash
# Generate with PrimeSpeech (Python)
conda activate mofa-studio
# ... (run PrimeSpeech script above, saves to /tmp/test_primespeech.wav)

# Generate with Rust ONNX
cargo run --example voice_clone --release -- "今天天气真不错" \
    --voice marc \
    --vits-onnx /tmp/patched_vits.onnx \
    --output /tmp/test_rust_onnx.wav

# Compare
afplay /tmp/test_primespeech.wav
afplay /tmp/test_rust_onnx.wav
```

Both outputs should sound similar with the same voice characteristics.

## Quick Reference

```bash
# Export finetuned weights to ONNX (Python)
cd ~/home/mofa-studio/node-hub/dora-primespeech/dora_primespeech
conda activate mofa-studio
PYTHONPATH=".:moyoyo_tts" python ~/home/OminiX-MLX/gpt-sovits-mlx/scripts/export_finetuned_onnx.py \
    --input /path/to/finetuned.pth \
    --output /path/to/output.onnx

# Patch ONNX with safetensors (Pure Rust)
cd ~/home/OminiX-MLX/gpt-sovits-mlx
cargo run --release --example export_vits_onnx -- \
    --base ~/.dora/models/primespeech/gpt-sovits-mlx/vits.onnx \
    --weights /path/to/finetuned.generator.safetensors \
    --output /path/to/patched.onnx

# Test with Rust pipeline
cargo run --example voice_clone --release -- "测试文本" \
    --vits-onnx /path/to/output.onnx \
    --ref /path/to/ref.wav \
    --ref-text "参考音频文本"

# Test with PrimeSpeech (Python)
conda activate mofa-studio
python3 -c "
import sys, os
sys.path.insert(0, os.path.expanduser('~/home/mofa-studio/node-hub/dora-primespeech'))
os.environ['PRIMESPEECH_MODEL_DIR'] = os.path.expanduser('~/.dora/models/primespeech')
from dora_primespeech.moyoyo_tts_wrapper_streaming_fix import StreamingMoYoYoTTSWrapper
wrapper = StreamingMoYoYoTTSWrapper(voice='Marc', device='mps', enable_streaming=False,
    voice_config={'sovits_weights': 'SoVITS_weights/rust_finetuned.pth',
                  'gpt_weights': 'gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt',
                  'reference_audio': 'ref_audios/fewshot_ref.wav',
                  'prompt_text': '一个是clawdbot可以替代人做事情的agent'})
sr, audio = wrapper.synthesize('测试文本', language='zh')
import soundfile as sf; sf.write('/tmp/test.wav', audio, sr)
"
```
