#!/usr/bin/env python3
"""Export GPT-SoVITS VITS model to ONNX format.

Usage:
    python export_vits_onnx.py \
        --moyoyo-tts /path/to/dora-primespeech/dora_primespeech/moyoyo_tts \
        --checkpoint /path/to/doubao-mixed.pth \
        --output /tmp/gpt-sovits-mlx/vits.onnx
"""

import argparse
import sys
import os
import types
import numpy as np

def setup_imports(moyoyo_tts_dir):
    """Set up imports so 'module.models' and 'moyoyo_tts.module.*' resolve
    without triggering moyoyo_tts/__init__.py (which pulls in lightning/torchaudio)."""
    # Add moyoyo_tts dir so 'module.*' works directly
    sys.path.insert(0, moyoyo_tts_dir)
    # Add parent so 'moyoyo_tts.module.*' works, but block the __init__.py
    parent = os.path.dirname(moyoyo_tts_dir)
    sys.path.insert(0, parent)
    # Create a dummy moyoyo_tts package that uses the real directory for submodules
    dummy = types.ModuleType("moyoyo_tts")
    dummy.__path__ = [moyoyo_tts_dir]
    dummy.__package__ = "moyoyo_tts"
    sys.modules["moyoyo_tts"] = dummy

import torch
import torch.nn as nn


class VITSDecodeWrapper(nn.Module):
    """Wrapper that exposes VITS decode() as a standard forward() for ONNX export.

    Freezes: version=v2, speed=1.0, semantic_frame_rate=25hz, single refer tensor.
    """

    def __init__(self, vits_model):
        super().__init__()
        self.vits = vits_model

    def forward(self, codes, text, refer, noise_scale):
        """
        Args:
            codes: [1, 1, T_codes] int64 - semantic token indices
            text: [1, T_text] int64 - phoneme indices
            refer: [1, 704, T_refer] float32 - reference mel spectrogram (v2 uses 704 channels)
            noise_scale: scalar float32
        Returns:
            audio: [1, 1, T_audio] float32
        """
        from module import commons

        # Reference encoder
        refer_lengths = torch.LongTensor([refer.size(2)]).to(refer.device)
        refer_mask = torch.unsqueeze(
            commons.sequence_mask(refer_lengths, refer.size(2)), 1
        ).to(refer.dtype)
        ge = self.vits.ref_enc(refer * refer_mask, refer_mask)

        # Quantizer decode
        quantized = self.vits.quantizer.decode(codes)

        # 25hz → double length (nearest-neighbor upsample 2x)
        # Use repeat_interleave instead of F.interpolate to avoid baked-in shapes
        quantized = quantized.repeat_interleave(2, dim=2)

        # Lengths — use tensor ops to keep dynamic
        y_lengths = (codes.size(2) * 2) * torch.ones(1, dtype=torch.long, device=codes.device)
        text_lengths = text.size(-1) * torch.ones(1, dtype=torch.long, device=text.device)

        # Text encoder
        x, m_p, logs_p, y_mask = self.vits.enc_p(
            quantized, y_lengths, text, text_lengths, ge, speed=1.0
        )

        # Sample
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        # Flow reverse
        z = self.vits.flow(z_p, y_mask, g=ge, reverse=True)

        # Decode audio
        o = self.vits.dec((z * y_mask)[:, :, :], g=ge)
        return o


def load_vits_model(checkpoint_path):
    """Load SynthesizerTrn from checkpoint."""
    from module.models import SynthesizerTrn

    dict_s2 = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hps = dict_s2["config"]

    # Detect version
    if dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
        version = "v1"
    else:
        version = "v2"

    model_config = vars(hps.model) if hasattr(hps.model, '__dict__') else dict(hps.model)
    model_config["version"] = version
    model_config["semantic_frame_rate"] = "25hz"

    filter_length = hps.data.filter_length
    segment_size = hps.train.segment_size
    hop_length = hps.data.hop_length
    n_speakers = hps.data.n_speakers

    vits = SynthesizerTrn(
        filter_length // 2 + 1,
        segment_size // hop_length,
        n_speakers=n_speakers,
        **model_config
    )

    if hasattr(vits, "enc_q"):
        del vits.enc_q

    vits.eval()
    vits.load_state_dict(dict_s2["weight"], strict=False)

    print(f"Loaded VITS model: version={version}, filter_length={filter_length}, "
          f"hop_length={hop_length}, sampling_rate={hps.data.sampling_rate}")

    return vits


def export_onnx(vits, output_path):
    """Export VITS decode to ONNX."""
    wrapper = VITSDecodeWrapper(vits)
    wrapper.eval()

    # Dummy inputs
    T_codes = 100
    T_text = 50
    T_refer = 200

    codes = torch.randint(0, 1024, (1, 1, T_codes), dtype=torch.long)
    text = torch.randint(0, 732, (1, T_text), dtype=torch.long)
    refer = torch.randn(1, 704, T_refer, dtype=torch.float32)
    noise_scale = torch.tensor(0.5, dtype=torch.float32)

    print("Exporting to ONNX (legacy dynamo_export=False)...")
    torch.onnx.export(
        wrapper,
        (codes, text, refer, noise_scale),
        output_path,
        input_names=["codes", "text", "refer", "noise_scale"],
        output_names=["audio"],
        dynamic_axes={
            "codes": {2: "T_codes"},
            "text": {1: "T_text"},
            "refer": {2: "T_refer"},
            "audio": {2: "T_audio"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Exported to {output_path}")

    # Validate
    print("Validating ONNX model...")
    import onnxruntime as ort

    sess = ort.InferenceSession(output_path)

    # Run with same inputs
    with torch.no_grad():
        torch_out = wrapper(codes, text, refer, noise_scale).numpy()

    onnx_out = sess.run(None, {
        "codes": codes.numpy(),
        "text": text.numpy(),
        "refer": refer.numpy(),
        "noise_scale": np.array(0.5, dtype=np.float32),
    })[0]

    diff = np.abs(torch_out - onnx_out)
    print(f"Max diff: {diff.max():.6f}, Mean diff: {diff.mean():.6f}")
    print(f"Output shape: {onnx_out.shape}")

    if diff.max() < 0.01:
        print("✅ Validation passed!")
    else:
        print("⚠️  Large difference — check export")


def main():
    parser = argparse.ArgumentParser(description="Export VITS to ONNX")
    parser.add_argument("--moyoyo-tts", required=True,
                        help="Path to moyoyo_tts directory (e.g., .../dora_primespeech/moyoyo_tts)")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    args = parser.parse_args()

    setup_imports(os.path.abspath(args.moyoyo_tts))

    vits = load_vits_model(args.checkpoint)
    export_onnx(vits, args.output)


if __name__ == "__main__":
    main()
