#!/usr/bin/env python3
"""
Dump full TTS pipeline outputs using the original PyTorch GPT-SoVITS.
Runs T2S with greedy decoding and VITS vocoding.

Usage: python scripts/dump_full_pipeline.py "你好世界" [output_dir]
"""
import os
import sys
import json
import numpy as np

os.environ["PRIMESPEECH_MODEL_DIR"] = os.path.expanduser("~/.dora/models/primespeech")
os.environ["bert_path"] = "hfl/chinese-roberta-wwm-ext-large"
os.environ["HF_HUB_OFFLINE"] = "1"

MOYOYO_PATH = "/Users/yuechen/home/mofa-studio/node-hub/dora-primespeech/dora_primespeech/moyoyo_tts"
sys.path.insert(0, os.path.dirname(MOYOYO_PATH))

# Mock modules to avoid broken torchaudio/pytorch_lightning chain
import types

# Mock torchaudio before anything imports it (broken dylib)
import importlib
torchaudio_mock = types.ModuleType("torchaudio")
torchaudio_mock.__version__ = "0.0.0"
torchaudio_mock.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)
sys.modules["torchaudio"] = torchaudio_mock

moyoyo_mock = types.ModuleType("moyoyo_tts")
moyoyo_mock.__path__ = [MOYOYO_PATH]
sys.modules["moyoyo_tts"] = moyoyo_mock

import torch

# Fix missing typing imports in moyoyo_tts modules (Python 3.12 compat)
import builtins
from typing import Tuple, Optional
builtins.Tuple = Tuple
builtins.Optional = Optional
from moyoyo_tts.text.chinese2 import g2p, text_normalize


def get_phone_ids(phones, version="v2"):
    sys.path.insert(0, os.path.join(MOYOYO_PATH, "text"))
    from moyoyo_tts.text import cleaned_text_to_sequence
    return cleaned_text_to_sequence(phones, version)


def extract_bert_features(text, word2ph):
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    bert_path = os.environ.get("bert_path", "hfl/chinese-roberta-wwm-ext-large")
    tokenizer = AutoTokenizer.from_pretrained(bert_path, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(bert_path, local_files_only=True, use_safetensors=False)
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    phone_level = []
    for i in range(len(word2ph)):
        phone_level.append(res[i].repeat(word2ph[i], 1))
    return torch.cat(phone_level, dim=0).T  # [1024, total_phones]


def load_t2s_model(ckpt_path, device="cpu"):
    """Load the T2S model from PyTorch checkpoint."""
    sys.path.insert(0, MOYOYO_PATH)
    from moyoyo_tts.AR.models.t2s_lightning_module import Text2SemanticLightningModule
    dict_s1 = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = dict_s1["config"]
    model = Text2SemanticLightningModule(config, "****", is_train=False)
    model.load_state_dict(dict_s1["weight"])
    model = model.to(device).eval()
    return model, config


def load_vits_model(weights_path, device="cpu"):
    """Load the VITS model from PyTorch checkpoint."""
    from moyoyo_tts.module.models import SynthesizerTrn
    dict_s2 = torch.load(weights_path, map_location="cpu", weights_only=False)
    hps = dict_s2["config"]
    if dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
        version = "v1"
    else:
        version = "v2"
    hps["model"]["version"] = version
    kwargs = hps["model"]
    vits = SynthesizerTrn(
        hps["data"]["filter_length"] // 2 + 1,
        hps["train"]["segment_size"] // hps["data"]["hop_length"],
        n_speakers=hps["data"]["n_speakers"],
        **kwargs
    )
    if hasattr(vits, "enc_q"):
        del vits.enc_q
    vits = vits.to(device).eval()
    vits.load_state_dict(dict_s2["weight"], strict=False)
    return vits, hps


def load_hubert_model(device="cpu"):
    """Load CNHuBERT for reference audio encoding."""
    from moyoyo_tts.feature_extractor import cnhubert
    cnhubert.cnhubert_base_path = os.path.join(
        os.environ["PRIMESPEECH_MODEL_DIR"],
        "moyoyo/chinese-hubert-base"
    )
    ssl_model = cnhubert.get_model()
    ssl_model = ssl_model.to(device).eval()
    return ssl_model


def get_ref_spec(ref_audio_path, hps, device="cpu"):
    """Get reference spectrogram from audio."""
    from moyoyo_tts.module.mel_processing import spectrogram_torch
    import librosa
    audio, _ = librosa.load(ref_audio_path, sr=int(hps["data"]["sampling_rate"]))
    audio = torch.FloatTensor(audio).unsqueeze(0).to(device)
    spec = spectrogram_torch(
        audio,
        hps["data"]["filter_length"],
        hps["data"]["sampling_rate"],
        hps["data"]["hop_length"],
        hps["data"]["win_length"],
        center=False,
    )
    return spec


def get_prompt_semantic(ref_audio_path, ssl_model, device="cpu"):
    """Extract semantic codes from reference audio using HuBERT."""
    import librosa
    audio, _ = librosa.load(ref_audio_path, sr=16000)
    audio = torch.FloatTensor(audio).unsqueeze(0).to(device)
    with torch.no_grad():
        ssl_content = ssl_model.model(audio)["last_hidden_state"].transpose(1, 2)
    # Use pre-computed codes instead
    return None  # Will use pre-computed


def run_t2s_greedy(t2s_model, phoneme_ids, bert_features, prompt_semantic=None, device="cpu"):
    """Run T2S with greedy decoding (argmax)."""
    with torch.no_grad():
        # Use the model's infer method with temperature near 0
        pred_semantic, idx = t2s_model.model.infer_panel(
            phoneme_ids.to(device),
            torch.LongTensor([phoneme_ids.shape[-1]]).to(device),
            prompt_semantic.to(device) if prompt_semantic is not None else None,
            bert_features.to(device),
            top_k=1,           # Only take the top-1 token (greedy)
            top_p=1.0,
            temperature=0.001,  # Near-zero for determinism
            early_stop_num=-1,
        )
    return pred_semantic, idx


def main():
    if len(sys.argv) < 2:
        print("Usage: python dump_full_pipeline.py <text> [output_dir]", file=sys.stderr)
        sys.exit(1)

    text = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/python_pipeline"
    os.makedirs(output_dir, exist_ok=True)

    device = "cpu"  # Use CPU for reproducibility

    print("=== Python Full Pipeline Dump (PyTorch) ===")
    print(f"Input: {text[:80]}...")
    print()

    # Stage 1: Text preprocessing
    # Match original inference_webui.py: append period if text doesn't end with punctuation
    splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…"}
    if text[-1] not in splits:
        text += "。"
    print(f"  (After period append: {text[:80]})")
    normalized = text_normalize(text)
    phones, word2ph = g2p(normalized)
    phone_ids = get_phone_ids(phones)
    print(f"[Stage 1-3] Normalized: {len(normalized)} chars, Phones: {len(phones)}, IDs: {len(phone_ids)}")

    # Stage 4: BERT features
    print("[Stage 4] Extracting BERT features...")
    bert_features = extract_bert_features(normalized, word2ph)
    print(f"  BERT shape: {bert_features.shape}")

    # Stage 5: Load T2S model
    gpt_path = os.path.expanduser("~/.dora/models/primespeech/moyoyo/GPT_weights/doubao-mixed.ckpt")
    print(f"[Stage 5] Loading T2S model: {gpt_path}")
    t2s_model, t2s_config = load_t2s_model(gpt_path, device)
    print(f"  T2S config: hz={t2s_config.get('data', {}).get('hz', 'N/A')}")

    # Prepare T2S inputs
    # For zero-shot (no reference): just use target phonemes + BERT
    phoneme_ids_tensor = torch.LongTensor(phone_ids).unsqueeze(0)  # [1, seq]
    bert_features_tensor = bert_features.unsqueeze(0)  # [1, 1024, seq]

    # Load pre-computed prompt semantic codes for few-shot
    codes_path = "~/.dora/models/primespeech/gpt-sovits-mlx/doubao_mixed_codes.bin"
    if os.path.exists(codes_path):
        codes_data = open(codes_path, "rb").read()
        prompt_codes = np.frombuffer(codes_data, dtype=np.int32)
        prompt_semantic = torch.from_numpy(prompt_codes).long().unsqueeze(0)  # [1, 145]
        print(f"  Prompt semantic: {prompt_semantic.shape} ({prompt_codes[:5]}...)")

        # For few-shot, we need reference text phonemes + BERT too
        ref_text = "这家resturant的steak很有名，但是vegetable salad的price有点贵"
        ref_normalized = text_normalize(ref_text)
        ref_phones, ref_word2ph = g2p(ref_normalized)
        ref_phone_ids = get_phone_ids(ref_phones)
        ref_bert = extract_bert_features(ref_normalized, ref_word2ph)

        # Combine ref + target
        all_phone_ids = ref_phone_ids + phone_ids
        all_bert = torch.cat([ref_bert, bert_features], dim=1)  # [1024, ref+target]

        phoneme_ids_tensor = torch.LongTensor(all_phone_ids).unsqueeze(0)
        bert_features_tensor = all_bert.unsqueeze(0)  # [1, 1024, total]
        print(f"  Combined phonemes: {phoneme_ids_tensor.shape}, BERT: {bert_features_tensor.shape}")
    else:
        prompt_semantic = None
        print("  No prompt codes found, using zero-shot mode")

    # Stage 6: T2S greedy decoding
    print("[Stage 6] Running T2S greedy decoding...")
    pred_semantic, idx = run_t2s_greedy(
        t2s_model, phoneme_ids_tensor, bert_features_tensor,
        prompt_semantic=prompt_semantic, device=device,
    )
    pred_semantic = pred_semantic.cpu()
    print(f"  T2S output shape: {pred_semantic.shape}, idx={idx}")

    # Extract only the newly generated tokens (exclude prompt)
    if prompt_semantic is not None:
        prompt_len = prompt_semantic.shape[-1]
        new_tokens = pred_semantic[:, prompt_len:]
    else:
        new_tokens = pred_semantic

    semantic_tokens = new_tokens.squeeze(0).numpy().astype(np.int32)
    print(f"  New semantic tokens: {len(semantic_tokens)}")
    print(f"  First 20: {semantic_tokens[:20].tolist()}")

    # Stage 7: VITS vocoding
    sovits_path = os.path.expanduser("~/.dora/models/primespeech/moyoyo/SoVITS_weights/doubao-mixed.pth")
    print(f"[Stage 7] Loading VITS model: {sovits_path}")
    vits_model, hps = load_vits_model(sovits_path, device)

    # Get reference spectrogram
    ref_audio_path = os.path.expanduser("~/.dora/models/primespeech/moyoyo/ref_audios/doubao_ref_mix_new.wav")
    print(f"  Loading reference audio: {ref_audio_path}")
    refer_spec = get_ref_spec(ref_audio_path, hps, device)
    print(f"  Reference spec shape: {refer_spec.shape}")

    # Prepare VITS inputs
    # decode(codes, text, refer, noise_scale=0.5, speed=1)
    # codes: [1, 1, semantic_len], text: [1, phone_len], refer: [1, freq, time]
    target_phone_ids_tensor = torch.LongTensor(phone_ids).unsqueeze(0).to(device)
    semantic_codes = torch.LongTensor(semantic_tokens).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, seq]

    # Run VITS decode
    print("  Running VITS vocoding...")
    with torch.no_grad():
        audio = vits_model.decode(
            semantic_codes,
            target_phone_ids_tensor,
            refer_spec,
            noise_scale=0.5,
        )
    audio_np = audio.squeeze().cpu().numpy()
    sample_rate = hps["data"]["sampling_rate"]
    print(f"  Audio shape: {audio_np.shape}, sample_rate: {sample_rate}")
    print(f"  Duration: {len(audio_np) / sample_rate:.2f}s")

    # Save outputs
    np.save(os.path.join(output_dir, "bert_features.npy"), bert_features.numpy())
    np.save(os.path.join(output_dir, "semantic_tokens.npy"), semantic_tokens)
    np.save(os.path.join(output_dir, "audio.npy"), audio_np)

    # Save WAV
    import soundfile as sf
    wav_path = os.path.join(output_dir, "output.wav")
    sf.write(wav_path, audio_np, sample_rate)
    print(f"\n  Saved: bert_features.npy, semantic_tokens.npy, audio.npy, output.wav")

    # Save text outputs
    with open(os.path.join(output_dir, "phones.txt"), "w") as f:
        for p in phones: f.write(p + "\n")
    with open(os.path.join(output_dir, "phone_ids.txt"), "w") as f:
        for pid in phone_ids: f.write(str(pid) + "\n")
    with open(os.path.join(output_dir, "word2ph.txt"), "w") as f:
        for w in word2ph: f.write(str(w) + "\n")
    with open(os.path.join(output_dir, "semantic_tokens.txt"), "w") as f:
        for t in semantic_tokens: f.write(str(t) + "\n")

    result = {
        "input": text,
        "normalized": normalized,
        "phones": phones,
        "word2ph": word2ph,
        "phone_ids": phone_ids,
        "semantic_tokens": semantic_tokens.tolist(),
        "audio_samples": len(audio_np),
        "sample_rate": sample_rate,
    }
    with open(os.path.join(output_dir, "pipeline.json"), "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nSaved all outputs to {output_dir}/")


if __name__ == "__main__":
    main()
