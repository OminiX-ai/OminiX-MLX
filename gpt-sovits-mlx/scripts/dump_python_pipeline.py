#!/usr/bin/env python3
"""
Dump all intermediate outputs from the Python GPT-SoVITS pipeline.
Stages: normalization, phones, phone_ids, word2ph, BERT features.

Usage: python scripts/dump_python_pipeline.py "你的文本" [output_dir]
"""
import os
import sys
import json
import numpy as np

# Set environment FIRST
os.environ["PRIMESPEECH_MODEL_DIR"] = os.path.expanduser("~/.dora/models/primespeech")
os.environ["bert_path"] = "hfl/chinese-roberta-wwm-ext-large"

# Add text module path
MOYOYO_PATH = "/Users/yuechen/home/mofa-studio/node-hub/dora-primespeech/dora_primespeech/moyoyo_tts"
MOYOYO_TEXT_PATH = os.path.join(MOYOYO_PATH, "text")
sys.path.insert(0, os.path.dirname(MOYOYO_TEXT_PATH))

# Block moyoyo_tts top-level import to avoid pytorch_lightning
import types
moyoyo_tts_mock = types.ModuleType("moyoyo_tts")
moyoyo_tts_mock.__path__ = [os.path.dirname(MOYOYO_TEXT_PATH)]
sys.modules["moyoyo_tts"] = moyoyo_tts_mock

from moyoyo_tts.text.chinese2 import g2p, text_normalize

def get_phone_ids(phones, version="v2"):
    """Convert phones to IDs using Python's symbol table."""
    sys.path.insert(0, MOYOYO_TEXT_PATH)
    from moyoyo_tts.text import cleaned_text_to_sequence
    return cleaned_text_to_sequence(phones, version)

def extract_bert_features(normalized_text, word2ph):
    """Extract BERT features using the same pipeline as GPT-SoVITS."""
    os.environ['HF_HUB_OFFLINE'] = '1'
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    bert_path = os.environ.get("bert_path", "hfl/chinese-roberta-wwm-ext-large")
    print(f"  Loading BERT model: {bert_path}")
    tokenizer = AutoTokenizer.from_pretrained(bert_path, local_files_only=True)
    bert_model = AutoModelForMaskedLM.from_pretrained(bert_path, local_files_only=True, use_safetensors=False)
    bert_model.eval()

    with torch.no_grad():
        inputs = tokenizer(normalized_text, return_tensors="pt")
        res = bert_model(**inputs, output_hidden_states=True)
        # 3rd-from-last hidden layer, remove CLS and SEP
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]

    assert len(word2ph) == len(normalized_text), \
        f"word2ph length {len(word2ph)} != text length {len(normalized_text)}"

    # Expand to phoneme level using word2ph
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    # Return as [1024, total_phones] matching Python convention
    return phone_level_feature.T.numpy()

def main():
    if len(sys.argv) < 2:
        print("Usage: python dump_python_pipeline.py <text> [output_dir]", file=sys.stderr)
        sys.exit(1)

    text = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/python_pipeline"
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Python Pipeline Dump ===")
    print(f"Input: {text[:80]}...")
    print()

    # Stage 1: Normalization
    normalized = text_normalize(text)
    print(f"[Stage 1] Normalized ({len(normalized)} chars):")
    print(f"  {normalized[:120]}...")
    print()

    # Stage 2: Phones + word2ph
    phones, word2ph = g2p(normalized)
    print(f"[Stage 2] Phones ({len(phones)}):")
    print(f"  {phones[:40]}...")
    print(f"[Stage 2] Word2Ph ({len(word2ph)}):")
    print(f"  {word2ph[:40]}...")
    print(f"  Sum of word2ph: {sum(word2ph)}")
    print(f"  Phones == sum(word2ph): {len(phones) == sum(word2ph)}")
    print()

    # Stage 3: Phone IDs
    try:
        phone_ids = get_phone_ids(phones)
        print(f"[Stage 3] Phone IDs ({len(phone_ids)}):")
        print(f"  {phone_ids[:40]}...")
        print()
    except Exception as e:
        print(f"[Stage 3] Phone IDs: ERROR - {e}")
        phone_ids = []

    # Stage 4: BERT features
    print(f"[Stage 4] BERT Features:")
    try:
        bert_features = extract_bert_features(normalized, word2ph)
        print(f"  Shape: {bert_features.shape}")
        print(f"  dtype: {bert_features.dtype}")
        print(f"  Range: [{bert_features.min():.4f}, {bert_features.max():.4f}]")
        print(f"  Mean: {bert_features.mean():.6f}, Std: {bert_features.std():.4f}")
        print()
    except Exception as e:
        print(f"  ERROR - {e}")
        import traceback; traceback.print_exc()
        bert_features = None

    # Save all outputs
    result = {
        "input": text,
        "normalized": normalized,
        "phones": phones,
        "word2ph": word2ph,
        "phone_ids": phone_ids,
    }

    # Save JSON
    with open(os.path.join(output_dir, "pipeline.json"), "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Save phones as one-per-line for easy diff
    with open(os.path.join(output_dir, "phones.txt"), "w") as f:
        for p in phones:
            f.write(p + "\n")

    with open(os.path.join(output_dir, "phone_ids.txt"), "w") as f:
        for pid in phone_ids:
            f.write(str(pid) + "\n")

    with open(os.path.join(output_dir, "word2ph.txt"), "w") as f:
        for w in word2ph:
            f.write(str(w) + "\n")

    # Save BERT features as numpy
    if bert_features is not None:
        np.save(os.path.join(output_dir, "bert_features.npy"), bert_features)
        print(f"  Saved bert_features.npy ({bert_features.shape})")

    print(f"\nSaved to {output_dir}/")

if __name__ == "__main__":
    main()
