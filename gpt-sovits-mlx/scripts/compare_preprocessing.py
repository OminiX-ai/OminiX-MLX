#!/usr/bin/env python3
"""
Compare Python (primespeech) vs Rust (gpt-sovits-mlx) text preprocessing.
"""

import sys
import os

# Add primespeech to path
sys.path.insert(0, "/Users/yuechen/home/mofa-studio/node-hub/dora-primespeech")

from dora_primespeech.moyoyo_tts.text.chinese2 import g2p, text_normalize
from dora_primespeech.moyoyo_tts.text.cleaner import clean_text

def test_python_preprocessing(text: str):
    """Run Python preprocessing and return results."""
    print(f"\n{'='*60}")
    print(f"Input: {text}")
    print(f"{'='*60}")

    # 1. Text normalization
    norm_text = text_normalize(text)
    print(f"\n[Python] Normalized text: {norm_text}")

    # 2. G2P conversion
    phones, word2ph = g2p(norm_text)
    print(f"\n[Python] Phonemes ({len(phones)}): {phones}")
    print(f"[Python] Word2Ph ({len(word2ph)}): {word2ph}")

    # 3. Full clean_text (includes tone sandhi)
    try:
        full_phones, full_word2ph, full_norm = clean_text(text, "zh", "v2")
        print(f"\n[Python] Full phones ({len(full_phones)}): {full_phones}")
        print(f"[Python] Full word2ph ({len(full_word2ph)}): {full_word2ph}")
        print(f"[Python] Full norm_text: {full_norm}")
    except Exception as e:
        print(f"[Python] clean_text error: {e}")

    return {
        "norm_text": norm_text,
        "phones": phones,
        "word2ph": word2ph,
    }

# Test cases - focusing on tone sandhi
TEST_CASES = [
    # Basic Chinese
    "你好",
    "你好世界",

    # Yi (一) sandhi
    "一百",      # 一 before tone 3 → yi4
    "一样",      # 一 before tone 4 → yi2
    "看一看",    # 一 in X一X → yi5
    "第一",      # Ordinal - keep yi1
    "一二三",    # Number sequence - keep yi1

    # Bu (不) sandhi
    "不对",      # 不 before tone 4 → bu2
    "不好",      # 不 before tone 3 → bu4
    "看不懂",    # X不X pattern → bu5

    # Three tone sandhi
    "你好",      # Two tone 3 → first becomes tone 2
    "老虎",      # Two tone 3 words
    "展览馆",    # Three consecutive tone 3

    # Neutral tone words
    "朋友",      # Must-have neutral tone
    "妈妈",      # Reduplication → neutral on second
    "东西",      # Common neutral tone word

    # Mixed content
    "我有100元",
    "2024年1月1日",
    "温度是25°C",
]

if __name__ == "__main__":
    print("Python Preprocessing Test Results")
    print("=" * 60)

    results = {}
    for text in TEST_CASES:
        try:
            results[text] = test_python_preprocessing(text)
        except Exception as e:
            print(f"\n[Error] {text}: {e}")
            import traceback
            traceback.print_exc()

    # Output summary for comparison
    print("\n" + "=" * 60)
    print("SUMMARY FOR RUST COMPARISON")
    print("=" * 60)

    for text, result in results.items():
        print(f"\nInput: {text}")
        print(f"  Phones: {result['phones']}")
        print(f"  Word2Ph: {result['word2ph']}")
