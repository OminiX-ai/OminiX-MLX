#!/usr/bin/env python3
"""Compare Python vs Rust preprocessing - using direct path approach"""
import os
import sys

# Add primespeech to path properly
sys.path.insert(0, "/Users/yuechen/home/mofa-studio/node-hub/dora-primespeech")

from dora_primespeech.moyoyo_tts.text.chinese import g2p, text_normalize

def test_python_preprocessing(text: str):
    """Run Python preprocessing and return results."""
    print(f"\n{'='*60}")
    print(f"Input: {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"{'='*60}")

    # 1. Text normalization
    norm_text = text_normalize(text)
    print(f"\n[Python] Normalized: {norm_text[:100]}{'...' if len(norm_text) > 100 else ''}")

    # 2. G2P conversion
    phones, word2ph = g2p(norm_text)
    print(f"\n[Python] Phonemes ({len(phones)}): {phones}")
    print(f"[Python] Word2Ph ({len(word2ph)}): {word2ph}")

    return {
        "norm_text": norm_text,
        "phones": phones,
        "word2ph": word2ph,
    }

# Test cases
TEST_CASES = [
    "你好",
    "你好世界",
    "一百",
    "一样",
    "看一看",
    "第一",
    "一二三",
    "不对",
    "不好",
    "看不懂",
    "老虎",
    "展览馆",
    "朋友",
    "妈妈",
    "东西",
    # User's mixed test case
    '1845年，在英国"铁路狂热"时期，该报一度与《银行公报》（Bankers\' Gazette）及《铁路观察》（Railway Monitor）合并，并将刊名改为《经济学人：商业周报、银行公报及铁路观察——政治化和文学化的大众报纸》（The Economist, Weekly Commercial Times, Bankers\' Gazette, and Railway Monitor. A Political, Literary and General Newspaper）。',
]

if __name__ == "__main__":
    print("Python Preprocessing Test Results")
    print("=" * 60)

    results = {}
    for text in TEST_CASES:
        try:
            results[text] = test_python_preprocessing(text)
        except Exception as e:
            print(f"\n[Error] {text[:50]}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for text, result in results.items():
        print(f"\nInput: {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"  Phones: {result['phones']}")
