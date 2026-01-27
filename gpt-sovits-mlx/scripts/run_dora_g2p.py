#!/usr/bin/env python3
"""
Standalone script to run dora-primespeech G2P without importing full package.
"""
import os
import sys
import json

# Set environment FIRST
os.environ["PRIMESPEECH_MODEL_DIR"] = os.path.expanduser("~/.dora/models/primespeech")
os.environ["bert_path"] = "hfl/chinese-roberta-wwm-ext-large"  # Use HuggingFace model

# Add text module path directly
MOYOYO_TEXT_PATH = "/Users/yuechen/home/mofa-studio/node-hub/dora-primespeech/dora_primespeech/moyoyo_tts/text"
sys.path.insert(0, os.path.dirname(MOYOYO_TEXT_PATH))

# Block moyoyo_tts top-level import to avoid pytorch_lightning
import types
moyoyo_tts_mock = types.ModuleType("moyoyo_tts")
moyoyo_tts_mock.__path__ = [os.path.dirname(MOYOYO_TEXT_PATH)]
sys.modules["moyoyo_tts"] = moyoyo_tts_mock

# Now import just the text modules
from moyoyo_tts.text.chinese2 import g2p, text_normalize

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_dora_g2p.py <text>", file=sys.stderr)
        sys.exit(1)

    text = sys.argv[1]

    try:
        normalized = text_normalize(text)
        phones, word2ph = g2p(normalized)

        result = {
            "input": text,
            "normalized": normalized,
            "phones": phones,
            "word2ph": word2ph,
        }
        print(json.dumps(result, ensure_ascii=False))
    except Exception as e:
        import traceback
        result = {"error": str(e), "traceback": traceback.format_exc()}
        print(json.dumps(result, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
