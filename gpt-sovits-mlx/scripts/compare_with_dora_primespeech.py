#!/usr/bin/env python3
"""
Compare Rust gpt-sovits-mlx text preprocessing with real dora-primespeech Python.

This script runs the real GPT-SoVITS Python preprocessing and outputs results
that can be compared with the Rust implementation.

Usage:
    python scripts/compare_with_dora_primespeech.py [--text "你好世界"]
    python scripts/compare_with_dora_primespeech.py --generate-test-cases
"""

import os
import sys
import json
import argparse
import subprocess

DORA_PRIMESPEECH_PATH = "/Users/yuechen/home/mofa-studio/node-hub/dora-primespeech/dora_primespeech"


def get_python_g2p(text: str):
    """Run real dora-primespeech G2P on text using subprocess."""
    # Create a minimal script to run G2P
    script = f'''
import os
import sys
import json

os.environ["PRIMESPEECH_MODEL_DIR"] = os.path.expanduser("~/.dora/models/primespeech")
sys.path.insert(0, "{DORA_PRIMESPEECH_PATH}")

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

try:
    from moyoyo_tts.text.chinese2 import g2p, text_normalize

    text = """{text}"""
    normalized = text_normalize(text)
    phones, word2ph = g2p(normalized)

    result = {{
        "input": text,
        "normalized": normalized,
        "phones": phones,
        "word2ph": word2ph,
    }}
    print(json.dumps(result, ensure_ascii=False))
except Exception as e:
    import traceback
    result = {{"error": str(e), "traceback": traceback.format_exc()}}
    print(json.dumps(result, ensure_ascii=False))
'''

    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
        )

        if result.returncode != 0:
            # Check if there's valid JSON in stdout despite errors
            if result.stdout.strip():
                try:
                    return json.loads(result.stdout.strip().split('\n')[-1])
                except:
                    pass
            return {"error": f"Process failed: {result.stderr[:500]}"}

        # Parse JSON from last line (skip any warnings)
        lines = result.stdout.strip().split('\n')
        for line in reversed(lines):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

        return {"error": "No valid JSON output"}

    except subprocess.TimeoutExpired:
        return {"error": "Timeout"}
    except Exception as e:
        return {"error": str(e)}


def generate_test_cases():
    """Generate test cases for Rust comparison."""
    test_texts = [
        # Basic Chinese
        ("你好", "basic greeting"),
        ("你好世界", "hello world"),

        # Yi sandhi
        ("一百", "yi before tone 3"),
        ("一样", "yi before tone 4"),
        ("看一看", "yi reduplication"),
        ("第一", "ordinal yi"),
        ("一二三", "yi in number sequence"),
        ("一个", "yi ge"),
        ("一个新的", "yi ge phrase"),

        # Bu sandhi
        ("不对", "bu before tone 4"),
        ("不好", "bu before tone 3"),
        ("看不懂", "bu in V不V pattern"),

        # Three-tone sandhi
        ("老虎", "two tone-3"),
        ("展览馆", "three tone-3"),

        # Neutral tone
        ("朋友", "friend - neutral"),
        ("妈妈", "reduplication neutral"),
        ("东西", "thing - neutral"),
        ("我的", "de particle"),
        ("走了", "le aspect marker"),

        # Polyphonic
        ("改为", "wei as 'change to'"),
        ("成为", "wei as 'become'"),
        ("作为", "wei as 'act as'"),

        # Numbers
        ("2017年", "year number"),
        ("100个", "quantity"),

        # Complex
        ("大西洋杂志宣布", "atlantic magazine"),
    ]

    results = []
    for text, desc in test_texts:
        print(f"Processing: {text} ({desc})...")
        result = get_python_g2p(text)
        result["description"] = desc
        results.append(result)

        if "error" not in result:
            print(f"  -> {len(result['phones'])} phones: {result['phones'][:10]}...")
        else:
            print(f"  -> ERROR: {result['error'][:100]}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare with dora-primespeech")
    parser.add_argument("--text", type=str, help="Text to process")
    parser.add_argument("--generate-test-cases", action="store_true",
                        help="Generate test cases for Rust comparison")
    parser.add_argument("--output", type=str, default="tests/fixtures/python_expected.json",
                        help="Output file for test cases")
    args = parser.parse_args()

    if args.generate_test_cases:
        print("Generating test cases from real dora-primespeech Python...")
        print("="*60)
        results = generate_test_cases()

        # Save to JSON
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "..", args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print("="*60)
        print(f"Saved {len(results)} test cases to {output_path}")

        # Print summary
        success = sum(1 for r in results if "error" not in r)
        print(f"Success: {success}/{len(results)}")

    elif args.text:
        result = get_python_g2p(args.text)

        if "error" in result:
            print(f"Error: {result['error']}")
            if "traceback" in result:
                print(result["traceback"])
            return 1

        print("="*60)
        print("DORA-PRIMESPEECH PYTHON OUTPUT")
        print("="*60)
        print(f"Input: {result['input']}")
        print(f"Normalized ({len(result['normalized'])} chars): {result['normalized']}")
        print(f"Phones ({len(result['phones'])}): {result['phones']}")
        print(f"Word2Ph ({len(result['word2ph'])}): {result['word2ph']}")

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
