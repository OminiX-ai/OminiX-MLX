#!/usr/bin/env python3
"""
Dynamic Python vs Rust comparison - no hardcoding.
Both implementations process text dynamically.

Usage:
    python scripts/compare.py "你的文本"
    python scripts/compare.py  # runs default test suite
"""
import subprocess
import json
import sys

def get_python(text):
    """Run real dora-primespeech Python."""
    r = subprocess.run(
        [sys.executable, "scripts/run_dora_g2p.py", text],
        capture_output=True, text=True, timeout=120
    )
    return json.loads(r.stdout) if r.returncode == 0 else None

def get_rust(text):
    """Run Rust G2P binary."""
    r = subprocess.run(
        ["cargo", "run", "--features", "jieba", "-p", "gpt-sovits-mlx", "--bin", "g2p", "--", text],
        capture_output=True, text=True, timeout=120
    )
    return json.loads(r.stdout) if r.returncode == 0 else None

def compare(text):
    """Compare Python and Rust output."""
    py = get_python(text)
    rs = get_rust(text)

    if not py or not rs:
        print(f"ERROR: {text}")
        return False

    match = py['phones'] == rs['phones']
    status = "✓" if match else "✗"

    print(f"{status} {text}")
    if not match:
        print(f"  Python: {py['phones']}")
        print(f"  Rust:   {rs['phones']}")

    return match

def main():
    if len(sys.argv) > 1:
        texts = [sys.argv[1]]
    else:
        texts = [
            # Basic tests
            "你好", "你好世界", "杂志", "大西洋杂志",
            # Yi sandhi
            "一个", "一样", "一百", "看一看",
            # Bu sandhi
            "不对", "不好", "看不懂",
            # Tone 3 sandhi
            "老虎", "展览馆",
            # Polyphones
            "改为", "成为", "作为",
            # Neutral tone
            "朋友", "妈妈", "东西",
            # Complex texts
            "亿万富翁投资者",
            "前苹果公司总裁",
            "爱默生集团",
            # Erhua - MUST_ERHUA (inherit tone)
            "小院儿", "胡同儿",
            # Erhua - NOT_ERHUA (keep er2)
            "女儿", "花儿", "婴儿",
            # Standalone 儿
            "儿",
        ]

    print("=== Python vs Rust Comparison ===\n")
    matches = sum(1 for t in texts if compare(t))
    print(f"\n{matches}/{len(texts)} identical")

if __name__ == "__main__":
    main()
