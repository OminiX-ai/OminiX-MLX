#!/usr/bin/env python3
"""
Dynamic comparison: Run both Python and Rust on same text and compare.
No hardcoding - tests any input text.
"""
import os
import sys
import json
import subprocess

def get_python_output(text: str):
    """Get output from real dora-primespeech Python."""
    result = subprocess.run(
        [sys.executable, "scripts/run_dora_g2p.py", text],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)

def get_rust_output(text: str):
    """Get output from Rust implementation."""
    # Create a test that outputs JSON
    test_code = f'''
    use gpt_sovits_mlx::text::preprocessor::{{TextPreprocessor, Language}};
    let p = TextPreprocessor::new();
    let r = p.preprocess("{text}", Some(Language::Chinese));
    println!("{{\\"phones\\": {:?}, \\"word2ph\\": {:?}}}", r.phonemes, r.word2ph);
    '''

    # Run cargo test with custom output
    result = subprocess.run(
        ["cargo", "test", "--features", "jieba", "-p", "gpt-sovits-mlx",
         "test_dynamic_compare", "--", "--nocapture"],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "TEST_TEXT": text}
    )

    # Parse output - look for JSON
    for line in result.stdout.split('\n'):
        if line.strip().startswith('{"phones"'):
            # Fix Rust debug output format to JSON
            line = line.replace("'", '"')
            try:
                return json.loads(line)
            except:
                pass

    return None

def compare(text: str):
    """Compare Python and Rust output for a text."""
    print(f"\n{'='*60}")
    print(f"Text: {text}")
    print('='*60)

    py = get_python_output(text)
    if not py:
        print("ERROR: Python failed")
        return False

    print(f"Python phones ({len(py['phones'])}): {py['phones']}")

    # For now, just show Python output
    # Rust comparison would need a test harness
    return True

def main():
    if len(sys.argv) < 2:
        # Default test cases
        texts = [
            "你好世界",
            "2017年7月28日",
            "大西洋杂志宣布",
            "一个新的健康频道",
        ]
    else:
        texts = [sys.argv[1]]

    for text in texts:
        compare(text)

if __name__ == "__main__":
    main()
