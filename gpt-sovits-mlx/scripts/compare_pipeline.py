#!/usr/bin/env python3
"""
Compare Python vs Rust pipeline outputs at every stage.

Usage: python scripts/compare_pipeline.py [python_dir] [rust_dir]
"""
import sys
import json
import numpy as np
from pathlib import Path


def compare_text_files(py_path, rs_path, label):
    """Compare line-by-line text files."""
    py_lines = Path(py_path).read_text().strip().split('\n')
    rs_lines = Path(rs_path).read_text().strip().split('\n')

    if py_lines == rs_lines:
        print(f"  {label}: IDENTICAL ({len(py_lines)} entries)")
        return True

    mismatches = sum(1 for a, b in zip(py_lines, rs_lines) if a != b)
    len_diff = abs(len(py_lines) - len(rs_lines))
    print(f"  {label}: {mismatches} mismatches, {len_diff} length diff "
          f"(py={len(py_lines)}, rs={len(rs_lines)})")

    # Show first 5 diffs
    shown = 0
    for i, (a, b) in enumerate(zip(py_lines, rs_lines)):
        if a != b and shown < 5:
            print(f"    [{i+1}] py={a!r} rs={b!r}")
            shown += 1
    return False


def compare_bert_features(py_path, rs_path):
    """Compare BERT feature arrays."""
    py = np.load(py_path)
    rs = np.load(rs_path)

    print(f"  Shape: py={py.shape} rs={rs.shape}")

    if py.shape != rs.shape:
        print(f"  SHAPE MISMATCH!")
        return False

    diff = np.abs(py - rs)
    max_diff = diff.max()
    mean_diff = diff.mean()

    # Cosine similarity per phone position
    # Shape is [1024, total_phones]
    total_phones = py.shape[1]
    cos_sims = []
    for i in range(total_phones):
        a = py[:, i]
        b = rs[:, i]
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        cos_sims.append(cos_sim)

    min_cos = min(cos_sims)
    mean_cos = np.mean(cos_sims)

    print(f"  Max abs diff: {max_diff:.6e}")
    print(f"  Mean abs diff: {mean_diff:.6e}")
    print(f"  Cosine similarity: min={min_cos:.6f}, mean={mean_cos:.6f}")

    if max_diff < 1e-4:
        print(f"  VERDICT: IDENTICAL (within float32 precision)")
        return True
    elif min_cos > 0.999:
        print(f"  VERDICT: NEAR-IDENTICAL (cosine > 0.999)")
        return True
    else:
        print(f"  VERDICT: DIFFERENT")
        # Show worst positions
        worst = sorted(range(total_phones), key=lambda i: cos_sims[i])[:5]
        for idx in worst:
            print(f"    Phone [{idx}]: cosine={cos_sims[idx]:.6f}, "
                  f"max_diff={np.abs(py[:, idx] - rs[:, idx]).max():.4f}")
        return False


def main():
    py_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/python_pipeline")
    rs_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/rust_pipeline")

    print("=" * 60)
    print("Pipeline Comparison: Python vs Rust")
    print("=" * 60)

    # Stage 1: Normalized text
    print("\n[Stage 1] Normalized Text:")
    py_json = json.loads((py_dir / "pipeline.json").read_text())
    # Try to get Rust normalized from JSON if available
    rs_json_path = rs_dir / "pipeline.json"
    if rs_json_path.exists():
        # Not available yet in Rust dump, skip
        pass
    py_norm = py_json["normalized"]
    print(f"  Python: {py_norm[:80]}...")

    # Stage 2: Phones
    print("\n[Stage 2] Phones:")
    all_match = compare_text_files(py_dir / "phones.txt", rs_dir / "phones.txt", "Phones")

    # Stage 3: Phone IDs
    print("\n[Stage 3] Phone IDs:")
    compare_text_files(py_dir / "phone_ids.txt", rs_dir / "phone_ids.txt", "Phone IDs")

    # Word2Ph
    print("\n[Stage 2b] Word2Ph:")
    compare_text_files(py_dir / "word2ph.txt", rs_dir / "word2ph.txt", "Word2Ph")

    # Stage 4: BERT Features
    print("\n[Stage 4] BERT Features:")
    py_bert = py_dir / "bert_features.npy"
    rs_bert = rs_dir / "bert_features.npy"
    if py_bert.exists() and rs_bert.exists():
        compare_bert_features(py_bert, rs_bert)
    else:
        missing = []
        if not py_bert.exists(): missing.append("Python")
        if not rs_bert.exists(): missing.append("Rust")
        print(f"  SKIP: {' and '.join(missing)} BERT features not found")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
