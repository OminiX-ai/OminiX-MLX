#!/bin/bash
# Dynamic comparison: Run Python and Rust on same text, compare output
# Usage: ./scripts/compare.sh "你的文本"

TEXT="${1:-你好世界}"

echo "=== Comparing: $TEXT ==="
echo ""

# Python output
echo "--- Python (dora-primespeech) ---"
PY_OUT=$(python3 scripts/run_dora_g2p.py "$TEXT" 2>/dev/null)
PY_PHONES=$(echo "$PY_OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['phones'])" 2>/dev/null)
echo "Phones: $PY_PHONES"

# Rust output - run inline test
echo ""
echo "--- Rust (gpt-sovits-mlx) ---"
RS_OUT=$(cargo test --features jieba -p gpt-sovits-mlx test_inline_text -- --nocapture 2>&1 | grep -A1 "TEST_OUTPUT" | tail -1)

# For now, use the existing test infrastructure
# Create temp test file
cat > /tmp/test_compare.rs << EOF
#[test]
fn test_inline_text() {
    use gpt_sovits_mlx::text::preprocessor::{chinese_g2p, normalize_chinese};
    let text = "$TEXT";
    let norm = normalize_chinese(text);
    let (phones, word2ph) = chinese_g2p(&norm);
    println!("TEST_OUTPUT");
    println!("{:?}", phones);
}
EOF

# Actually just run a quick Rust command
echo "Running Rust..."
cd /Users/yuechen/home/OminiX-MLX/gpt-sovits-mlx

# Use cargo to run a simple test
cargo test --features jieba -p gpt-sovits-mlx test_chinese_g2p -- --nocapture 2>&1 | head -20
