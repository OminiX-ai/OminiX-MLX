#!/usr/bin/env bash
# End-to-end test for Gemma4 chat via OminiX-API using Goose AI CLI.
#
# Prerequisites:
#   - goose CLI installed (brew install goose or similar)
#   - Gemma4 model weights at models/gemma-4-E4B-it/ (or 26B-A4B)
#   - OminiX-API repo at ../OminiX-API (sibling directory)
#
# Usage:
#   ./gemma4-mlx/tests/e2e_goose.sh [model_dir] [port]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
API_DIR="${ROOT_DIR}/../OminiX-API"

MODEL_DIR="${1:-${ROOT_DIR}/models/gemma-4-E4B-it}"
PORT="${2:-18093}"
API_URL="http://localhost:${PORT}"
PASS=0
FAIL=0
ERRORS=""

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

pass() { PASS=$((PASS + 1)); echo -e "  ${GREEN}PASS${NC}: $1"; }
fail() { FAIL=$((FAIL + 1)); ERRORS="${ERRORS}\n  - $1"; echo -e "  ${RED}FAIL${NC}: $1"; }

cleanup() {
    if [ -n "${SERVER_PID:-}" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "=== Gemma4 E2E Tests ==="
echo "Model: $MODEL_DIR"
echo "Port:  $PORT"
echo "API:   $API_DIR"
echo ""

# ── Validate ─────────────────────────────────────────────────────────────────
if [ ! -d "$API_DIR" ]; then
    echo "Error: OminiX-API not found at $API_DIR" >&2
    exit 1
fi
if [ ! -f "${MODEL_DIR}/config.json" ]; then
    echo "Error: Model not found at $MODEL_DIR" >&2
    exit 1
fi

# ── Build ────────────────────────────────────────────────────────────────────
echo "Building OminiX-API..."
(cd "$API_DIR" && cargo build --release 2>/dev/null)

# ── Start server ─────────────────────────────────────────────────────────────
echo "Starting server..."
PORT="$PORT" LLM_MODEL="$MODEL_DIR" \
    "$API_DIR/target/release/ominix-api" 2>/dev/null &
SERVER_PID=$!

for i in $(seq 1 120); do
    if curl -s "${API_URL}/health" > /dev/null 2>&1; then
        echo "Server ready (${i}s)"
        break
    fi
    sleep 1
    if [ "$i" -eq 120 ]; then
        echo "Server failed to start after 120s"
        exit 1
    fi
done
echo ""

# ── Test 1: Non-streaming math ───────────────────────────────────────────────
echo "Test 1: Non-streaming math"
RESP=$(curl -s "${API_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4","messages":[{"role":"user","content":"What is 7 * 8? Reply with just the number."}],"temperature":0,"max_tokens":32}')

CONTENT=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "PARSE_ERROR")
if echo "$CONTENT" | grep -q "56"; then
    pass "Non-streaming math: got 56"
else
    fail "Non-streaming math: expected 56, got: $CONTENT"
fi

# ── Test 2: Streaming completeness ───────────────────────────────────────────
echo "Test 2: Streaming completeness"
STREAM_OUT=$(curl -s -N "${API_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4","messages":[{"role":"user","content":"Say exactly: Hello World"}],"stream":true,"max_tokens":32}' 2>&1)

if echo "$STREAM_OUT" | grep -q '"finish_reason":"stop"'; then
    pass "Streaming ends with finish_reason=stop"
else
    fail "Streaming missing finish_reason=stop"
fi
if echo "$STREAM_OUT" | grep -q 'data: \[DONE\]'; then
    pass "Streaming ends with [DONE]"
else
    fail "Streaming missing [DONE]"
fi
if echo "$STREAM_OUT" | grep -q '<|channel>'; then
    fail "Streaming leaked <|channel> token"
else
    pass "Streaming: no control token leaks"
fi

# ── Test 3: System prompt ────────────────────────────────────────────────────
echo "Test 3: System prompt"
RESP=$(curl -s "${API_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4","messages":[{"role":"system","content":"You are a pirate. Always say Arrr."},{"role":"user","content":"Hello"}],"temperature":0,"max_tokens":64}')

CONTENT=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "PARSE_ERROR")
if echo "$CONTENT" | grep -qi "arrr\|ahoy\|matey\|pirate"; then
    pass "System prompt: pirate persona detected"
else
    fail "System prompt: expected pirate, got: $CONTENT"
fi

# ── Test 4: Goose AI (if available) ──────────────────────────────────────────
if command -v goose &>/dev/null; then
    echo "Test 4: Goose AI simple prompt"
    GOOSE_OUT=$(OPENAI_API_KEY=unused OPENAI_HOST="${API_URL}" OPENAI_BASE_PATH="v1/chat/completions" \
        goose run -q --provider openai --model gemma4 \
        -t "What is the capital of France? Reply with just the city name." 2>&1 || true)

    if echo "$GOOSE_OUT" | grep -qi "paris"; then
        pass "Goose: got Paris"
    else
        fail "Goose: expected Paris, got: $(echo "$GOOSE_OUT" | head -3)"
    fi
else
    echo "Test 4: Goose AI — SKIPPED (goose CLI not found)"
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
if [ "$FAIL" -gt 0 ]; then
    echo -e "Failures:${ERRORS}"
    exit 1
fi
