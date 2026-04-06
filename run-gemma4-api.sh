#!/usr/bin/env bash
# Run OminiX-API with Gemma4 E4B model
#
# Usage:
#   ./run-gemma4-api.sh                          # defaults: E4B, port 18080
#   ./run-gemma4-api.sh models/gemma-4-26B-A4B-it  # use 26B model
#   PORT=8080 ./run-gemma4-api.sh                # custom port
#
# Test:
#   curl http://localhost:18080/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{"model":"gemma4","messages":[{"role":"user","content":"Hello!"}]}'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
API_DIR="${SCRIPT_DIR}/../OminiX-API"
MODEL_DIR="${1:-${SCRIPT_DIR}/models/gemma-4-E4B-it}"

export PORT="${PORT:-18080}"
export LLM_MODEL="${MODEL_DIR}"

if [ ! -d "${API_DIR}" ]; then
    echo "Error: OminiX-API not found at ${API_DIR}" >&2
    exit 1
fi

if [ ! -f "${MODEL_DIR}/config.json" ]; then
    echo "Error: Model not found at ${MODEL_DIR}" >&2
    exit 1
fi

echo "Starting OminiX-API on port ${PORT} with model: ${MODEL_DIR}"
cd "${API_DIR}" && exec cargo run --release
