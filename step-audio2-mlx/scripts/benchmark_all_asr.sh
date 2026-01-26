#!/bin/bash
# Unified ASR Benchmark Script
# Compares step-audio2-mlx, funasr-mlx, and funasr-nano-mlx
#
# Usage:
#   ./scripts/benchmark_all_asr.sh <audio.wav> [iterations]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
AUDIO_FILE="${1:-/Users/yuechen/home/OminiX-MLX/funasr-nano-mlx/Fun-ASR-Nano-2512/example/zh.wav}"
ITERATIONS="${2:-10}"

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(dirname "$PROJECT_ROOT")"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              UNIFIED ASR BENCHMARK - MLX Implementations             ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Audio file: ${YELLOW}$AUDIO_FILE${NC}"
echo -e "Iterations: ${YELLOW}$ITERATIONS${NC}"
echo ""

# Check audio file
if [ ! -f "$AUDIO_FILE" ]; then
    echo -e "${RED}Error: Audio file not found: $AUDIO_FILE${NC}"
    exit 1
fi

# Get audio duration
AUDIO_DURATION=$(python3 -c "
import wave
with wave.open('$AUDIO_FILE', 'rb') as w:
    frames = w.getnframes()
    rate = w.getframerate()
    print(f'{frames/rate:.2f}')
" 2>/dev/null || echo "unknown")

echo -e "Audio duration: ${YELLOW}${AUDIO_DURATION}s${NC}"
echo ""

# Results storage
declare -A RESULTS
declare -A RTF

echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Benchmark 1: step-audio2-mlx (Whisper encoder + Qwen2.5-7B)${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""

cd "$PROJECT_ROOT"
if [ -d "$PROJECT_ROOT/Step-Audio-2-mini" ] || [ -d "$WORKSPACE_ROOT/Step-Audio-2-mini" ]; then
    MODEL_DIR="$PROJECT_ROOT/Step-Audio-2-mini"
    if [ ! -d "$MODEL_DIR" ]; then
        MODEL_DIR="$WORKSPACE_ROOT/Step-Audio-2-mini"
    fi
    echo "Model found at: $MODEL_DIR"
    cargo run --release --example benchmark_asr -- "$AUDIO_FILE" "$ITERATIONS" 2>&1 || echo "step-audio2-mlx benchmark failed or model not loaded"
else
    echo -e "${YELLOW}Skipping: Step-Audio-2-mini model not found${NC}"
    echo "Expected at: $PROJECT_ROOT/Step-Audio-2-mini or $WORKSPACE_ROOT/Step-Audio-2-mini"
fi
echo ""

echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Benchmark 2: funasr-mlx (Paraformer)${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""

FUNASR_DIR="$WORKSPACE_ROOT/funasr-mlx"
if [ -d "$FUNASR_DIR" ]; then
    cd "$FUNASR_DIR"
    PARAFORMER_DIR="$FUNASR_DIR/paraformer"
    if [ ! -d "$PARAFORMER_DIR" ]; then
        PARAFORMER_DIR="$WORKSPACE_ROOT/paraformer"
    fi
    if [ -d "$PARAFORMER_DIR" ]; then
        echo "Model found at: $PARAFORMER_DIR"
        cargo run --release --example benchmark -- "$AUDIO_FILE" "$PARAFORMER_DIR" "$ITERATIONS" 2>&1 || echo "funasr-mlx benchmark failed"
    else
        echo -e "${YELLOW}Skipping: paraformer model not found${NC}"
        echo "Expected at: $PARAFORMER_DIR"
    fi
else
    echo -e "${YELLOW}Skipping: funasr-mlx not found at $FUNASR_DIR${NC}"
fi
echo ""

echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Benchmark 3: funasr-nano-mlx (SenseVoice + Qwen)${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""

FUNASR_NANO_DIR="$WORKSPACE_ROOT/funasr-nano-mlx"
if [ -d "$FUNASR_NANO_DIR" ]; then
    cd "$FUNASR_NANO_DIR"
    NANO_MODEL_DIR="$FUNASR_NANO_DIR/Fun-ASR-Nano-2512"
    if [ -d "$NANO_MODEL_DIR" ]; then
        echo "Model found at: $NANO_MODEL_DIR"
        cargo run --release --example fair_benchmark 2>&1 || \
        cargo run --release --example benchmark -- "$AUDIO_FILE" "$ITERATIONS" 2>&1 || \
        echo "funasr-nano-mlx benchmark failed"
    else
        echo -e "${YELLOW}Skipping: Fun-ASR-Nano-2512 model not found${NC}"
        echo "Expected at: $NANO_MODEL_DIR"
    fi
else
    echo -e "${YELLOW}Skipping: funasr-nano-mlx not found at $FUNASR_NANO_DIR${NC}"
fi
echo ""

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                        BENCHMARK COMPLETE                            ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Note: Compare the Mean RTF values across implementations."
echo "Lower RTF = faster (e.g., 0.1x RTF = 10x real-time speed)"
