# ominix-api

Unified OpenAI-compatible HTTP API server for OminiX-MLX models on Apple Silicon. Exposes ASR, TTS, LLM, and OCR through a single REST endpoint with drop-in compatibility for OpenAI client libraries.

**Version**: 1.0.0

## Features

| Feature | Default | Model Crate | Endpoint |
|---------|---------|-------------|----------|
| `asr` | Yes | `qwen3-asr-mlx` | `POST /v1/audio/transcriptions` |
| `tts` | Yes | `qwen3-tts-mlx` | `POST /v1/audio/speech`, `POST /v1/audio/speech/clone` |
| `llm` | No | `qwen3.5-35B-mlx` | `POST /v1/chat/completions` |
| `ocr` | No | `deepseek-ocr2-mlx` | `POST /v1/ocr` |

## Quick Start

```bash
# Install from source
cargo install --git https://github.com/OminiX-ai/OminiX-MLX ominix-api --features tts

# Or build locally
cargo build --release -p ominix-api --features tts

# Start with TTS
ominix-api --tts-model ~/.OminiX/models/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit --port 8080

# Start with ASR + TTS
ominix-api \
    --asr-model ~/.OminiX/models/qwen3-asr-1.7b \
    --tts-model ~/.OminiX/models/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit \
    --port 8080

# Multi-port deployment (recommended for production)
ominix-api --asr-model ~/.OminiX/models/qwen3-asr-1.7b --port 8081 &
ominix-api --tts-model ~/.OminiX/models/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit --port 8082 &
ominix-api --tts-model ~/.OminiX/models/Qwen3-TTS-12Hz-1.7B-Base --port 8083 &
```

## CLI Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--port` | 8080 | Port to listen on |
| `--asr-model <PATH>` | None | ASR model directory |
| `--tts-model <PATH>` | None | TTS model directory |
| `--llm-model <PATH>` | None | LLM model directory |
| `--ocr-model <PATH>` | None | OCR model directory |
| `--language` | Chinese | Default ASR language |
| `--tts-speaker` | vivian | Default TTS speaker |
| `--tts-language` | english | Default TTS language |
| `--models-dir` | ~/.ominix/models | Model management directory |

## Endpoints

### Health Check

```
GET /health
```

```json
{"status": "ok", "version": "1.0.0"}
```

### Speech-to-Text (ASR)

```
POST /v1/audio/transcriptions
```

OpenAI Whisper-compatible. Supports multipart/form-data and JSON.

```bash
# Multipart (OpenAI-compatible)
curl http://localhost:8080/v1/audio/transcriptions \
    -F file=@audio.wav -F language=Chinese

# JSON with file path
curl http://localhost:8080/v1/audio/transcriptions \
    -H "Content-Type: application/json" \
    -d '{"file_path": "/path/to/audio.wav", "language": "English"}'

# JSON with base64
curl http://localhost:8080/v1/audio/transcriptions \
    -H "Content-Type: application/json" \
    -d '{"file": "<base64-encoded-audio>", "language": "Chinese"}'
```

**Request fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | string | * | Base64-encoded audio |
| `file_path` | string | * | Local audio file path |
| `language` | string | No | Language code (Chinese, English, etc.) |
| `response_format` | string | No | `json` (default) or `verbose_json` |

*One of `file` or `file_path` required.

### Text-to-Speech (TTS)

```
POST /v1/audio/speech
```

Returns WAV audio (16-bit PCM, mono, 24kHz). Headers include `X-Audio-Duration` and `X-Processing-Time`.

```bash
# Basic preset speaker
curl http://localhost:8080/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"input": "Hello, world!", "voice": "vivian", "language": "english"}' \
    --output output.wav

# With emotion/style prompt
curl http://localhost:8080/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
      "input": "我太开心了！终于完成了！",
      "voice": "serena",
      "language": "chinese",
      "prompt": "用兴奋激动的语气说话，充满热情和活力"
    }' --output excited.wav

# Speed control
curl http://localhost:8080/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"input": "This is slower speech.", "voice": "ryan", "speed": 0.7}' \
    --output slow.wav

# Voice design (text-described voice)
curl http://localhost:8080/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
      "input": "Hello there!",
      "prompt": "A deep male voice with a calm, authoritative tone",
      "language": "english"
    }' --output designed.wav
```

**Request fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input` | string | (required) | Text to synthesize |
| `voice` | string | default speaker | Speaker name: vivian, serena, ryan, aiden, eric, dylan, uncle_fu, ono_anna, sohee |
| `language` | string | server default | chinese, english, japanese, korean, french, german, spanish, russian, italian, portuguese |
| `prompt` | string | None | Style/emotion instruction (alias: `instruct`) |
| `speed` | float | 1.0 | Speed factor: 0.5-2.0 (>1.0 = faster, <1.0 = slower) |
| `temperature` | float | 0.9 | Sampling temperature |
| `top_k` | int | 50 | Top-k sampling |
| `top_p` | float | 1.0 | Nucleus sampling threshold |
| `seed` | int | None | Random seed for deterministic output |
| `repetition_penalty` | float | 1.05 | Penalizes repeated codec tokens |

**Dispatch logic:**

| Condition | Mode |
|-----------|------|
| `reference_audio` + `prompt` | Voice cloning + emotion/style (experimental) |
| `reference_audio` only | Voice cloning (x-vector) |
| `prompt` + `voice` | Combined speaker + style instruction |
| `prompt` only | Voice design (text-described voice) |
| Default | Preset speaker |

#### Verified Emotion Prompts

These Chinese prompts are tested and produce accurate style control:

| Style | Prompt |
|-------|--------|
| Excited | `用兴奋激动的语气说话，充满热情和活力` |
| Sad | `用悲伤失望的语气说话，声音低沉，语速缓慢` |
| Cheerful | `用开朗愉快的语气说话，声音明亮上扬，节奏轻快` |
| Shout | `用大声喊叫的方式说话，声音高亢有力，语速快` |
| Sarcastic | `用讽刺嘲讽的语气说话，语调阴阳怪气，拖长尾音` |
| Soft | `用温柔轻柔的语气说话` |
| Panic | `用惊慌恐惧的语气说话，声音颤抖，语速急促` (use with `repetition_penalty: 1.3`) |

Custom free-form prompts are also supported. Include emotion + timbre + pace for strongest control.

### Voice Cloning

```
POST /v1/audio/speech/clone
```

Clone a voice from a reference audio clip (3-10 seconds recommended).

```bash
# x-vector mode (recommended)
curl http://localhost:8083/v1/audio/speech/clone \
    -H "Content-Type: application/json" \
    -d '{
      "input": "你好，很高兴认识你。",
      "reference_audio": "<base64-wav>",
      "language": "chinese"
    }' --output cloned.wav

# Voice cloning + emotion/style control (experimental)
curl http://localhost:8083/v1/audio/speech/clone \
    -H "Content-Type: application/json" \
    -d '{
      "input": "你好，很高兴认识你。",
      "reference_audio": "<base64-wav>",
      "prompt": "用悲伤的语气说",
      "language": "chinese"
    }' --output cloned_sad.wav
```

**Additional fields:**

| Field | Type | Description |
|-------|------|-------------|
| `reference_audio` | string | Base64-encoded reference audio (WAV, 3-10s) |
| `prompt` / `instruct` | string | Emotion/style instruction for cloned voice (experimental) |

> **Note**: Voice cloning + emotion is experimental. The Base model was not trained for this
> combination — emotion effects are weaker than with preset speakers. Some emotions (sad, angry,
> soft) work well; others (fearful, surprised) may sound flat. Use short prompts like `"用...语气说"`.
> The upcoming Qwen3-TTS-25Hz-VoiceEditing model is expected to provide native support.

### Chat Completions (LLM)

```
POST /v1/chat/completions
```

OpenAI-compatible chat completion. Supports streaming SSE.

```bash
# Non-streaming
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "messages": [{"role": "user", "content": "What is Rust?"}],
      "temperature": 0.7,
      "max_tokens": 256
    }'

# Streaming
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "messages": [{"role": "user", "content": "Tell me a story."}],
      "stream": true
    }'
```

**Request fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `messages` | array | (required) | Array of `{role, content}` messages |
| `temperature` | float | 0.7 | Sampling temperature |
| `max_tokens` | int | 2048 | Maximum output tokens |
| `stream` | bool | false | Enable SSE streaming |

### OCR / Document Understanding

```
POST /v1/ocr
```

Uses DeepSeek-OCR-2 vision-language model. Supports multipart and JSON.

```bash
# Multipart
curl http://localhost:8080/v1/ocr \
    -F file=@document.png \
    -F prompt="Convert this document to markdown."

# JSON with base64
curl http://localhost:8080/v1/ocr \
    -H "Content-Type: application/json" \
    -d '{
      "image": "<base64-png>",
      "prompt": "Extract all text from this image.",
      "max_tokens": 4096
    }'
```

**Request fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` / `image` | bytes/string | (required) | Image file or base64 string |
| `prompt` | string | Markdown conversion | Custom OCR prompt |
| `temperature` | float | 0.0 | Sampling temperature |
| `max_tokens` | int | 8192 | Maximum output tokens |
| `stream` | bool | false | Enable SSE streaming |

### Model Management

```bash
# List available models
curl http://localhost:8080/v1/models

# Download a model
curl -X POST http://localhost:8080/v1/models/download \
    -H "Content-Type: application/json" \
    -d '{"repo_id": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"}'

# Delete a model
curl -X DELETE http://localhost:8080/v1/models/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit
```

Model type is auto-detected: `talker_config` in config.json = TTS, `audio_config` = ASR, otherwise LLM.

## Architecture

```
Client (curl / OpenAI SDK / mofa-fm)
    │
    ▼
ominix-api (hyper 1.0 + tokio)
    │
    ├── /v1/audio/transcriptions ──→ qwen3-asr-mlx (async worker)
    ├── /v1/audio/speech ──────────→ qwen3-tts-mlx (async worker, panic recovery)
    ├── /v1/chat/completions ──────→ qwen3.5-35B-mlx (async worker, SSE streaming)
    ├── /v1/ocr ───────────────────→ deepseek-ocr2-mlx (async worker)
    └── /v1/models ────────────────→ config.json + filesystem
```

Each model type runs in a dedicated blocking thread with mpsc channels for request queuing. MLX models are not thread-safe, so inference is single-threaded per model. For production, run separate instances per model type on different ports.

## Error Responses

```json
{
  "error": {
    "message": "Error description",
    "type": "invalid_request_error"
  }
}
```

| Status | Type | Description |
|--------|------|-------------|
| 400 | `invalid_request_error` | Bad JSON, missing fields |
| 404 | `not_found` | Endpoint not found |
| 501 | `not_implemented` | Feature not enabled in build |
| 503 | `service_unavailable` | Model not loaded |
| 500 | `server_error` | Inference failure |

## Configuration

**Config file**: `~/.ominix/config.json` (auto-created)

**Environment variables:**

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace token for model downloads |

## Building

```bash
# ASR + TTS (default features)
cargo build --release -p ominix-api

# All features
cargo build --release -p ominix-api --features "asr,tts,llm,ocr"

# TTS only
cargo build --release -p ominix-api --no-default-features --features tts
```

**Requirements:**
- macOS with Apple Silicon (M1/M2/M3/M4)
- Rust 1.82.0+
- Xcode Command Line Tools

## License

MIT OR Apache-2.0
