# ominix-api

> **Work in progress.** This crate is a stub for the unified OpenAI-compatible HTTP API server for OminiX-MLX models.

## Intent

`ominix-api` will expose OminiX-MLX model crates (ASR, TTS, VLM, LLM) through a single HTTP server that implements the [OpenAI API](https://platform.openai.com/docs/api-reference) surface, making the models drop-in compatible with OpenAI client libraries.

## Planned Endpoints

| Endpoint | Model | Status |
|----------|-------|--------|
| `POST /v1/audio/transcriptions` | `qwen3-asr-mlx` | Implemented |
| `POST /v1/chat/completions` | LLM crates | Planned |
| `POST /v1/audio/speech` | `qwen3-tts-mlx` | Planned |

## Usage

```bash
# Build and start the server (ASR feature enabled by default)
cargo run --release -p ominix-api -- \
    --asr-model ~/.OminiX/models/qwen3-asr-1.7b \
    --port 8080

# Transcribe (OpenAI Whisper-compatible multipart)
curl http://localhost:8080/v1/audio/transcriptions \
    -F file=@audio.wav -F language=Chinese

# Transcribe (JSON body)
curl http://localhost:8080/v1/audio/transcriptions \
    -H "Content-Type: application/json" \
    -d '{"file_path": "audio.wav", "language": "English"}'
```

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `asr` | yes | Enable ASR endpoint via `qwen3-asr-mlx` |

## License

MIT OR Apache-2.0
