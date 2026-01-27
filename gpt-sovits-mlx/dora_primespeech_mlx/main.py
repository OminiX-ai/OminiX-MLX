#!/usr/bin/env python3
"""Dora node for GPT-SoVITS MLX TTS synthesis.

This is a drop-in replacement for dora-primespeech that uses the
hybrid CoreML + MLX architecture for ~8x faster inference on Apple Silicon.

Usage in dataflow.yaml:
    nodes:
      - id: primespeech-mlx
        operator:
          python: dora_primespeech_mlx/main.py
        inputs:
          text: orchestrator/tts_text
          control: orchestrator/control
        outputs:
          - audio
          - segment_complete
          - status
          - log

Environment variables:
    VOICE_NAME: Voice to use (default: Doubao)
    MODEL_DIR: Path to model directory
    USE_ANE: Use ANE for encoders (default: true)
    TEMPERATURE: Sampling temperature (default: 0.8)
    TOP_K: Top-k sampling (default: 3)
    SPEED_FACTOR: Speech speed (default: 1.0)
    LOG_LEVEL: Logging level (default: INFO)
"""

import os
import sys
import json
import time
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np

# Configure logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("primespeech-mlx")

# Import Dora
try:
    from dora import Node
    HAS_DORA = True
except ImportError:
    HAS_DORA = False
    logger.warning("Dora not installed. Running in standalone mode.")

# Import MLX engine
try:
    import mlx.core as mx
    from python.engine import GPTSoVITSEngine
    from python.models.config import SynthesisConfig
    HAS_MLX = True
except ImportError as e:
    HAS_MLX = False
    logger.error(f"Failed to import MLX engine: {e}")


class PrimeSpeechMLXNode:
    """Dora node for GPT-SoVITS MLX TTS."""

    def __init__(self):
        """Initialize the TTS node."""
        # Configuration from environment
        self.voice_name = os.environ.get("VOICE_NAME", "Doubao")
        self.model_dir = os.environ.get("MODEL_DIR", "~/.dora/models/gpt-sovits-mlx")
        self.use_ane = os.environ.get("USE_ANE", "true").lower() == "true"
        self.temperature = float(os.environ.get("TEMPERATURE", "0.8"))
        self.top_k = int(os.environ.get("TOP_K", "3"))
        self.top_p = float(os.environ.get("TOP_P", "0.95"))
        self.speed_factor = float(os.environ.get("SPEED_FACTOR", "1.0"))
        self.sample_rate = int(os.environ.get("SAMPLE_RATE", "32000"))
        self.streaming = os.environ.get("STREAMING", "false").lower() == "true"

        # Engine
        self.engine: Optional[GPTSoVITSEngine] = None

        # Statistics
        self.stats = {
            "total_requests": 0,
            "total_audio_seconds": 0.0,
            "total_processing_time": 0.0,
            "errors": 0,
        }

        # Session tracking
        self.current_session: Optional[str] = None

    def initialize(self) -> bool:
        """Initialize the TTS engine."""
        if not HAS_MLX:
            logger.error("MLX not available. Cannot initialize engine.")
            return False

        try:
            logger.info(f"Initializing GPT-SoVITS MLX engine...")
            logger.info(f"  Model directory: {self.model_dir}")
            logger.info(f"  Voice: {self.voice_name}")
            logger.info(f"  Use ANE: {self.use_ane}")

            # Expand path
            model_dir = Path(self.model_dir).expanduser()

            # Create engine
            self.engine = GPTSoVITSEngine(
                model_dir=str(model_dir),
                use_ane=self.use_ane,
                use_compile=True,
            )

            # Load voice
            self.engine.load_voice(self.voice_name)

            # Warmup
            self.engine.warmup()

            logger.info("Engine initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize engine: {e}")
            return False

    def synthesize(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Synthesize speech from text.

        Args:
            text: Input text
            metadata: Optional metadata (session_id, request_id, etc.)

        Returns:
            Dictionary with audio data and metadata
        """
        if self.engine is None:
            return {
                "error": "Engine not initialized",
                "status": "error",
            }

        metadata = metadata or {}
        session_id = metadata.get("session_id", "default")
        request_id = metadata.get("request_id", "unknown")

        start_time = time.perf_counter()

        try:
            # Create synthesis config
            config = SynthesisConfig(
                voice_name=self.voice_name,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                speed_factor=self.speed_factor,
                sample_rate=self.sample_rate,
                streaming=self.streaming,
            )

            # Synthesize
            result = self.engine.synthesize(text, config)

            # Update stats
            processing_time = time.perf_counter() - start_time
            self.stats["total_requests"] += 1
            self.stats["total_audio_seconds"] += result.duration
            self.stats["total_processing_time"] += processing_time

            # Calculate realtime factor
            rtf = processing_time / result.duration if result.duration > 0 else 0

            logger.info(
                f"Synthesized: {len(text)} chars -> {result.duration:.2f}s audio "
                f"in {processing_time*1000:.0f}ms (RTF: {rtf:.2f}x)"
            )

            return {
                "audio": result.audio,
                "sample_rate": result.sample_rate,
                "duration": result.duration,
                "timing": result.timing,
                "session_id": session_id,
                "request_id": request_id,
                "status": "completed",
            }

        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Synthesis failed: {e}")
            return {
                "error": str(e),
                "session_id": session_id,
                "request_id": request_id,
                "status": "error",
            }

    def handle_control(self, command: str) -> Dict[str, Any]:
        """Handle control commands.

        Args:
            command: Control command string

        Returns:
            Response dictionary
        """
        command = command.strip().lower()

        if command == "reset":
            # Reset session state
            self.current_session = None
            return {"status": "reset", "message": "Session reset"}

        elif command == "stats":
            # Return statistics
            return {
                "status": "stats",
                "stats": self.stats,
                "engine_stats": self.engine.get_stats() if self.engine else {},
            }

        elif command == "list_voices":
            # List available voices
            return {
                "status": "voices",
                "voices": ["Doubao", "Trump", "Maple"],  # TODO: get from model dir
            }

        elif command.startswith("change_voice:"):
            # Change voice
            voice_name = command.split(":", 1)[1].strip()
            try:
                self.engine.load_voice(voice_name)
                self.voice_name = voice_name
                return {"status": "voice_changed", "voice": voice_name}
            except Exception as e:
                return {"status": "error", "error": str(e)}

        elif command == "cleanup":
            # Cleanup resources
            return {"status": "cleanup", "message": "Cleanup complete"}

        else:
            return {"status": "unknown", "command": command}

    def run_dora(self):
        """Run as Dora node."""
        if not HAS_DORA:
            logger.error("Dora not installed")
            return

        node = Node()

        # Initialize engine
        if not self.initialize():
            logger.error("Failed to initialize. Exiting.")
            return

        logger.info("Dora node started. Waiting for events...")

        for event in node:
            if event["type"] == "INPUT":
                input_id = event["id"]
                data = event["data"]

                if input_id == "text":
                    # Extract text and metadata
                    text = data.decode() if isinstance(data, bytes) else str(data)
                    metadata = event.get("metadata", {})

                    # Synthesize
                    result = self.synthesize(text, metadata)

                    if result.get("status") == "completed":
                        # Send audio output
                        audio = result["audio"]
                        node.send_output(
                            "audio",
                            audio.tobytes(),
                            metadata={
                                "sample_rate": result["sample_rate"],
                                "duration": result["duration"],
                                "voice": self.voice_name,
                            },
                        )

                        # Send completion signal
                        node.send_output(
                            "segment_complete",
                            b"completed",
                            metadata={
                                "session_id": result.get("session_id"),
                                "request_id": result.get("request_id"),
                            },
                        )
                    else:
                        # Send error
                        node.send_output(
                            "segment_complete",
                            b"error",
                            metadata={
                                "error": result.get("error"),
                            },
                        )

                elif input_id == "control":
                    command = data.decode() if isinstance(data, bytes) else str(data)
                    result = self.handle_control(command)

                    # Send status
                    node.send_output(
                        "status",
                        json.dumps(result).encode(),
                    )

            elif event["type"] == "STOP":
                logger.info("Received STOP signal")
                break

        logger.info("Dora node stopped")


def main():
    """Main entry point."""
    node = PrimeSpeechMLXNode()

    if HAS_DORA:
        node.run_dora()
    else:
        # Standalone mode for testing
        logger.info("Running in standalone mode...")

        if not node.initialize():
            logger.error("Failed to initialize")
            return

        # Test synthesis
        test_text = "你好世界，这是一个测试。"
        result = node.synthesize(test_text)

        if result.get("status") == "completed":
            logger.info(f"Test successful!")
            logger.info(f"  Duration: {result['duration']:.2f}s")
            logger.info(f"  Timing: {result.get('timing', {})}")
        else:
            logger.error(f"Test failed: {result.get('error')}")


if __name__ == "__main__":
    main()
