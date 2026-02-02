# GPT-SoVITS MLX Migration: Development Plan

**Version:** 1.0
**Date:** 2026-01-20
**Status:** Draft

---

## Overview

This document outlines the development plan for migrating dora-primespeech (GPT-SoVITS) to a hybrid CoreML + MLX architecture.

### Project Goals

| Goal | Metric | Target |
|------|--------|--------|
| Performance | End-to-end latency | < 150ms (8x improvement) |
| Quality | Audio quality (MOS) | No regression from baseline |
| Compatibility | Voice support | All 14 existing voices |
| Integration | Dora compatibility | Drop-in replacement |

### Timeline Summary

```
Phase 0: Setup & Validation     ████░░░░░░░░░░░░░░░░  Week 1
Phase 1: GPT Stage (MLX)        ░░░░████████░░░░░░░░  Weeks 2-4
Phase 2: CoreML Encoders        ░░░░░░░░░░░░████░░░░  Weeks 5-6
Phase 3: Vocoder (MLX)          ░░░░░░░░░░░░░░░░████  Weeks 7-9
Phase 4: Integration            ░░░░░░░░░░░░░░░░░░██  Week 10
Phase 5: Optimization           ░░░░░░░░░░░░░░░░░░░█  Week 11-12
                                ────────────────────
                                1  2  3  4  5  6  7  8  9 10 11 12
```

---

## Phase 0: Setup & Validation (Week 1)

### Objectives
- Set up development environment
- Validate baseline performance
- Create benchmark infrastructure

### Tasks

#### 0.1 Environment Setup
```
□ 0.1.1 Install MLX and mlx-rs dependencies
        Command: pip install mlx mlx-lm coremltools
        Command: cargo add mlx-rs --path ../mlx-rs

□ 0.1.2 Set up dora-primespeech locally
        Command: cd ~/home/mofa-studio/node-hub/dora-primespeech
        Command: pip install -e .

□ 0.1.3 Download test voice models (Doubao, Trump)
        Verify: Models load successfully

□ 0.1.4 Create project structure
        mkdir -p gpt-sovits-mlx/{python,rust,tests,models,scripts}
```

#### 0.2 Baseline Benchmarking
```
□ 0.2.1 Create benchmark script
        File: scripts/benchmark_baseline.py
        Metrics: latency per component, memory usage, GPU utilization

□ 0.2.2 Run baseline benchmarks (CPU)
        Record: CNHubert, RoBERTa, GPT, Vocoder timings

□ 0.2.3 Run baseline benchmarks (MPS if available)
        Compare: CPU vs MPS performance

□ 0.2.4 Document baseline results
        File: docs/baseline_benchmarks.md
```

#### 0.3 Test Infrastructure
```
□ 0.3.1 Create test audio samples
        Files: tests/fixtures/reference_*.wav (3 voices)

□ 0.3.2 Create test text corpus
        Files: tests/fixtures/test_texts.json (short/medium/long)

□ 0.3.3 Set up pytest infrastructure
        File: tests/conftest.py
        Dependencies: pytest, pytest-benchmark

□ 0.3.4 Create golden output files for regression testing
        Generate: Reference outputs from PyTorch implementation
```

### Deliverables
- [x] Working development environment
- [x] Baseline benchmark report
- [x] Test infrastructure ready

### Exit Criteria
- Can run dora-primespeech locally
- Have baseline numbers for all components
- Test fixtures created

---

## Phase 1: GPT Stage Migration (Weeks 2-4)

### Objectives
- Port GPT model to MLX (Python first)
- Implement optimized KV cache
- Achieve 5x speedup on GPT generation

### Week 2: Model Definition

#### 1.1 MLX Model Implementation
```
□ 1.1.1 Define GPTSoVITSConfig dataclass
        File: gpt-sovits-mlx/python/models/config.py
        Fields: hidden_size, num_layers, num_heads, vocab sizes

□ 1.1.2 Implement GPTSoVITSAttention
        File: gpt-sovits-mlx/python/models/attention.py
        Reference: mlx-examples/musicgen/transformer.py
        Include: RoPE, KV projection, scaled dot product

□ 1.1.3 Implement GPTSoVITSMLP
        File: gpt-sovits-mlx/python/models/mlp.py
        Include: SwiGLU activation (gate_proj, up_proj, down_proj)

□ 1.1.4 Implement GPTSoVITSBlock
        File: gpt-sovits-mlx/python/models/block.py
        Structure: RMSNorm → Attention → RMSNorm → MLP

□ 1.1.5 Implement GPTSoVITS full model
        File: gpt-sovits-mlx/python/models/gpt.py
        Include: Embeddings, N blocks, output head
```

#### 1.2 Weight Conversion
```
□ 1.2.1 Analyze PyTorch checkpoint structure
        Script: scripts/analyze_checkpoint.py
        Output: Layer names, shapes, dtypes

□ 1.2.2 Create weight mapping dictionary
        Map: PyTorch layer names → MLX layer names

□ 1.2.3 Implement conversion script
        File: scripts/convert_gpt_weights.py
        Output: model.safetensors

□ 1.2.4 Validate converted weights
        Test: Forward pass produces same output (within tolerance)
```

### Week 3: KV Cache & Generation

#### 1.3 KV Cache Implementation
```
□ 1.3.1 Implement basic KVCache class
        File: gpt-sovits-mlx/python/cache.py
        Methods: update(), reset(), get_seq_len()

□ 1.3.2 Implement step-allocated KVCache
        Optimization: Pre-allocate in 256-token steps
        Method: In-place slice updates

□ 1.3.3 Benchmark KV cache performance
        Compare: Concat vs step-allocated
        Target: < 0.1ms per token update

□ 1.3.4 Add cache management to model
        Modify: GPTSoVITS.forward() to accept/return cache
```

#### 1.4 Generation Loop
```
□ 1.4.1 Implement basic generation loop
        File: gpt-sovits-mlx/python/generate.py
        Function: generate_semantic_tokens()

□ 1.4.2 Add temperature scaling
        Parameter: temperature (default 0.8)

□ 1.4.3 Add top-k sampling
        Parameter: top_k (default 3)
        Implementation: argpartition-based filtering

□ 1.4.4 Add top-p (nucleus) sampling
        Parameter: top_p (default 0.95)

□ 1.4.5 Add EOS token handling
        Stop: When token == 1024

□ 1.4.6 Add async evaluation
        Use: mx.async_eval() for pipelining
```

### Week 4: Testing & Optimization

#### 1.5 Validation
```
□ 1.5.1 Create unit tests for model components
        File: tests/test_gpt_model.py
        Coverage: Attention, MLP, Block, Full model

□ 1.5.2 Create integration test
        Test: Full generation produces valid token sequence

□ 1.5.3 Create numerical comparison test
        Compare: MLX output vs PyTorch output
        Tolerance: < 1e-4 absolute difference

□ 1.5.4 Create quality test
        Test: Generated tokens produce intelligible audio
```

#### 1.6 Optimization
```
□ 1.6.1 Profile generation loop
        Tool: mx.metal.start_capture() / py-spy

□ 1.6.2 Identify bottlenecks
        Analyze: Per-operation timings

□ 1.6.3 Apply mx.compile to forward pass
        Target: Fuse operations, reduce kernel launches

□ 1.6.4 Benchmark optimized version
        Target: 60-80ms for 100 tokens (5x speedup)
```

### Deliverables
- [ ] GPT model in MLX (Python)
- [ ] Weight conversion script
- [ ] Optimized KV cache
- [ ] Generation loop with sampling
- [ ] Benchmark showing 5x+ speedup

### Exit Criteria
- GPT stage runs in < 80ms for 100 tokens
- Output quality matches PyTorch version
- All tests passing

---

## Phase 2: CoreML Encoders (Weeks 5-6)

### Objectives
- Convert CNHubert to CoreML with ANE optimization
- Convert RoBERTa to CoreML with ANE optimization
- Achieve 10x+ speedup on encoders

### Week 5: CNHubert

#### 2.1 Model Analysis
```
□ 2.1.1 Document CNHubert architecture
        Layers: Conv stack + Transformer encoder

□ 2.1.2 Identify ANE-compatible operations
        Check: All layers supported in CoreML

□ 2.1.3 Identify required modifications
        Note: Conv1d → Conv2d, attention chunking
```

#### 2.2 CoreML Conversion
```
□ 2.2.1 Create traced PyTorch model
        Script: scripts/trace_cnhubert.py
        Output: cnhubert_traced.pt

□ 2.2.2 Apply ANE optimizations
        Use: ane_transformers patterns
        - Convert Linear → Conv2d
        - Chunk attention heads
        - Use FP16 precision

□ 2.2.3 Convert to CoreML
        Script: scripts/convert_cnhubert_coreml.py
        Output: cnhubert_ane.mlpackage

□ 2.2.4 Validate CoreML model
        Compare: Output vs PyTorch (tolerance < 1e-3)

□ 2.2.5 Benchmark CoreML model
        Target: < 10ms (15x speedup vs CPU)
```

### Week 6: RoBERTa & Integration

#### 2.3 RoBERTa Conversion
```
□ 2.3.1 Trace RoBERTa model
        Script: scripts/trace_roberta.py

□ 2.3.2 Apply ANE optimizations
        Same patterns as CNHubert

□ 2.3.3 Convert to CoreML
        Output: roberta_ane.mlpackage

□ 2.3.4 Validate and benchmark
        Target: < 10ms
```

#### 2.4 CoreML-MLX Integration
```
□ 2.4.1 Create encoder wrapper
        File: gpt-sovits-mlx/python/encoders.py
        Class: CoreMLEncoderWrapper

□ 2.4.2 Implement zero-copy transfer
        Method: MLMultiArray → mx.array without copy

□ 2.4.3 Create unified API
        Function: encode_audio(audio) → features
        Function: encode_text(tokens) → features

□ 2.4.4 Benchmark full encoder pipeline
        Target: < 15ms total (CNHubert + RoBERTa)
```

### Deliverables
- [ ] CNHubert CoreML model (ANE optimized)
- [ ] RoBERTa CoreML model (ANE optimized)
- [ ] Zero-copy CoreML → MLX transfer
- [ ] Encoder wrapper with unified API

### Exit Criteria
- Encoders run in < 15ms total
- No quality regression
- Zero-copy data transfer working

---

## Phase 3: Vocoder Migration (Weeks 7-9)

### Objectives
- Port SoVITS vocoder to MLX
- Implement RVQ decoder
- Achieve 4x speedup on vocoder

### Week 7: RVQ & Duration Predictor

#### 3.1 RVQ Decoder
```
□ 3.1.1 Implement VectorQuantizer
        File: gpt-sovits-mlx/python/models/vq.py
        Reference: mlx-examples/encodec/model.py

□ 3.1.2 Implement ResidualVectorQuantizer
        Structure: 8 codebooks, sum outputs

□ 3.1.3 Convert RVQ codebooks from PyTorch
        Script: scripts/convert_rvq_weights.py

□ 3.1.4 Validate RVQ decoder
        Test: Codebook lookup matches PyTorch
```

#### 3.2 Duration Predictor
```
□ 3.2.1 Implement DurationPredictor
        File: gpt-sovits-mlx/python/models/duration.py
        Structure: Conv → Flow layers

□ 3.2.2 Implement flow layers
        Include: AffineCouplingLayer

□ 3.2.3 Implement length regulator
        Function: Expand tokens by durations

□ 3.2.4 Convert duration predictor weights
        Validate: Duration predictions match
```

### Week 8: Upsampler

#### 3.3 Upsampler Network
```
□ 3.3.1 Implement ResBlock
        File: gpt-sovits-mlx/python/models/resblock.py
        Include: Dilated convolutions

□ 3.3.2 Implement upsampler
        File: gpt-sovits-mlx/python/models/upsampler.py
        Structure: ConvTranspose1d stack

□ 3.3.3 Implement full vocoder
        File: gpt-sovits-mlx/python/models/vocoder.py
        Combine: Duration → RVQ → MRTE → Upsampler

□ 3.3.4 Convert all vocoder weights
        Script: scripts/convert_vocoder_weights.py
```

### Week 9: Optimization

#### 3.4 Vocoder Optimization
```
□ 3.4.1 Profile vocoder
        Identify: Slow operations

□ 3.4.2 Fuse convolution operations
        Use: mx.compile

□ 3.4.3 Optimize memory access
        Ensure: Contiguous tensors

□ 3.4.4 Benchmark optimized vocoder
        Target: < 60ms for 1s audio
```

#### 3.5 End-to-End Validation
```
□ 3.5.1 Create E2E test
        Input: Text → Output: Audio

□ 3.5.2 Audio quality validation
        Method: MOS comparison
        Target: No regression

□ 3.5.3 Benchmark E2E pipeline
        Target: < 150ms total
```

### Deliverables
- [ ] RVQ decoder in MLX
- [ ] Duration predictor in MLX
- [ ] Full vocoder in MLX
- [ ] E2E synthesis working

### Exit Criteria
- Vocoder runs in < 60ms
- E2E latency < 150ms
- Audio quality maintained

---

## Phase 4: Integration (Week 10)

### Objectives
- Create production-ready package
- Integrate with Dora framework
- Add streaming support

### Tasks

#### 4.1 Package Structure
```
□ 4.1.1 Create Python package
        Package: gpt-sovits-mlx
        Structure: pip-installable

□ 4.1.2 Create CLI interface
        Command: gpt-sovits-mlx synthesize "text"

□ 4.1.3 Write API documentation
        File: docs/api.md

□ 4.1.4 Create example scripts
        Files: examples/*.py
```

#### 4.2 Dora Integration
```
□ 4.2.1 Create Dora node
        File: dora_primespeech_mlx/main.py
        Interface: Match existing dora-primespeech

□ 4.2.2 Implement input handlers
        Inputs: text, control

□ 4.2.3 Implement output handlers
        Outputs: audio, segment_complete, status

□ 4.2.4 Add configuration via environment
        Variables: VOICE_NAME, USE_ANE, etc.

□ 4.2.5 Test with Dora dataflow
        Verify: Drop-in replacement works
```

#### 4.3 Streaming Support
```
□ 4.3.1 Implement chunked synthesis
        Method: Process text in segments

□ 4.3.2 Implement audio chunk yielding
        Generator: Yield chunks as produced

□ 4.3.3 Add flow control
        Handle: Backpressure from downstream

□ 4.3.4 Test streaming latency
        Target: First audio chunk < 100ms
```

### Deliverables
- [ ] Installable Python package
- [ ] Dora node (drop-in replacement)
- [ ] Streaming synthesis
- [ ] Documentation

### Exit Criteria
- Package installs cleanly
- Dora integration passes all tests
- Streaming works with acceptable latency

---

## Phase 5: Optimization & Polish (Weeks 11-12)

### Objectives
- Optimize for production
- Add mlx-rs Rust implementation (optional)
- Complete documentation

### Week 11: Production Optimization

#### 5.1 Performance Tuning
```
□ 5.1.1 Profile full pipeline
        Tool: Instruments / py-spy

□ 5.1.2 Identify remaining bottlenecks
        Focus: Memory allocations, kernel launches

□ 5.1.3 Apply final optimizations
        Techniques: Compile, fusion, async

□ 5.1.4 Validate performance targets
        Target: < 120ms E2E (consistent)
```

#### 5.2 Memory Optimization
```
□ 5.2.1 Implement model sharding
        Support: Run on 8GB devices

□ 5.2.2 Add memory monitoring
        Log: Peak memory usage

□ 5.2.3 Implement cache clearing
        Method: Release memory between requests

□ 5.2.4 Test memory under load
        Scenario: Many sequential requests
```

### Week 12: Documentation & Release

#### 5.3 Documentation
```
□ 5.3.1 Complete API documentation
        Format: Docstrings + Sphinx

□ 5.3.2 Write user guide
        File: docs/user_guide.md

□ 5.3.3 Write deployment guide
        File: docs/deployment.md

□ 5.3.4 Create troubleshooting guide
        File: docs/troubleshooting.md
```

#### 5.4 Release Preparation
```
□ 5.4.1 Create release checklist
        Include: Tests, benchmarks, docs

□ 5.4.2 Version and tag release
        Version: 0.1.0

□ 5.4.3 Create release notes
        Include: Features, performance, breaking changes

□ 5.4.4 Publish package
        Target: Internal package registry
```

#### 5.5 Optional: mlx-rs Port
```
□ 5.5.1 Port GPT model to Rust
        File: mlx-rs-lm/src/models/gpt_sovits.rs

□ 5.5.2 Create PyO3 bindings
        Functions: generate_semantic_tokens, vocode

□ 5.5.3 Benchmark Rust vs Python
        Target: Additional 10-20% speedup

□ 5.5.4 Create hybrid Python/Rust package
        Optional: For maximum performance
```

### Deliverables
- [ ] Production-optimized code
- [ ] Complete documentation
- [ ] Release package
- [ ] (Optional) Rust implementation

### Exit Criteria
- Meets all performance targets
- Documentation complete
- Ready for production deployment

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| ANE not available on target hardware | Medium | High | Implement GPU-only fallback |
| CoreML conversion fails for some ops | Low | High | Use custom ops or simplify model |
| Performance targets not met | Medium | Medium | Prioritize GPT stage (biggest win) |
| Audio quality regression | Low | High | Extensive A/B testing |
| Memory constraints on 8GB devices | Medium | Medium | Implement model sharding |

---

## Resource Requirements

### Development Environment
- Mac with Apple Silicon (M1 Pro or better recommended)
- 16GB+ RAM
- macOS 14.0+ (for latest CoreML)
- Xcode 15+ (for Metal tools)

### Dependencies
```
# Python
mlx >= 0.10.0
mlx-lm >= 0.10.0
coremltools >= 7.0
ane_transformers >= 0.1.0
safetensors >= 0.4.0
tokenizers >= 0.15.0
numpy >= 1.24.0
pytest >= 7.0.0

# Rust (optional)
mlx-rs = "0.25"
pyo3 = "0.20"
numpy = "0.20"
```

### Compute Resources
- Development: Local Mac
- CI/CD: Mac runners (GitHub Actions or self-hosted)
- Benchmarking: Dedicated M2 Pro/Max for consistent results

---

## Milestone Summary

| Milestone | Week | Key Deliverable | Success Metric |
|-----------|------|-----------------|----------------|
| M0: Setup | 1 | Baseline benchmarks | Have reference numbers |
| M1: GPT | 4 | GPT in MLX | < 80ms for 100 tokens |
| M2: Encoders | 6 | CoreML encoders | < 15ms total |
| M3: Vocoder | 9 | Full vocoder | < 60ms |
| M4: Integration | 10 | Dora node | Drop-in works |
| M5: Release | 12 | v0.1.0 | All targets met |

---

## Appendix: Quick Reference

### Key Files to Create

```
gpt-sovits-mlx/
├── python/
│   ├── __init__.py
│   ├── engine.py              # High-level API
│   ├── models/
│   │   ├── __init__.py
│   │   ├── config.py          # Model configurations
│   │   ├── gpt.py             # GPT model
│   │   ├── attention.py       # Attention layers
│   │   ├── mlp.py             # MLP layers
│   │   ├── vocoder.py         # Vocoder model
│   │   └── vq.py              # Vector quantization
│   ├── encoders.py            # CoreML encoder wrappers
│   ├── cache.py               # KV cache
│   └── generate.py            # Generation loop
├── rust/                      # Optional mlx-rs implementation
│   └── src/
│       └── lib.rs
├── scripts/
│   ├── convert_gpt_weights.py
│   ├── convert_vocoder_weights.py
│   ├── convert_cnhubert_coreml.py
│   └── benchmark.py
├── tests/
│   ├── conftest.py
│   ├── test_gpt.py
│   ├── test_vocoder.py
│   └── test_e2e.py
└── dora_primespeech_mlx/      # Dora node
    └── main.py
```

### Key Commands

```bash
# Setup
pip install -e gpt-sovits-mlx

# Convert models
python scripts/convert_gpt_weights.py --input models/gpt.ckpt --output models/gpt.safetensors
python scripts/convert_cnhubert_coreml.py --output models/cnhubert_ane.mlpackage

# Test
pytest tests/ -v

# Benchmark
python scripts/benchmark.py --model models/ --text "你好世界"

# Run Dora node
dora up dataflow.yaml
```
