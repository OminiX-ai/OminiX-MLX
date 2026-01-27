#!/usr/bin/env python3
"""Benchmark script for GPT-SoVITS MLX.

Measures performance of individual components and end-to-end synthesis.

Usage:
    python benchmark.py --model-dir /path/to/models --voice Doubao

    # Component-level benchmarks
    python benchmark.py --component gpt --iterations 100

    # Full pipeline benchmark
    python benchmark.py --full --text "你好世界"
"""

import argparse
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("MLX not available")


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    iterations: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    throughput: Optional[float] = None  # tokens/sec or samples/sec


def benchmark_function(
    fn,
    warmup: int = 3,
    iterations: int = 10,
    name: str = "function",
) -> BenchmarkResult:
    """Benchmark a function.

    Args:
        fn: Function to benchmark (should return something to prevent optimization)
        warmup: Number of warmup iterations
        iterations: Number of timed iterations
        name: Name for the benchmark

    Returns:
        BenchmarkResult with timing statistics
    """
    # Warmup
    for _ in range(warmup):
        result = fn()
        if HAS_MLX:
            mx.eval(result) if hasattr(result, '__mlx_array__') else None

    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = fn()
        if HAS_MLX:
            mx.eval(result) if hasattr(result, '__mlx_array__') else None
        times.append((time.perf_counter() - start) * 1000)  # ms

    return BenchmarkResult(
        name=name,
        iterations=iterations,
        mean_ms=np.mean(times),
        std_ms=np.std(times),
        min_ms=np.min(times),
        max_ms=np.max(times),
    )


def benchmark_kv_cache(iterations: int = 100) -> Dict[str, BenchmarkResult]:
    """Benchmark KV cache implementations."""
    from python.models.cache import KVCache, ConcatKVCache

    results = {}

    # Parameters
    batch_size = 1
    num_heads = 8
    head_dim = 64
    seq_lengths = [32, 64, 128, 256, 512]

    for seq_len in seq_lengths:
        # Step-allocated cache
        def bench_step():
            cache = KVCache(step=256, num_heads=num_heads, head_dim=head_dim)
            for _ in range(seq_len):
                k = mx.random.normal((batch_size, num_heads, 1, head_dim))
                v = mx.random.normal((batch_size, num_heads, 1, head_dim))
                cache.update(k, v)
            return cache._keys

        result = benchmark_function(
            bench_step,
            iterations=iterations,
            name=f"KVCache (step-alloc) seq={seq_len}",
        )
        results[f"step_cache_{seq_len}"] = result

        # Concat cache
        def bench_concat():
            cache = ConcatKVCache()
            for _ in range(seq_len):
                k = mx.random.normal((batch_size, num_heads, 1, head_dim))
                v = mx.random.normal((batch_size, num_heads, 1, head_dim))
                cache.update(k, v)
            return cache._keys

        result = benchmark_function(
            bench_concat,
            iterations=iterations,
            name=f"ConcatKVCache seq={seq_len}",
        )
        results[f"concat_cache_{seq_len}"] = result

    return results


def benchmark_attention(iterations: int = 50) -> Dict[str, BenchmarkResult]:
    """Benchmark attention module."""
    from python.models.attention import Attention
    from python.models.cache import KVCache

    results = {}

    hidden_size = 512
    num_heads = 8
    seq_lengths = [32, 64, 128, 256]

    attn = Attention(
        hidden_size=hidden_size,
        num_heads=num_heads,
    )

    for seq_len in seq_lengths:
        # Without cache (prefill)
        def bench_prefill():
            x = mx.random.normal((1, seq_len, hidden_size))
            output, _ = attn(x)
            return output

        result = benchmark_function(
            bench_prefill,
            iterations=iterations,
            name=f"Attention prefill seq={seq_len}",
        )
        results[f"attn_prefill_{seq_len}"] = result

        # With cache (decode)
        def bench_decode():
            cache = KVCache(step=256, num_heads=num_heads, head_dim=hidden_size // num_heads)
            # Prefill
            x = mx.random.normal((1, seq_len, hidden_size))
            _, cache = attn(x, cache=cache)
            # Decode single token
            x_new = mx.random.normal((1, 1, hidden_size))
            output, _ = attn(x_new, cache=cache)
            return output

        result = benchmark_function(
            bench_decode,
            iterations=iterations,
            name=f"Attention decode seq={seq_len}",
        )
        results[f"attn_decode_{seq_len}"] = result

    return results


def benchmark_gpt(iterations: int = 20) -> Dict[str, BenchmarkResult]:
    """Benchmark GPT model."""
    from python.models.config import GPTConfig
    from python.models.gpt import GPTSoVITS

    results = {}

    config = GPTConfig(
        hidden_size=512,
        num_layers=12,
        num_heads=8,
        intermediate_size=2048,
    )

    model = GPTSoVITS(config)

    phoneme_lens = [10, 20, 50]
    gen_lens = [50, 100, 200]

    for phoneme_len in phoneme_lens:
        for gen_len in gen_lens:
            def bench_generation():
                phoneme_ids = mx.random.randint(0, 512, (1, phoneme_len))
                audio_features = mx.random.normal((1, phoneme_len, 768))

                caches = model.create_caches()
                semantic_ids = mx.array([[0]], dtype=mx.int32)

                # Prefill
                logits, caches = model(phoneme_ids, semantic_ids, audio_features, cache=caches)

                # Generate tokens
                for _ in range(gen_len):
                    next_token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
                    logits, caches = model(phoneme_ids, next_token, audio_features, cache=caches)

                return logits

            result = benchmark_function(
                bench_generation,
                warmup=2,
                iterations=iterations,
                name=f"GPT gen phoneme={phoneme_len} tokens={gen_len}",
            )
            result.throughput = gen_len / (result.mean_ms / 1000)  # tokens/sec
            results[f"gpt_p{phoneme_len}_t{gen_len}"] = result

    return results


def benchmark_vocoder(iterations: int = 20) -> Dict[str, BenchmarkResult]:
    """Benchmark vocoder."""
    from python.models.config import VocoderConfig
    from python.models.vocoder import SoVITSVocoder

    results = {}

    config = VocoderConfig()
    vocoder = SoVITSVocoder(config)

    token_lengths = [50, 100, 200]

    for token_len in token_lengths:
        def bench_vocoder():
            semantic_tokens = mx.random.randint(0, 1024, (1, token_len))
            # MLX uses channels-last format: [batch, time, channels]
            audio_features = mx.random.normal((1, token_len, 768))
            audio = vocoder(semantic_tokens, audio_features)
            return audio

        result = benchmark_function(
            bench_vocoder,
            warmup=2,
            iterations=iterations,
            name=f"Vocoder tokens={token_len}",
        )
        results[f"vocoder_{token_len}"] = result

    return results


def print_results(results: Dict[str, BenchmarkResult]) -> None:
    """Print benchmark results in a table."""
    print("\n" + "=" * 80)
    print(f"{'Benchmark':<45} {'Mean (ms)':<12} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("=" * 80)

    for name, result in results.items():
        throughput_str = f" ({result.throughput:.0f}/s)" if result.throughput else ""
        print(
            f"{result.name:<45} "
            f"{result.mean_ms:>8.2f}ms  "
            f"{result.std_ms:>6.2f}ms  "
            f"{result.min_ms:>6.2f}ms  "
            f"{result.max_ms:>6.2f}ms"
            f"{throughput_str}"
        )

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark GPT-SoVITS MLX")
    parser.add_argument(
        "--component",
        choices=["kv_cache", "attention", "gpt", "vocoder", "all"],
        default="all",
        help="Component to benchmark"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of iterations per benchmark"
    )
    parser.add_argument(
        "--model-dir",
        help="Path to model directory (for full pipeline)"
    )
    parser.add_argument(
        "--voice",
        default="Doubao",
        help="Voice to use for full pipeline"
    )

    args = parser.parse_args()

    if not HAS_MLX:
        print("MLX is required for benchmarking")
        return

    print("GPT-SoVITS MLX Benchmark")
    print(f"MLX version: {mx.__version__ if hasattr(mx, '__version__') else 'unknown'}")
    print(f"Device: {mx.default_device()}")
    print()

    all_results = {}

    if args.component in ["kv_cache", "all"]:
        print("Benchmarking KV Cache...")
        results = benchmark_kv_cache(args.iterations)
        all_results.update(results)
        print_results(results)

    if args.component in ["attention", "all"]:
        print("\nBenchmarking Attention...")
        results = benchmark_attention(args.iterations)
        all_results.update(results)
        print_results(results)

    if args.component in ["gpt", "all"]:
        print("\nBenchmarking GPT Model...")
        results = benchmark_gpt(min(args.iterations, 10))  # GPT is slower
        all_results.update(results)
        print_results(results)

    if args.component in ["vocoder", "all"]:
        print("\nBenchmarking Vocoder...")
        results = benchmark_vocoder(args.iterations)
        all_results.update(results)
        print_results(results)

    # Summary
    if all_results:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        # Find key metrics
        gpt_results = [r for k, r in all_results.items() if k.startswith("gpt_")]
        if gpt_results:
            avg_throughput = np.mean([r.throughput for r in gpt_results if r.throughput])
            print(f"GPT Average Throughput: {avg_throughput:.0f} tokens/sec")

        cache_step = [r for k, r in all_results.items() if "step_cache" in k]
        cache_concat = [r for k, r in all_results.items() if "concat_cache" in k]
        if cache_step and cache_concat:
            step_mean = np.mean([r.mean_ms for r in cache_step])
            concat_mean = np.mean([r.mean_ms for r in cache_concat])
            speedup = concat_mean / step_mean
            print(f"KV Cache Speedup (step vs concat): {speedup:.1f}x")


if __name__ == "__main__":
    main()
