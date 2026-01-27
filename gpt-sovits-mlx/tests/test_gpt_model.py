"""Tests for GPT model components."""

import pytest
import numpy as np

# Skip all tests if MLX is not available
pytest.importorskip("mlx")

import mlx.core as mx
from python.models.config import GPTConfig
from python.models.cache import KVCache, ConcatKVCache, create_kv_caches
from python.models.attention import Attention, RoPE
from python.models.mlp import MLP
from python.models.gpt import GPTSoVITS, RMSNorm, TransformerBlock


class TestKVCache:
    """Tests for KV cache implementations."""

    def test_kv_cache_basic(self):
        """Test basic KV cache operations."""
        cache = KVCache(step=64, num_heads=8, head_dim=64)

        # Initial state
        assert cache.seq_len == 0
        assert cache.max_len == 0

        # First update
        k1 = mx.random.normal((1, 8, 10, 64))
        v1 = mx.random.normal((1, 8, 10, 64))
        keys, values = cache.update(k1, v1)

        assert cache.seq_len == 10
        assert keys.shape == (1, 8, 10, 64)
        assert values.shape == (1, 8, 10, 64)

    def test_kv_cache_multiple_updates(self):
        """Test multiple cache updates."""
        cache = KVCache(step=64, num_heads=8, head_dim=64)

        # Multiple updates
        for i in range(5):
            k = mx.random.normal((1, 8, 1, 64))
            v = mx.random.normal((1, 8, 1, 64))
            keys, values = cache.update(k, v)

        assert cache.seq_len == 5
        assert keys.shape == (1, 8, 5, 64)

    def test_kv_cache_step_allocation(self):
        """Test that cache allocates in steps."""
        cache = KVCache(step=32, num_heads=4, head_dim=32)

        # First update allocates one step
        k = mx.random.normal((1, 4, 10, 32))
        v = mx.random.normal((1, 4, 10, 32))
        cache.update(k, v)

        assert cache.max_len == 32  # Allocated one step

        # Adding more should trigger expansion
        k2 = mx.random.normal((1, 4, 30, 32))
        v2 = mx.random.normal((1, 4, 30, 32))
        cache.update(k2, v2)

        assert cache.max_len == 64  # Allocated two steps
        assert cache.seq_len == 40

    def test_kv_cache_reset(self):
        """Test cache reset."""
        cache = KVCache(step=64, num_heads=8, head_dim=64)

        k = mx.random.normal((1, 8, 20, 64))
        v = mx.random.normal((1, 8, 20, 64))
        cache.update(k, v)

        assert cache.seq_len == 20

        cache.reset()
        assert cache.seq_len == 0
        # Memory should still be allocated
        assert cache.max_len == 64

    def test_concat_cache_equivalence(self):
        """Test that KVCache and ConcatKVCache produce same results."""
        kv_cache = KVCache(step=64, num_heads=4, head_dim=32)
        concat_cache = ConcatKVCache()

        # Same sequence of updates
        for _ in range(10):
            k = mx.random.normal((1, 4, 1, 32))
            v = mx.random.normal((1, 4, 1, 32))

            keys1, values1 = kv_cache.update(k, v)
            keys2, values2 = concat_cache.update(k, v)

        # Results should be close (allowing for float precision)
        assert kv_cache.seq_len == concat_cache.seq_len
        np.testing.assert_allclose(
            np.array(keys1), np.array(keys2), rtol=1e-5
        )


class TestRoPE:
    """Tests for Rotary Position Embeddings."""

    def test_rope_shape(self):
        """Test RoPE output shape."""
        rope = RoPE(dim=64, max_seq_len=512)

        x = mx.random.normal((2, 8, 16, 64))
        y = rope(x, offset=0)

        assert y.shape == x.shape

    def test_rope_with_offset(self):
        """Test RoPE with position offset."""
        rope = RoPE(dim=64, max_seq_len=512)

        x = mx.random.normal((1, 4, 1, 64))

        y1 = rope(x, offset=0)
        y2 = rope(x, offset=10)

        # Results should be different with different offsets
        assert not mx.allclose(y1, y2)


class TestAttention:
    """Tests for attention module."""

    def test_attention_shape(self):
        """Test attention output shape."""
        attn = Attention(
            hidden_size=256,
            num_heads=4,
            max_seq_len=512,
        )

        x = mx.random.normal((2, 16, 256))
        output, _ = attn(x)

        assert output.shape == x.shape

    def test_attention_with_cache(self):
        """Test attention with KV cache."""
        attn = Attention(
            hidden_size=256,
            num_heads=4,
            max_seq_len=512,
        )

        cache = KVCache(step=64, num_heads=4, head_dim=64)

        # First pass with context
        x1 = mx.random.normal((1, 10, 256))
        out1, cache = attn(x1, cache=cache)

        assert out1.shape == (1, 10, 256)
        assert cache.seq_len == 10

        # Second pass with single token
        x2 = mx.random.normal((1, 1, 256))
        out2, cache = attn(x2, cache=cache)

        assert out2.shape == (1, 1, 256)
        assert cache.seq_len == 11


class TestMLP:
    """Tests for MLP module."""

    def test_mlp_shape(self):
        """Test MLP output shape."""
        mlp = MLP(hidden_size=256, intermediate_size=1024)

        x = mx.random.normal((2, 16, 256))
        y = mlp(x)

        assert y.shape == x.shape

    def test_swiglu_activation(self):
        """Test that SwiGLU is applied correctly."""
        mlp = MLP(hidden_size=64, intermediate_size=256)

        x = mx.random.normal((1, 1, 64))
        y = mlp(x)

        # Output should be non-zero and different from input
        assert not mx.allclose(x, y)
        assert mx.any(y != 0)


class TestRMSNorm:
    """Tests for RMS normalization."""

    def test_rmsnorm_shape(self):
        """Test RMSNorm output shape."""
        norm = RMSNorm(hidden_size=256)

        x = mx.random.normal((2, 16, 256))
        y = norm(x)

        assert y.shape == x.shape

    def test_rmsnorm_normalized(self):
        """Test that output has approximately unit RMS."""
        norm = RMSNorm(hidden_size=256)

        x = mx.random.normal((1, 1, 256)) * 10  # Large values
        y = norm(x)

        # RMS should be approximately 1
        rms = mx.sqrt(mx.mean(y * y))
        assert abs(float(rms) - 1.0) < 0.5  # Allow some tolerance


class TestTransformerBlock:
    """Tests for transformer block."""

    def test_block_shape(self):
        """Test transformer block output shape."""
        block = TransformerBlock(
            hidden_size=256,
            num_heads=4,
            intermediate_size=1024,
        )

        x = mx.random.normal((2, 16, 256))
        y, _ = block(x)

        assert y.shape == x.shape

    def test_block_with_cache(self):
        """Test block with KV cache."""
        block = TransformerBlock(
            hidden_size=256,
            num_heads=4,
            intermediate_size=1024,
        )

        cache = KVCache(step=64, num_heads=4, head_dim=64)

        x1 = mx.random.normal((1, 10, 256))
        y1, cache = block(x1, cache=cache)

        x2 = mx.random.normal((1, 1, 256))
        y2, cache = block(x2, cache=cache)

        assert y1.shape == (1, 10, 256)
        assert y2.shape == (1, 1, 256)
        assert cache.seq_len == 11


class TestGPTSoVITS:
    """Tests for full GPT model."""

    def test_gpt_forward(self):
        """Test GPT forward pass."""
        config = GPTConfig(
            hidden_size=256,
            num_layers=2,
            num_heads=4,
            intermediate_size=512,
            phoneme_vocab_size=100,
            semantic_vocab_size=200,
            text_feature_dim=128,  # BERT feature dimension
        )

        model = GPTSoVITS(config)

        # Create inputs
        phoneme_ids = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
        semantic_ids = mx.array([[0]], dtype=mx.int32)
        bert_features = mx.random.normal((1, 10, 128))  # BERT features

        # Forward pass
        logits, _ = model(phoneme_ids, semantic_ids, bert_features)

        assert logits.shape == (1, 1, 200)

    def test_gpt_with_cache(self):
        """Test GPT with KV cache for generation."""
        config = GPTConfig(
            hidden_size=256,
            num_layers=2,
            num_heads=4,
            intermediate_size=512,
            text_feature_dim=1024,  # BERT feature dimension
        )

        model = GPTSoVITS(config)
        caches = model.create_caches()

        phoneme_ids = mx.array([[1, 2, 3]], dtype=mx.int32)
        bert_features = mx.random.normal((1, 5, 1024))  # BERT features

        # First token
        semantic_ids = mx.array([[0]], dtype=mx.int32)
        logits1, caches = model(
            phoneme_ids, semantic_ids, bert_features, cache=caches
        )

        # Second token
        semantic_ids = mx.array([[10]], dtype=mx.int32)
        logits2, caches = model(
            phoneme_ids, semantic_ids, bert_features, cache=caches
        )

        assert logits1.shape == (1, 1, 1025)
        assert logits2.shape == (1, 1, 1025)
        assert all(c.seq_len == 2 for c in caches)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
