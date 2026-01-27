"""Attention module for GPT-SoVITS.

Implements multi-head self-attention with:
- Rotary Position Embeddings (RoPE)
- KV cache support for efficient generation
- Optional Flash Attention via MLX's scaled_dot_product_attention
"""

from typing import Optional, Tuple
import math
import mlx.core as mx
import mlx.nn as nn

from python.models.cache import KVCache


class RoPE(nn.Module):
    """Rotary Position Embeddings.

    Applies rotary embeddings to query and key tensors for relative
    position encoding without learned parameters.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequency bands
        # freqs = 1 / (theta ^ (2i / dim)) for i in [0, dim/2)
        freqs = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        self._freqs = freqs

    def __call__(
        self,
        x: mx.array,
        offset: int = 0,
    ) -> mx.array:
        """Apply rotary embeddings to input tensor.

        Args:
            x: Input tensor [batch, heads, seq, head_dim]
            offset: Position offset (for KV cache)

        Returns:
            Tensor with rotary embeddings applied
        """
        seq_len = x.shape[2]
        positions = mx.arange(offset, offset + seq_len, dtype=mx.float32)

        # [seq_len, dim/2]
        freqs = mx.outer(positions, self._freqs)

        # [seq_len, dim] - interleave cos and sin
        cos = mx.cos(freqs)
        sin = mx.sin(freqs)

        # Reshape for broadcasting: [1, 1, seq, dim/2]
        cos = cos.reshape(1, 1, seq_len, -1)
        sin = sin.reshape(1, 1, seq_len, -1)

        # Split x into even and odd indices
        x1, x2 = x[..., ::2], x[..., 1::2]

        # Apply rotation
        # rotated = [x1 * cos - x2 * sin, x1 * sin + x2 * cos]
        rotated = mx.concatenate(
            [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
            axis=-1,
        )

        return rotated.astype(x.dtype)


class Attention(nn.Module):
    """Multi-head self-attention with RoPE and KV cache support.

    This is the core attention mechanism for GPT-SoVITS. It supports:
    - Rotary position embeddings
    - KV cache for efficient autoregressive generation
    - Causal masking
    - Optional scaled_dot_product_attention for efficiency
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        rope_theta: float = 10000.0,
        max_seq_len: int = 2048,
        bias: bool = False,
    ):
        """Initialize attention module.

        Args:
            hidden_size: Model hidden dimension
            num_heads: Number of attention heads
            head_dim: Dimension per head (default: hidden_size // num_heads)
            rope_theta: RoPE theta parameter
            max_seq_len: Maximum sequence length for RoPE
            bias: Whether to use bias in projections
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)

        # Output projection
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=bias)

        # RoPE for position encoding
        self.rope = RoPE(self.head_dim, max_seq_len=max_seq_len, theta=rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> Tuple[mx.array, Optional[KVCache]]:
        """Forward pass for attention.

        Args:
            x: Input tensor [batch, seq, hidden]
            mask: Optional attention mask [1, 1, seq, seq] or None for causal
            cache: Optional KV cache for generation

        Returns:
            Tuple of (output tensor, updated cache)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to [batch, heads, seq, head_dim]
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Get position offset from cache
        offset = cache.seq_len if cache is not None else 0

        # Apply RoPE to Q and K
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        # Update KV cache
        if cache is not None:
            k, v = cache.update(k, v)

        # Compute attention scores
        # Use MLX's optimized scaled_dot_product_attention if available
        if hasattr(mx, 'fast') and hasattr(mx.fast, 'scaled_dot_product_attention'):
            # MLX's fused attention kernel
            output = mx.fast.scaled_dot_product_attention(
                q, k, v,
                scale=self.scale,
                mask=mask,
            )
        else:
            # Manual attention computation
            scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

            # Apply mask (causal by default)
            if mask is None:
                # Create causal mask
                kv_len = k.shape[2]
                q_len = q.shape[2]
                causal_mask = mx.triu(
                    mx.full((q_len, kv_len), float("-inf")),
                    k=kv_len - q_len + 1,
                )
                scores = scores + causal_mask
            elif mask is not None:
                scores = scores + mask

            # Softmax and attention
            attn_weights = mx.softmax(scores, axis=-1)
            output = attn_weights @ v

        # Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        # Output projection
        output = self.o_proj(output)

        return output, cache


class CrossAttention(nn.Module):
    """Cross-attention for conditioning on encoder outputs.

    Used for conditioning on audio features from CNHubert or text
    features from RoBERTa.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        encoder_dim: Optional[int] = None,
        bias: bool = False,
    ):
        """Initialize cross-attention module.

        Args:
            hidden_size: Decoder hidden dimension
            num_heads: Number of attention heads
            head_dim: Dimension per head
            encoder_dim: Encoder output dimension (default: same as hidden_size)
            bias: Whether to use bias
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.encoder_dim = encoder_dim or hidden_size
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Q from decoder, K/V from encoder
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(self.encoder_dim, num_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(self.encoder_dim, num_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=bias)

    def __call__(
        self,
        x: mx.array,
        encoder_output: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass for cross-attention.

        Args:
            x: Decoder input [batch, seq, hidden]
            encoder_output: Encoder output [batch, enc_seq, encoder_dim]
            mask: Optional attention mask

        Returns:
            Output tensor [batch, seq, hidden]
        """
        batch_size, seq_len, _ = x.shape
        enc_len = encoder_output.shape[1]

        # Project Q from decoder, K/V from encoder
        q = self.q_proj(x)
        k = self.k_proj(encoder_output)
        v = self.v_proj(encoder_output)

        # Reshape to [batch, heads, seq, head_dim]
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, enc_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, enc_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Attention
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale

        if mask is not None:
            scores = scores + mask

        attn_weights = mx.softmax(scores, axis=-1)
        output = attn_weights @ v

        # Reshape and project
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        output = self.o_proj(output)

        return output
