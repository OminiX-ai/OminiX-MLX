"""MLP module for GPT-SoVITS.

Implements feed-forward network with SwiGLU activation:
    FFN(x) = (SiLU(gate(x)) * up(x)) @ down

SwiGLU provides better performance than standard GELU FFN.
"""

import mlx.core as mx
import mlx.nn as nn


class MLP(nn.Module):
    """Feed-forward network with SwiGLU or GELU activation.

    SwiGLU mode: SiLU(gate(x)) * up(x) @ down
    GELU mode: GELU(gate(x)) @ down (standard 2-layer FFN)

    SwiGLU is more parameter-efficient and often performs better,
    but GELU mode is compatible with original GPT-SoVITS weights.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
        use_gelu: bool = False,
    ):
        """Initialize MLP.

        Args:
            hidden_size: Input/output dimension
            intermediate_size: Hidden layer dimension
            bias: Whether to use bias in linear layers
            use_gelu: Use GELU activation (original GPT-SoVITS style)
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.use_gelu = use_gelu

        # First projection (gate in SwiGLU, or just first layer in GELU)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)

        # Up projection (only used in SwiGLU mode)
        if not use_gelu:
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)

        # Down projection
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor [batch, seq, hidden]

        Returns:
            Output tensor [batch, seq, hidden]
        """
        if self.use_gelu:
            # Standard GELU FFN: GELU(x @ W1) @ W2
            return self.down_proj(nn.gelu(self.gate_proj(x)))
        else:
            # SwiGLU: SiLU(gate(x)) * up(x)
            gate = nn.silu(self.gate_proj(x))
            up = self.up_proj(x)
            return self.down_proj(gate * up)


class FusedSwiGLU(nn.Module):
    """Optimized SwiGLU that fuses gate and up projections.

    Instead of two separate Linear layers for gate and up, this uses
    a single larger Linear and splits the output. Can be more efficient
    on some hardware.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ):
        """Initialize fused SwiGLU.

        Args:
            hidden_size: Input/output dimension
            intermediate_size: Hidden layer dimension (per projection)
            bias: Whether to use bias
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Fused gate + up projection
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=bias)

        # Down projection
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor [batch, seq, hidden]

        Returns:
            Output tensor [batch, seq, hidden]
        """
        # Single projection, then split
        gate_up = self.gate_up_proj(x)
        gate, up = mx.split(gate_up, 2, axis=-1)

        # SwiGLU and project
        return self.down_proj(nn.silu(gate) * up)


class StandardMLP(nn.Module):
    """Standard FFN with GELU activation for comparison.

    FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = True,
    ):
        super().__init__()

        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))
