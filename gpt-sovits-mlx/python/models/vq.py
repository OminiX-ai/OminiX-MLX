"""Vector Quantization modules for SoVITS vocoder.

Implements Residual Vector Quantization (RVQ) with multiple codebooks
for high-quality audio compression and reconstruction.

Based on EnCodec's RVQ implementation, adapted for MLX.
"""

from typing import Optional, Tuple, List
import mlx.core as mx
import mlx.nn as nn


class VectorQuantizer(nn.Module):
    """Single codebook vector quantizer.

    Maps continuous embeddings to discrete codes using nearest-neighbor
    lookup in a learned codebook.
    """

    def __init__(
        self,
        codebook_size: int = 1024,
        codebook_dim: int = 256,
        commitment_weight: float = 0.25,
    ):
        """Initialize vector quantizer.

        Args:
            codebook_size: Number of codes in the codebook
            codebook_dim: Dimension of each code vector
            commitment_weight: Weight for commitment loss (training only)
        """
        super().__init__()

        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_weight = commitment_weight

        # Codebook embeddings
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def encode(self, x: mx.array) -> mx.array:
        """Encode continuous embeddings to discrete codes.

        Args:
            x: Input tensor [batch, seq, dim]

        Returns:
            Code indices [batch, seq]
        """
        batch, seq, dim = x.shape

        # Flatten for distance computation
        x_flat = x.reshape(-1, dim)  # [batch*seq, dim]

        # Get codebook vectors
        codebook = self.codebook.weight  # [codebook_size, dim]

        # Compute distances: ||x - c||^2 = ||x||^2 + ||c||^2 - 2*x.c
        x_norm = mx.sum(x_flat ** 2, axis=1, keepdims=True)  # [batch*seq, 1]
        c_norm = mx.sum(codebook ** 2, axis=1, keepdims=True).T  # [1, codebook_size]
        distances = x_norm + c_norm - 2 * (x_flat @ codebook.T)  # [batch*seq, codebook_size]

        # Find nearest code
        indices = mx.argmin(distances, axis=1)  # [batch*seq]

        return indices.reshape(batch, seq)

    def decode(self, indices: mx.array) -> mx.array:
        """Decode discrete codes to continuous embeddings.

        Args:
            indices: Code indices [batch, seq]

        Returns:
            Quantized embeddings [batch, seq, dim]
        """
        return self.codebook(indices)

    def __call__(
        self,
        x: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Forward pass: encode and decode.

        Args:
            x: Input tensor [batch, seq, dim]

        Returns:
            Tuple of (quantized, indices, commitment_loss)
        """
        indices = self.encode(x)
        quantized = self.decode(indices)

        # Commitment loss (for training)
        commitment_loss = mx.mean((x - mx.stop_gradient(quantized)) ** 2)

        # Straight-through estimator: gradient flows through quantized
        quantized = x + mx.stop_gradient(quantized - x)

        return quantized, indices, commitment_loss


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer with multiple codebooks.

    Iteratively quantizes the residual from previous codebooks,
    allowing for fine-grained audio reconstruction.

    Used in SoVITS for semantic-to-acoustic mapping.
    """

    def __init__(
        self,
        num_codebooks: int = 8,
        codebook_size: int = 1024,
        codebook_dim: int = 256,
    ):
        """Initialize RVQ.

        Args:
            num_codebooks: Number of residual codebooks
            codebook_size: Size of each codebook
            codebook_dim: Dimension of code vectors
        """
        super().__init__()

        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        # Create quantizers for each level
        self.quantizers = [
            VectorQuantizer(codebook_size, codebook_dim)
            for _ in range(num_codebooks)
        ]

    def encode(self, x: mx.array) -> List[mx.array]:
        """Encode to multiple codebook indices.

        Args:
            x: Input tensor [batch, seq, dim]

        Returns:
            List of code indices, one per codebook
        """
        codes = []
        residual = x

        for quantizer in self.quantizers:
            indices = quantizer.encode(residual)
            quantized = quantizer.decode(indices)
            residual = residual - quantized
            codes.append(indices)

        return codes

    def decode(self, codes: List[mx.array]) -> mx.array:
        """Decode from multiple codebook indices.

        Args:
            codes: List of code indices [batch, seq] per codebook

        Returns:
            Reconstructed embeddings [batch, seq, dim]
        """
        quantized = mx.zeros_like(self.quantizers[0].decode(codes[0]))

        for quantizer, indices in zip(self.quantizers, codes):
            quantized = quantized + quantizer.decode(indices)

        return quantized

    def decode_from_semantic(
        self,
        semantic_tokens: mx.array,
        num_codebooks: Optional[int] = None,
    ) -> mx.array:
        """Decode semantic tokens using first N codebooks.

        In SoVITS, semantic tokens correspond to the first codebook.
        Additional codebooks add acoustic detail.

        Args:
            semantic_tokens: Semantic token indices [batch, seq]
            num_codebooks: Number of codebooks to use (default: all)

        Returns:
            Decoded embeddings [batch, seq, dim]
        """
        n = num_codebooks or self.num_codebooks

        # First codebook from semantic tokens
        quantized = self.quantizers[0].decode(semantic_tokens)

        # Additional codebooks would need their own tokens
        # For now, return just the first codebook output
        return quantized

    def __call__(
        self,
        x: mx.array,
        num_codebooks: Optional[int] = None,
    ) -> Tuple[mx.array, List[mx.array], mx.array]:
        """Forward pass with residual quantization.

        Args:
            x: Input tensor [batch, seq, dim]
            num_codebooks: Number of codebooks to use

        Returns:
            Tuple of (quantized, codes_list, total_loss)
        """
        n = num_codebooks or self.num_codebooks
        codes = []
        total_loss = mx.array(0.0)
        quantized = mx.zeros_like(x)
        residual = x

        for i, quantizer in enumerate(self.quantizers[:n]):
            q, indices, loss = quantizer(residual)
            residual = residual - mx.stop_gradient(q)
            quantized = quantized + q
            codes.append(indices)
            total_loss = total_loss + loss

        return quantized, codes, total_loss / n


class SemanticToAcoustic(nn.Module):
    """Maps semantic tokens to acoustic features via RVQ.

    This is the core of the SoVITS decoder - it takes semantic tokens
    from the GPT stage and produces acoustic features for the vocoder.
    """

    def __init__(
        self,
        semantic_dim: int = 512,
        acoustic_dim: int = 256,
        num_codebooks: int = 8,
        codebook_size: int = 1024,
    ):
        """Initialize semantic-to-acoustic mapper.

        Args:
            semantic_dim: Dimension of semantic embeddings (from GPT)
            acoustic_dim: Dimension of acoustic features (for vocoder)
            num_codebooks: Number of RVQ codebooks
            codebook_size: Size of each codebook
        """
        super().__init__()

        self.semantic_dim = semantic_dim
        self.acoustic_dim = acoustic_dim

        # Project semantic to acoustic space
        self.semantic_proj = nn.Linear(semantic_dim, acoustic_dim)

        # RVQ for acoustic encoding
        self.rvq = ResidualVectorQuantizer(
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            codebook_dim=acoustic_dim,
        )

        # Output projection
        self.output_proj = nn.Linear(acoustic_dim, acoustic_dim)

    def __call__(
        self,
        semantic_tokens: mx.array,
        semantic_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        """Convert semantic tokens to acoustic features.

        Args:
            semantic_tokens: Token indices from GPT [batch, seq]
            semantic_embeddings: Optional pre-computed embeddings

        Returns:
            Acoustic features [batch, seq, acoustic_dim]
        """
        # Decode from first RVQ codebook
        acoustic = self.rvq.quantizers[0].decode(semantic_tokens)

        # Apply output projection
        acoustic = self.output_proj(acoustic)

        return acoustic
