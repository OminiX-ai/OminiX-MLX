"""Duration prediction and length regulation for SoVITS vocoder.

Implements:
- StochasticDurationPredictor: Flow-based duration prediction
- DurationPredictor: Deterministic CNN-based predictor
- LengthRegulator: Expands features based on durations

Note: MLX Conv1d uses channels-last format [batch, time, channels].
"""

from typing import Optional, Tuple
import mlx.core as mx
import mlx.nn as nn


class ConvBlock(nn.Module):
    """Convolutional block with normalization and activation.

    Uses MLX's channels-last format: [batch, time, channels]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
        )
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        """Apply conv block.

        Args:
            x: Input [batch, time, channels] (channels-last)

        Returns:
            Output [batch, time, out_channels]
        """
        x = self.conv(x)
        x = self.norm(x)
        x = nn.relu(x)
        x = self.dropout(x)
        return x


class DurationPredictor(nn.Module):
    """Deterministic duration predictor using CNNs.

    Predicts duration (number of acoustic frames) for each input token.
    Simpler alternative to StochasticDurationPredictor.

    Uses MLX's channels-last format: [batch, time, channels]
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 256,
        kernel_size: int = 3,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        """Initialize duration predictor.

        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            kernel_size: Convolution kernel size
            num_layers: Number of conv layers
            dropout: Dropout probability
        """
        super().__init__()

        self.layers = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            self.layers.append(
                ConvBlock(in_ch, hidden_channels, kernel_size, dropout=dropout)
            )

        # Output projection to scalar duration
        self.proj = nn.Conv1d(hidden_channels, 1, 1)

    def __call__(self, x: mx.array, x_mask: Optional[mx.array] = None) -> mx.array:
        """Predict durations.

        Args:
            x: Input features [batch, time, channels] (channels-last)
            x_mask: Optional mask [batch, time, 1]

        Returns:
            Durations [batch, time] (in log scale)
        """
        for layer in self.layers:
            x = layer(x)
            if x_mask is not None:
                x = x * x_mask

        # Project to duration (log scale)
        duration = self.proj(x)  # [batch, time, 1]
        duration = duration.squeeze(-1)  # [batch, time]

        return duration


class FlowLayer(nn.Module):
    """Single normalizing flow layer for stochastic duration prediction.

    Implements affine coupling layer with masking.
    Uses MLX's channels-last format: [batch, time, channels]
    """

    def __init__(
        self,
        channels: int = 256,
        hidden_channels: int = 256,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.channels = channels
        self.half_channels = channels // 2

        # Conditioning network
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.conv = nn.Conv1d(
            hidden_channels, hidden_channels, kernel_size,
            padding=kernel_size // 2
        )
        self.proj = nn.Conv1d(hidden_channels, self.half_channels * 2, 1)

    def __call__(
        self,
        x: mx.array,
        reverse: bool = False,
    ) -> Tuple[mx.array, mx.array]:
        """Apply flow transformation.

        Args:
            x: Input [batch, time, channels] (channels-last)
            reverse: Whether to apply inverse transformation

        Returns:
            Tuple of (output, log_det_jacobian)
        """
        # Split input along channel dimension (last axis)
        x0, x1 = mx.split(x, 2, axis=-1)

        # Compute affine parameters from x0
        h = self.pre(x0)
        h = nn.relu(h)
        h = self.conv(h)
        h = nn.relu(h)
        params = self.proj(h)  # [batch, time, half_channels*2]

        # Split into scale and shift
        log_s, t = mx.split(params, 2, axis=-1)

        if not reverse:
            # Forward: y1 = x1 * exp(s) + t
            y1 = x1 * mx.exp(log_s) + t
            log_det = mx.sum(log_s, axis=(1, 2))
        else:
            # Inverse: x1 = (y1 - t) * exp(-s)
            y1 = (x1 - t) * mx.exp(-log_s)
            log_det = -mx.sum(log_s, axis=(1, 2))

        y = mx.concatenate([x0, y1], axis=-1)
        return y, log_det


class StochasticDurationPredictor(nn.Module):
    """Stochastic duration predictor using normalizing flows.

    Models duration as a distribution, allowing for varied prosody.
    During inference, samples from the distribution.

    Uses MLX's channels-last format: [batch, time, channels]
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 256,
        kernel_size: int = 3,
        num_flows: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize stochastic duration predictor.

        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            kernel_size: Convolution kernel size
            num_flows: Number of flow layers
            dropout: Dropout probability
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        # Input projection
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)

        # Flow layers
        self.flows = [
            FlowLayer(hidden_channels, hidden_channels, kernel_size)
            for _ in range(num_flows)
        ]

        # Duration output projection
        self.post = nn.Conv1d(hidden_channels // 2, 1, 1)

    def __call__(
        self,
        x: mx.array,
        x_mask: Optional[mx.array] = None,
        noise_scale: float = 1.0,
    ) -> mx.array:
        """Predict durations stochastically.

        Args:
            x: Input features [batch, time, channels] (channels-last)
            x_mask: Optional mask [batch, time, 1]
            noise_scale: Scale for sampling noise

        Returns:
            Durations [batch, time]
        """
        # Project input
        x = self.pre(x)  # [batch, time, hidden]

        if x_mask is not None:
            x = x * x_mask

        # Sample noise [batch, time, channels]
        batch, time, channels = x.shape
        noise = mx.random.normal((batch, time, channels)) * noise_scale

        # Concatenate with noise for flow input
        # Actually for SDP we run flows backwards from noise
        z = noise

        # Run flows in reverse (sampling)
        for flow in reversed(self.flows):
            z, _ = flow(z, reverse=True)

        # Extract duration from first half of channels
        duration = z[:, :, :channels // 2]
        duration = self.post(duration).squeeze(-1)  # [batch, time]

        # Ensure positive durations
        duration = mx.softplus(duration)

        return duration


class LengthRegulator(nn.Module):
    """Expands features based on predicted durations.

    Takes a sequence and duration for each element, outputs expanded
    sequence where each element is repeated according to its duration.

    Uses MLX's channels-last format: [batch, time, channels]
    """

    def __init__(self, pad_value: float = 0.0):
        super().__init__()
        self.pad_value = pad_value

    def __call__(
        self,
        x: mx.array,
        durations: mx.array,
        max_len: Optional[int] = None,
    ) -> Tuple[mx.array, mx.array]:
        """Regulate length based on durations.

        Args:
            x: Input features [batch, time, channels] (channels-last)
            durations: Duration for each frame [batch, time]
            max_len: Maximum output length (optional)

        Returns:
            Tuple of (expanded features [batch, expanded_time, channels], output lengths)
        """
        batch_size, in_time, channels = x.shape

        # Round durations to integers
        durations = mx.round(durations).astype(mx.int32)
        durations = mx.maximum(durations, 1)  # Minimum duration of 1

        # Calculate output lengths
        output_lens = mx.sum(durations, axis=1)  # [batch]
        max_out_len = max_len or int(mx.max(output_lens).item())

        # Expand each sequence
        outputs = []
        for b in range(batch_size):
            # Get features and durations for this batch
            feat = x[b]  # [in_time, channels]
            dur = durations[b]  # [in_time]

            # Build expanded sequence
            expanded = []
            for t in range(in_time):
                d = int(dur[t].item())
                frame = feat[t:t+1, :]  # [1, channels]
                expanded.append(mx.repeat(frame, d, axis=0))

            if expanded:
                expanded = mx.concatenate(expanded, axis=0)  # [out_time, channels]
            else:
                expanded = mx.zeros((0, channels))

            # Pad or truncate to max_out_len
            out_time = expanded.shape[0]
            if out_time < max_out_len:
                pad = mx.full((max_out_len - out_time, channels), self.pad_value)
                expanded = mx.concatenate([expanded, pad], axis=0)
            elif out_time > max_out_len:
                expanded = expanded[:max_out_len, :]

            outputs.append(expanded)

        # Stack batches
        output = mx.stack(outputs, axis=0)  # [batch, max_out_len, channels]

        return output, output_lens


def regulate_length(
    x: mx.array,
    durations: mx.array,
    target_len: Optional[int] = None,
) -> mx.array:
    """Simple length regulation function.

    Args:
        x: Input [batch, seq, channels] (channels-last)
        durations: Durations [batch, seq]
        target_len: Target output length

    Returns:
        Expanded output [batch, target_len, channels]
    """
    regulator = LengthRegulator()

    # Input is already in channels-last format
    output, _ = regulator(x, durations, max_len=target_len)

    return output  # [batch, target_len, channels]
