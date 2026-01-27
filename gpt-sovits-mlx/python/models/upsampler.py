"""Upsampler modules for SoVITS vocoder.

Implements transposed convolution-based upsampling network that
converts acoustic features to audio waveforms.

Based on HiFi-GAN/EnCodec decoder architecture, adapted for MLX.
"""

from typing import List, Tuple, Optional
import mlx.core as mx
import mlx.nn as nn


class ConvTranspose1d(nn.Module):
    """1D Transposed Convolution for upsampling.

    MLX doesn't have built-in ConvTranspose1d, so we implement it
    using Conv1d with output padding and reshaping.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        # Weight shape: [out_channels, in_channels, kernel_size]
        # For transposed conv, we flip the channel dimensions
        scale = 1.0 / (in_channels * kernel_size) ** 0.5
        self.weight = mx.random.normal((in_channels, out_channels, kernel_size)) * scale
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array) -> mx.array:
        """Apply transposed convolution.

        Args:
            x: Input [batch, in_channels, time]

        Returns:
            Output [batch, out_channels, time * stride]
        """
        batch, in_ch, in_time = x.shape

        # Calculate output size
        out_time = (in_time - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding

        # Approach: insert zeros between samples, then apply regular conv
        # This is a simplified implementation

        # Upsample by inserting zeros
        if self.stride > 1:
            upsampled = mx.zeros((batch, in_ch, in_time * self.stride))
            # Place original samples at stride intervals
            indices = mx.arange(in_time) * self.stride
            for i in range(in_time):
                upsampled = upsampled.at[:, :, i * self.stride].set(x[:, :, i])
        else:
            upsampled = x

        # Apply convolution with transposed weights
        # weight: [in_ch, out_ch, kernel] -> [out_ch, in_ch, kernel] for conv
        weight_t = self.weight.transpose(1, 0, 2)

        # Pad input
        pad_total = self.kernel_size - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        # Apply padding
        upsampled = mx.pad(upsampled, [(0, 0), (0, 0), (pad_left, pad_right)])

        # Manual convolution
        out_time_actual = upsampled.shape[2] - self.kernel_size + 1
        output = mx.zeros((batch, self.out_channels, out_time_actual))

        for t in range(out_time_actual):
            window = upsampled[:, :, t:t + self.kernel_size]  # [batch, in_ch, kernel]
            # Einsum: batch,in_ch,kernel @ out_ch,in_ch,kernel -> batch,out_ch
            out_t = mx.sum(window[:, :, None, :] * weight_t[None, :, :, :], axis=(1, 3))
            output = output.at[:, :, t].set(out_t)

        # Add bias
        output = output + self.bias.reshape(1, -1, 1)

        return output


class ResBlock(nn.Module):
    """Residual block with dilated convolutions.

    Used in the upsampler for multi-resolution processing.
    Uses MLX's channels-last format: [batch, time, channels]
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: Tuple[int, ...] = (1, 3, 5),
    ):
        """Initialize residual block.

        Args:
            channels: Number of channels
            kernel_size: Convolution kernel size
            dilations: Tuple of dilation rates
        """
        super().__init__()

        self.convs1 = []
        self.convs2 = []

        for dilation in dilations:
            padding = (kernel_size - 1) * dilation // 2
            self.convs1.append(
                nn.Conv1d(channels, channels, kernel_size, padding=padding)
            )
            self.convs2.append(
                nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
            )

    def __call__(self, x: mx.array) -> mx.array:
        """Apply residual block.

        Args:
            x: Input [batch, time, channels] (channels-last)

        Returns:
            Output with same shape
        """
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = x
            x = nn.leaky_relu(x, negative_slope=0.1)
            x = conv1(x)
            x = nn.leaky_relu(x, negative_slope=0.1)
            x = conv2(x)
            x = x + residual

        return x


class MultiReceptiveFieldFusion(nn.Module):
    """Multi-receptive field fusion module.

    Combines outputs from multiple ResBlocks with different kernel sizes
    to capture patterns at different scales.
    Uses MLX's channels-last format: [batch, time, channels]
    """

    def __init__(
        self,
        channels: int,
        kernel_sizes: Tuple[int, ...] = (3, 7, 11),
        dilations: Tuple[Tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    ):
        super().__init__()

        self.resblocks = [
            ResBlock(channels, kernel_size, dils)
            for kernel_size, dils in zip(kernel_sizes, dilations)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        """Apply MRFF.

        Args:
            x: Input [batch, time, channels] (channels-last)

        Returns:
            Fused output [batch, time, channels]
        """
        outputs = [resblock(x) for resblock in self.resblocks]
        return sum(outputs) / len(outputs)


class Upsampler(nn.Module):
    """Upsampling network for vocoder.

    Converts low-resolution acoustic features to high-resolution audio.
    Uses transposed convolutions for upsampling and ResBlocks for
    multi-resolution processing.

    Uses MLX's channels-last format: [batch, time, channels]
    """

    def __init__(
        self,
        in_channels: int = 256,
        upsample_rates: Tuple[int, ...] = (8, 4, 2, 2),
        upsample_kernel_sizes: Tuple[int, ...] = (16, 8, 4, 4),
        upsample_channels: int = 256,
        resblock_kernel_sizes: Tuple[int, ...] = (3, 7, 11),
        resblock_dilations: Tuple[Tuple[int, ...], ...] = (
            (1, 3, 5), (1, 3, 5), (1, 3, 5)
        ),
    ):
        """Initialize upsampler.

        Args:
            in_channels: Input feature dimension
            upsample_rates: Upsampling rate at each stage
            upsample_kernel_sizes: Kernel size for each upsample stage
            upsample_channels: Hidden channel dimension
            resblock_kernel_sizes: Kernel sizes for ResBlocks
            resblock_dilations: Dilation patterns for ResBlocks
        """
        super().__init__()

        self.num_upsamples = len(upsample_rates)
        self.total_upsample = 1
        for r in upsample_rates:
            self.total_upsample *= r

        # Input projection
        self.conv_pre = nn.Conv1d(in_channels, upsample_channels, 7, padding=3)

        # Upsample stages
        self.upsamples = []
        self.resblocks = []

        channels = upsample_channels
        for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            # Halve channels at each stage (optional, can keep constant)
            out_channels = channels // 2 if i < len(upsample_rates) - 1 else channels // 2

            # Use simple upsampling instead of ConvTranspose1d for now
            # (ConvTranspose1d implementation is complex)
            self.upsamples.append((rate, channels, out_channels))

            # ResBlock fusion after each upsample
            self.resblocks.append(
                MultiReceptiveFieldFusion(out_channels, resblock_kernel_sizes, resblock_dilations)
            )

            channels = out_channels

        # Store final channel count
        self.final_channels = channels

        # Output projection to waveform
        self.conv_post = nn.Conv1d(channels, 1, 7, padding=3)

    def _upsample(self, x: mx.array, rate: int) -> mx.array:
        """Simple upsampling by repetition.

        Args:
            x: Input [batch, time, channels] (channels-last)
            rate: Upsample rate

        Returns:
            Upsampled output [batch, time * rate, channels]
        """
        batch, time, channels = x.shape

        # Repeat each timestep
        x = x[:, :, None, :]  # [batch, time, 1, channels]
        x = mx.repeat(x, rate, axis=2)  # [batch, time, rate, channels]
        x = x.reshape(batch, time * rate, channels)

        return x

    def __call__(self, x: mx.array) -> mx.array:
        """Upsample acoustic features to audio.

        Args:
            x: Acoustic features [batch, time, in_channels] (channels-last)

        Returns:
            Audio waveform [batch, time * total_upsample, 1]
        """
        # Initial projection
        x = self.conv_pre(x)
        x = nn.leaky_relu(x, negative_slope=0.1)

        # Upsample stages
        for (rate, in_ch, out_ch), resblock in zip(self.upsamples, self.resblocks):
            # Upsample
            x = self._upsample(x, rate)

            # Channel projection (simple linear for now)
            if in_ch != out_ch:
                # Average pool channels
                x = x.reshape(x.shape[0], x.shape[1], out_ch, -1)
                x = mx.mean(x, axis=-1)

            # Apply ResBlock
            x = resblock(x)

        # Output projection
        x = nn.leaky_relu(x, negative_slope=0.1)
        x = self.conv_post(x)
        x = mx.tanh(x)

        return x


class SimplifiedUpsampler(nn.Module):
    """Simplified upsampler using interpolation.

    A lighter-weight alternative that uses linear interpolation
    instead of learned transposed convolutions.

    Uses MLX's channels-last format: [batch, time, channels]
    """

    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 256,
        out_channels: int = 1,
        upsample_factor: int = 256,  # 16kHz features to 32kHz audio, 50Hz -> 32kHz
        num_layers: int = 4,
    ):
        super().__init__()

        self.upsample_factor = upsample_factor

        # Feature refinement
        self.layers = []
        channels = in_channels
        for i in range(num_layers):
            out_ch = hidden_channels if i < num_layers - 1 else out_channels
            self.layers.append(nn.Conv1d(channels, out_ch, 3, padding=1))
            channels = out_ch

    def __call__(self, x: mx.array) -> mx.array:
        """Upsample features to audio.

        Args:
            x: Features [batch, time, channels] (channels-last)

        Returns:
            Audio [batch, time * upsample_factor, 1]
        """
        # Upsample via repetition
        batch, time, channels = x.shape
        target_time = time * self.upsample_factor

        # Simple nearest-neighbor upsampling (channels-last)
        x = x[:, :, None, :]  # [batch, time, 1, channels]
        x = mx.repeat(x, self.upsample_factor, axis=2)  # [batch, time, upsample_factor, channels]
        x = x.reshape(batch, target_time, channels)

        # Apply refinement layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = nn.leaky_relu(x, negative_slope=0.1)

        # Final activation
        x = mx.tanh(x)

        return x
