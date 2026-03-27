"""
CNN modules: FiLM-conditioned convolutional blocks, encoder, and decoder.

Domain rationale:
- CNN captures local linkage disequilibrium (LD) patterns and haplotype block
  structure via local receptive fields.
- Downsampling reduces the sequence length so the DiT core operates on a
  manageable number of tokens.
- FiLM modulation (gamma * h + beta) allows population-specific adjustment of
  feature statistics at every block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMConvBlock(nn.Module):
    """
    FiLM-modulated 1-D convolutional block.

    Conv1d -> GroupNorm -> FiLM(gamma, beta) -> SiLU -> Conv1d -> GroupNorm -> SiLU
    + residual connection + optional stride-2 downsample.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        downsample: bool = True,
    ):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.norm1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.downsample = (
            nn.Conv1d(out_ch, out_ch, 2, stride=2) if downsample else nn.Identity()
        )
        self.skip_proj = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C_in, L) input feature map.
            gamma: (B, C_out) FiLM scale.
            beta: (B, C_out) FiLM shift.

        Returns:
            out: (B, C_out, L') downsampled output (L' = L/2 or L).
            skip: (B, C_out, L) skip connection (before downsampling).
        """
        residual = self.skip_proj(x)

        h = self.conv1(x)
        h = self.norm1(h)
        # FiLM modulation: channel-wise affine transform
        h = gamma.unsqueeze(-1) * h + beta.unsqueeze(-1)
        h = F.silu(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        # Residual (handle potential length mismatch from convolution)
        skip = h + residual[..., : h.shape[-1]]
        out = self.downsample(h)
        return out, skip


class FiLMDeconvBlock(nn.Module):
    """
    FiLM-modulated 1-D transposed convolutional block (decoder counterpart).

    Upsample -> Concat(skip) -> Conv1d -> GroupNorm -> FiLM -> SiLU
    -> Conv1d -> GroupNorm -> SiLU + residual.
    """

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        upsample: bool = True,
    ):
        super().__init__()
        pad = kernel_size // 2
        self.upsample = (
            nn.ConvTranspose1d(in_ch, in_ch, 2, stride=2) if upsample else nn.Identity()
        )
        # After concatenation with skip, input channels = in_ch + skip_ch
        concat_ch = in_ch + skip_ch
        self.conv1 = nn.Conv1d(concat_ch, out_ch, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.norm1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.skip_proj = (
            nn.Conv1d(concat_ch, out_ch, 1) if concat_ch != out_ch else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, L) input from previous decoder block.
            skip: (B, skip_ch, L_skip) skip connection from encoder.
            gamma: (B, C_out) FiLM scale.
            beta: (B, C_out) FiLM shift.

        Returns:
            (B, C_out, L_out) decoded feature map.
        """
        h = self.upsample(x)
        # Align lengths before concatenation
        if h.shape[-1] != skip.shape[-1]:
            h = F.pad(h, (0, skip.shape[-1] - h.shape[-1]))
        h = torch.cat([h, skip], dim=1)

        residual = self.skip_proj(h)

        h = self.conv1(h)
        h = self.norm1(h)
        h = gamma.unsqueeze(-1) * h + beta.unsqueeze(-1)
        h = F.silu(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + residual[..., : h.shape[-1]]


class CNNStemEncoder(nn.Module):
    """
    CNN encoder: (B, K, gene_size) -> (B, 4C, gene_size/8) + skip connections.

    Progressive downsampling: K -> C -> C -> 2C -> 4C with stride-2 convs.
    Each block receives FiLM params from UnifiedFiLMGenerator.

    Domain rationale: CNN captures local LD patterns and haplotype chunk
    structure. Downsampling reduces sequence length for DiT token count.
    """

    def __init__(
        self,
        in_channels: int = 8,
        base_channels: int = 64,
        channel_mult: tuple = (1, 1, 2, 4),
        kernel_size: int = 3,
    ):
        super().__init__()
        channels = [base_channels * m for m in channel_mult]
        self.blocks = nn.ModuleList()
        ch_in = in_channels
        for i, ch_out in enumerate(channels):
            # Last block does not downsample
            downsample = i < len(channels) - 1
            self.blocks.append(
                FiLMConvBlock(ch_in, ch_out, kernel_size, downsample)
            )
            ch_in = ch_out

    def forward(
        self, x: torch.Tensor, cnn_film_params: list
    ) -> tuple[torch.Tensor, list]:
        """
        Args:
            x: (B, K, gene_size) input Gene PCA tensor.
            cnn_film_params: list of (gamma, beta) per block from FiLM generator.

        Returns:
            x: (B, 4C, L_down) latent representation.
            skips: list of skip connection tensors (one per block).
        """
        assert len(cnn_film_params) == len(self.blocks), (
            f"FiLM params count {len(cnn_film_params)} != blocks {len(self.blocks)}"
        )
        skips = []
        for block, (gamma, beta) in zip(self.blocks, cnn_film_params):
            x, skip = block(x, gamma, beta)
            skips.append(skip)
        return x, skips


class CNNDecoder(nn.Module):
    """
    CNN decoder: symmetric to encoder with transposed convolutions + skip
    connections.

    (B, 4C, L_down) + skips -> (B, K, gene_size).
    """

    def __init__(
        self,
        out_channels: int = 8,
        base_channels: int = 64,
        channel_mult: tuple = (1, 1, 2, 4),
        kernel_size: int = 3,
    ):
        super().__init__()
        # Decoder mirrors encoder in reverse: 4C -> 2C -> C -> C -> K
        channels = [base_channels * m for m in channel_mult]
        reversed_channels = list(reversed(channels))

        self.blocks = nn.ModuleList()
        self.block_out_channels = []  # Track actual out_ch for FiLM param sizing
        for i in range(len(reversed_channels)):
            in_ch = reversed_channels[i]
            skip_ch = reversed_channels[i]  # Skip from corresponding encoder block
            if i < len(reversed_channels) - 1:
                out_ch = reversed_channels[i + 1]
                upsample = True
            else:
                out_ch = reversed_channels[i]
                upsample = False
            self.blocks.append(
                FiLMDeconvBlock(in_ch, skip_ch, out_ch, kernel_size, upsample)
            )
            self.block_out_channels.append(out_ch)

        # Final projection back to input channels
        self.final_conv = nn.Conv1d(reversed_channels[-1], out_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        skips: list,
        cnn_film_params: list,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, 4C, L_down) latent from DiT.
            skips: list of skip connections from encoder (reversed order).
            cnn_film_params: list of (gamma, beta) per decoder block.

        Returns:
            (B, K, gene_size) reconstructed output.
        """
        assert len(cnn_film_params) == len(self.blocks), (
            f"FiLM params count {len(cnn_film_params)} != blocks {len(self.blocks)}"
        )
        # Skips are in encoder order; decoder processes in reverse
        reversed_skips = list(reversed(skips))

        for block, skip, (gamma, beta) in zip(
            self.blocks, reversed_skips, cnn_film_params
        ):
            x = block(x, skip, gamma, beta)

        return self.final_conv(x)
