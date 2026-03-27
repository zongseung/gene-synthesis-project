"""
DiT (Diffusion Transformer) modules: patch embedding, AdaLN-Zero blocks,
and the DiT core stack.

Domain rationale:
- DiT captures long-range inter-gene interactions via self-attention, which
  CNN's local receptive field cannot reach.
- AdaLN-Zero (FiLM + zero-init gate) ensures DiT starts as an identity
  function and gradually learns corrections on top of CNN features.
"""

import torch
import torch.nn as nn


class PatchEmbed1D(nn.Module):
    """
    Convert CNN encoder output to DiT patch tokens.

    (B, C, L) -> reshape to patches -> linear projection -> (B, N_tokens, d_model)
    with learnable positional embeddings.
    """

    def __init__(
        self,
        seq_len: int,
        in_channels: int,
        patch_size: int = 16,
        d_model: int = 256,
    ):
        super().__init__()
        assert seq_len % patch_size == 0, (
            f"seq_len {seq_len} not divisible by patch_size {patch_size}"
        )
        self.patch_size = patch_size
        self.n_tokens = seq_len // patch_size
        self.proj = nn.Linear(in_channels * patch_size, d_model)
        self.pos_emb = nn.Parameter(
            torch.randn(1, self.n_tokens, d_model) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, L) CNN encoder output.

        Returns:
            (B, N_tokens, d_model) patch token sequence.
        """
        B, C, L = x.shape
        x = x.reshape(B, C, self.n_tokens, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(B, self.n_tokens, -1)
        x = self.proj(x) + self.pos_emb
        return x


class UnPatchify1D(nn.Module):
    """
    Convert DiT output tokens back to spatial representation.

    (B, N_tokens, d_model) -> linear -> reshape -> (B, C, L)
    """

    def __init__(
        self,
        n_tokens: int,
        out_channels: int,
        patch_size: int = 16,
        d_model: int = 256,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.proj = nn.Linear(d_model, out_channels * patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N_tokens, d_model) DiT output.

        Returns:
            (B, C, L) spatial feature map.
        """
        B = x.shape[0]
        x = self.proj(x)  # (B, N, C * patch_size)
        x = x.reshape(B, self.n_tokens, self.out_channels, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(
            B, self.out_channels, self.n_tokens * self.patch_size
        )
        return x


class DiTBlock(nn.Module):
    """
    AdaLN-Zero DiT Block (Peebles & Xie, ICCV 2023).

    Key design:
    - LayerNorm with elementwise_affine=False (no learned scale/bias).
    - FiLM parameters (gamma, beta) predicted from conditioning signal.
    - Alpha gate on residual initialized to zero, so DiT starts as identity.
    - This means at initialization, the model output equals the CNN features,
      and DiT gradually learns long-range corrections during training.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)

        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, film_params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, d_model) token sequence.
            film_params: (B, d_model*6) containing
                gamma1, beta1, alpha1 (for attention) +
                gamma2, beta2, alpha2 (for FFN).

        Returns:
            (B, N, d_model) processed token sequence.
        """
        d = x.shape[-1]
        assert film_params.shape[-1] == d * 6, (
            f"FiLM params dim {film_params.shape[-1]} != expected {d * 6}"
        )

        # Split into 6 modulation vectors, each (B, 1, d)
        g1, b1, a1, g2, b2, a2 = [
            p.unsqueeze(1) for p in film_params.chunk(6, dim=-1)
        ]

        # Self-Attention with AdaLN-Zero
        h = self.norm1(x)
        h = g1 * h + b1  # FiLM modulation
        h, _ = self.attn(h, h, h)
        x = x + a1 * h  # Gated residual (alpha initialized to 0)

        # Feed-Forward with AdaLN-Zero
        h = self.norm2(x)
        h = g2 * h + b2
        h = self.mlp(h)
        x = x + a2 * h

        return x


class DiTCore(nn.Module):
    """
    Stack of DiTBlock layers forming the transformer core.

    Processes patch tokens with population-conditioned self-attention to capture
    long-range inter-gene interactions.
    """

    def __init__(
        self,
        n_blocks: int = 4,
        d_model: int = 256,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                DiTBlock(d_model, n_heads, mlp_ratio, dropout)
                for _ in range(n_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, dit_params: list) -> torch.Tensor:
        """
        Args:
            x: (B, N_tokens, d_model) patch token sequence.
            dit_params: list of (B, d_model*6) FiLM params per block.

        Returns:
            (B, N_tokens, d_model) processed tokens.
        """
        assert len(dit_params) == len(self.blocks), (
            f"DiT params count {len(dit_params)} != blocks {len(self.blocks)}"
        )
        for block, params in zip(self.blocks, dit_params):
            x = block(x, params)
        return self.final_norm(x)
