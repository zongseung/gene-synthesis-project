"""
HybridCNNDiTFiLM: Main model assembling CNN encoder, DiT core, and CNN decoder
with unified FiLM conditioning.

Data flow:
    (B, K, 26624) -> CNN encoder [FiLM] -> (B, 4C, 3328) -> Patchify
    -> (B, 208, d) -> DiT [AdaLN-Zero] -> Un-patchify
    -> CNN decoder [FiLM] + skips -> (B, K, 26624)

Conditioning path:
    pop_label(0-25) -> HierarchicalPopulationEmbedding(pop + superpop)
    -> + timestep_emb -> UnifiedFiLMGenerator
    -> per-block (gamma, beta) for CNN + (gamma, beta, alpha) for DiT
"""

import torch
import torch.nn as nn

from .modules.conditioning import (
    HierarchicalPopulationEmbedding,
    UnifiedFiLMGenerator,
)
from .modules.cnn import CNNStemEncoder, CNNDecoder
from .modules.dit import PatchEmbed1D, UnPatchify1D, DiTCore


class HybridCNNDiTFiLM(nn.Module):
    """
    CNN-DiT Hybrid with Unified FiLM Conditioning.

    Domain mapping:
    - CNN Encoder: local LD + haplotype patterns (FiLM: population modulation)
    - DiT Core: long-range inter-gene interactions (AdaLN-Zero: population modulation)
    - CNN Decoder: genotype reconstruction (FiLM: population fine-tuning)

    The model supports Classifier-Free Guidance by accepting a null population
    label (n_pops, i.e. 26) during training with a configurable dropout rate.
    """

    def __init__(self, config: dict):
        super().__init__()

        # --- Extract and validate config ---
        model_cfg = config.get("model", config)
        data_cfg = config.get("data", {})

        self.in_channels = data_cfg.get(
            "num_channels", model_cfg.get("in_channels", 8)
        )
        self.gene_size = data_cfg.get("gene_size", model_cfg.get("gene_size", 26624))
        base_channels = model_cfg.get("base_channels", 64)
        channel_mult = tuple(model_cfg.get("channel_mult", (1, 1, 2, 4)))
        kernel_size = model_cfg.get("kernel_size", 3)
        d_model = model_cfg.get("d_model", 256)
        n_dit_blocks = model_cfg.get("n_dit_blocks", 4)
        n_heads = model_cfg.get("n_heads", 4)
        mlp_ratio = model_cfg.get("mlp_ratio", 4.0)
        dropout = model_cfg.get("dropout", 0.0)
        patch_size = model_cfg.get("patch_size", 16)
        n_pops = model_cfg.get("n_pops", 26)
        n_superpops = model_cfg.get("n_superpops", 5)

        pop_to_superpop = model_cfg.get("pop_to_superpop", None)
        if pop_to_superpop is None:
            raise ValueError(
                "config must include 'pop_to_superpop' mapping "
                "(loaded from label_hierarchy.pkl)"
            )

        # CNN channel progression
        cnn_channels = [base_channels * m for m in channel_mult]
        self.latent_channels = cnn_channels[-1]  # 4C

        # Compute latent spatial size after encoder downsampling.
        # Encoder has len(channel_mult)-1 stride-2 downsamples.
        n_downsamples = len(channel_mult) - 1
        self.latent_size = self.gene_size // (2 ** n_downsamples)
        assert self.latent_size * (2 ** n_downsamples) == self.gene_size, (
            f"gene_size {self.gene_size} not evenly divisible by "
            f"2^{n_downsamples}={2 ** n_downsamples}"
        )

        # --- Build submodules ---

        # Population embedding (hierarchical: pop + superpop)
        self.pop_embedding = HierarchicalPopulationEmbedding(
            n_pops=n_pops,
            n_superpops=n_superpops,
            d_model=d_model,
            pop_to_superpop=pop_to_superpop,
        )

        # CNN encoder
        self.encoder = CNNStemEncoder(
            in_channels=self.in_channels,
            base_channels=base_channels,
            channel_mult=channel_mult,
            kernel_size=kernel_size,
        )

        # CNN decoder (symmetric to encoder)
        self.decoder = CNNDecoder(
            out_channels=self.in_channels,
            base_channels=base_channels,
            channel_mult=channel_mult,
            kernel_size=kernel_size,
        )

        # Unified FiLM generator for all blocks.
        # Uses decoder's actual output channel sizes so FiLM (gamma, beta)
        # dimensions match the decoder block outputs.
        self.film_gen = UnifiedFiLMGenerator(
            d_model=d_model,
            d_time=d_model,
            cnn_channels=cnn_channels,
            cnn_dec_channels=self.decoder.block_out_channels,
            n_dit_blocks=n_dit_blocks,
            d_dit=d_model,
        )

        # Patchify: (B, 4C, latent_size) -> (B, n_tokens, d_model)
        self.patchify = PatchEmbed1D(
            seq_len=self.latent_size,
            in_channels=self.latent_channels,
            patch_size=patch_size,
            d_model=d_model,
        )

        # DiT core
        self.dit = DiTCore(
            n_blocks=n_dit_blocks,
            d_model=d_model,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # Un-patchify: (B, n_tokens, d_model) -> (B, 4C, latent_size)
        self.unpatchify = UnPatchify1D(
            n_tokens=self.patchify.n_tokens,
            out_channels=self.latent_channels,
            patch_size=patch_size,
            d_model=d_model,
        )

        # Optional zero mask for enforcing biological constraints
        self.register_buffer("zero_mask", None)

    def set_zero_mask(self, zero_mask: torch.Tensor) -> None:
        """
        Set the zero mask for enforcing biological constraints.

        Args:
            zero_mask: Boolean tensor of shape (K, gene_size) or (gene_size,).
                       True indicates positions that must always be zero.
        """
        self.zero_mask = zero_mask.bool()

    def enforce_zeros(self, x: torch.Tensor) -> torch.Tensor:
        """Apply zero mask: positions where zero_mask is True are set to 0."""
        if self.zero_mask is not None:
            # Broadcast: zero_mask may be (K, gene_size) or (gene_size,)
            x = x * (~self.zero_mask).to(x.dtype)
        return x

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: predict noise epsilon given noisy input, timestep, and
        population label.

        Args:
            x: (B, K, gene_size) noisy Gene PCA tensor.
            t: (B,) integer diffusion timesteps.
            y: (B,) population label indices [0, n_pops-1].
                Use n_pops (26) as null class for CFG unconditional pass.

        Returns:
            (B, K, gene_size) predicted noise (or v-prediction target).
        """
        assert x.dim() == 3, f"Expected 3D input, got {x.dim()}D: {x.shape}"
        assert x.shape[1] == self.in_channels, (
            f"Channel mismatch: expected {self.in_channels}, got {x.shape[1]}"
        )
        assert x.shape[2] == self.gene_size, (
            f"Gene size mismatch: expected {self.gene_size}, got {x.shape[2]}"
        )

        # --- Conditioning ---
        pop_emb = self.pop_embedding(y)
        cnn_enc_params, cnn_dec_params, dit_params = self.film_gen(pop_emb, t)

        # --- CNN Encoder ---
        features, skips = self.encoder(x, cnn_enc_params)

        # --- Patchify -> DiT -> Un-patchify ---
        tokens = self.patchify(features)
        tokens = self.dit(tokens, dit_params)
        features_out = self.unpatchify(tokens)

        # --- CNN Decoder with skip connections ---
        output = self.decoder(features_out, skips, cnn_dec_params)

        # Ensure output matches input spatial size
        if output.shape[-1] != self.gene_size:
            output = output[..., : self.gene_size]

        # Enforce zero constraints
        output = self.enforce_zeros(output)

        return output
