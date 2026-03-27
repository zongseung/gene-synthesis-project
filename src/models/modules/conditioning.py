"""
Hierarchical population embedding and unified FiLM parameter generator.

Conditioning path:
    pop_label(0-25)
    -> HierarchicalPopulationEmbedding (pop_emb + superpop_emb -> MLP -> cond)
    -> + timestep_emb
    -> UnifiedFiLMGenerator
    -> per-block (gamma, beta) for CNN + (gamma, beta, alpha) for DiT
"""

import torch
import torch.nn as nn

from .base import timestep_embedding, zero_module


class HierarchicalPopulationEmbedding(nn.Module):
    """
    Hierarchical embedding: 26 populations + 5 superpopulations.

    Domain rationale:
    - Superpopulation embeddings (e.g. AFR with 661 samples) provide a stable
      learning foundation.
    - Fine-grained population embeddings (e.g. ASW with 61 samples) learn only
      the residual differences on top.
    - The pop_to_superpop mapping is loaded from label_hierarchy.pkl produced
      during preprocessing.
    """

    def __init__(
        self,
        n_pops: int = 26,
        n_superpops: int = 5,
        d_model: int = 256,
        pop_to_superpop: dict = None,
    ):
        super().__init__()
        if pop_to_superpop is None:
            raise ValueError("pop_to_superpop mapping is required")

        mapping = torch.zeros(n_pops, dtype=torch.long)
        for pop_idx, superpop_idx in pop_to_superpop.items():
            mapping[int(pop_idx)] = int(superpop_idx)
        self.register_buffer("pop_to_superpop_map", mapping)

        self.pop_emb = nn.Embedding(n_pops, d_model)
        self.superpop_emb = nn.Embedding(n_superpops, d_model)

        # Fuse concatenated pop + superpop embeddings into a single vector.
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, pop_label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pop_label: (B,) population indices in [0, 25].

        Returns:
            (B, d_model) hierarchical population embedding.
        """
        assert pop_label.dim() == 1, f"Expected 1D labels, got {pop_label.shape}"
        p_emb = self.pop_emb(pop_label)
        sp_label = self.pop_to_superpop_map[pop_label]
        sp_emb = self.superpop_emb(sp_label)
        return self.fusion(torch.cat([p_emb, sp_emb], dim=-1))


class UnifiedFiLMGenerator(nn.Module):
    """
    Generate all FiLM parameters for CNN and DiT blocks from a unified
    conditioning signal (population embedding + timestep embedding).

    Outputs per forward call:
    - CNN encoder blocks: list of (gamma, beta), each (B, channels).
    - CNN decoder blocks: list of (gamma, beta), each (B, channels).
    - DiT blocks: list of (B, d_dit*6) tensors containing
      (gamma1, beta1, alpha1, gamma2, beta2, alpha2) for AdaLN-Zero.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_time: int = 256,
        cnn_channels: list = None,
        cnn_dec_channels: list = None,
        n_dit_blocks: int = 4,
        d_dit: int = 256,
    ):
        super().__init__()
        if cnn_channels is None:
            raise ValueError("cnn_channels list is required")

        self.d_time = d_time

        # Timestep MLP: sinusoidal embedding -> d_model
        self.time_mlp = nn.Sequential(
            nn.Linear(d_time, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Fuse pop_emb + time_emb into unified conditioning vector.
        self.cond_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # CNN encoder FiLM layers: each outputs (gamma, beta) for its block.
        self.cnn_enc_films = nn.ModuleList(
            [nn.Linear(d_model, ch * 2) for ch in cnn_channels]
        )

        # CNN decoder FiLM layers: use actual decoder block output channels.
        # If not provided, fall back to reversed encoder channels.
        if cnn_dec_channels is None:
            cnn_dec_channels = list(reversed(cnn_channels))
        self.cnn_dec_films = nn.ModuleList(
            [nn.Linear(d_model, ch * 2) for ch in cnn_dec_channels]
        )

        # DiT AdaLN-Zero layers: each outputs 6 * d_dit values
        # (gamma1, beta1, alpha1 for attention + gamma2, beta2, alpha2 for FFN).
        # Initialized to zero so DiT starts as identity.
        self.dit_films = nn.ModuleList(
            [
                nn.Sequential(
                    nn.SiLU(),
                    zero_module(nn.Linear(d_model, d_dit * 6)),
                )
                for _ in range(n_dit_blocks)
            ]
        )

    def forward(
        self, pop_emb: torch.Tensor, t: torch.Tensor
    ) -> tuple[list, list, list]:
        """
        Args:
            pop_emb: (B, d_model) from HierarchicalPopulationEmbedding.
            t: (B,) integer diffusion timesteps.

        Returns:
            cnn_enc_params: list of (gamma, beta) per encoder block.
            cnn_dec_params: list of (gamma, beta) per decoder block.
            dit_params: list of (B, d_dit*6) per DiT block.
        """
        t_emb = self.time_mlp(timestep_embedding(t, self.d_time))
        cond = self.cond_mlp(torch.cat([pop_emb, t_emb], dim=-1))

        cnn_enc_params = []
        for layer in self.cnn_enc_films:
            gb = layer(cond)
            gamma, beta = gb.chunk(2, dim=-1)
            cnn_enc_params.append((gamma, beta))

        cnn_dec_params = []
        for layer in self.cnn_dec_films:
            gb = layer(cond)
            gamma, beta = gb.chunk(2, dim=-1)
            cnn_dec_params.append((gamma, beta))

        dit_params = [layer(cond) for layer in self.dit_films]

        return cnn_enc_params, cnn_dec_params, dit_params
