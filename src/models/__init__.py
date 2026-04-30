"""
HiPoDiT model package.

Main exports:
    - HybridCNNDiTFiLM: The main hybrid CNN-DiT model with FiLM conditioning.
    - GaussianDiffusion: Diffusion process (cosine schedule, DDIM, Min-SNR, CFG).
"""

from .hybrid_geno_dit import HybridCNNDiTFiLM
from .diffusion import GaussianDiffusion

__all__ = ["HybridCNNDiTFiLM", "GaussianDiffusion"]
