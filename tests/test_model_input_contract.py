import pytest
import torch

from src.models import HybridCNNDiTFiLM
from src.models.modules.dit import PatchEmbed1D


def _model(gene_size: int = 512) -> HybridCNNDiTFiLM:
    config = {
        "data": {"num_channels": 4, "gene_size": gene_size},
        "model": {"base_channels": 8, "channel_mult": [1, 1, 2, 2, 4], "d_model": 16,
                  "n_dit_blocks": 1, "n_heads": 1, "patch_size": 16,
                  "pop_to_superpop": {i: i % 5 for i in range(26)}},
    }
    return HybridCNNDiTFiLM(config)


def test_forward_rejects_a_channel_count_that_differs_from_config():
    model = _model()
    with pytest.raises(ValueError, match="channels"):
        model(torch.randn(2, 3, 512), torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long))


def test_forward_rejects_a_gene_size_that_differs_from_config():
    model = _model()
    with pytest.raises(ValueError, match="gene_size"):
        model(torch.randn(2, 4, 256), torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long))


def test_patch_embed_rejects_a_sequence_the_patch_size_does_not_divide():
    with pytest.raises(ValueError, match="patch_size"):
        PatchEmbed1D(seq_len=10, in_channels=1, patch_size=4, d_model=8)
