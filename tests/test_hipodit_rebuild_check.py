from __future__ import annotations

import json
import pickle
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from scripts.hipodit_rebuild_prepare import _gene_matrix
from src.data.dataloader import create_dataloaders
from src.training.trainer import _bind_normalization_stats

SCRIPT = Path(__file__).parents[1] / "scripts" / "hipodit_rebuild_check.py"


def test_overlapping_genes_do_not_duplicate_owned_variant_columns() -> None:

    variants = [SimpleNamespace(ALT=["C"], REF="A", ID=f"v{position}", POS=position,
                                gt_types=np.array([0, 1, 3, 0])) for position in (10, 20)]
    matrix, metadata = _gene_matrix(lambda region: variants, {"start": 0, "end": 30},
                                    np.arange(4), 8, {(10, "A", "C")})
    assert matrix.shape == (4, 1)
    assert [item["position"] for item in metadata] == [20]


def test_cli_help_describes_the_diagnostic_stages() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "{prepare,train}" in result.stdout


def test_train_help_offers_the_decoder_and_the_evaluation_split() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "train", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--decoder-dir" in result.stdout
    assert "--eval-split {val,test}" in result.stdout


def test_cli_rejects_zero_genes() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "prepare", "--genes", "0"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "positive integer" in result.stderr


def test_cli_rejects_gene_count_incompatible_with_model_shape() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "prepare",
            "--genes",
            "33",
            "--output-dir",
            "unused",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "divisible by 8" in result.stderr


def test_dataloader_uses_validated_processed_directory(tmp_path: Path) -> None:

    for split in ("train", "val"):
        with (tmp_path / f"{split}_data.pkl").open("wb") as handle:
            pickle.dump((np.zeros((4, 8, 2), dtype=np.float32), np.arange(4)), handle)
    (tmp_path / "preprocessing_metadata.json").write_text(
        json.dumps({"dim_reduction_method": "glm_pca"})
    )
    config = {
        "data": {
            "processed_dir": str(tmp_path),
            "dim_reduction_method": "glm_pca",
            "gene_size": 8,
            "num_channels": 2,
        },
        "training": {"batch_size": 2, "num_workers": 0},
    }

    train, validation = create_dataloaders(config)

    assert len(train.dataset) == 4
    assert len(validation.dataset) == 4

    config["data"]["gene_size"] = 24576
    try:
        create_dataloaders(config)
    except ValueError as error:
        assert "gene_size" in str(error)
    else:
        raise AssertionError("a gene_size mismatch with the processed tensors must be rejected")


def test_dataloader_rejects_unproven_preprocessing_method(tmp_path: Path) -> None:

    config = {
        "data": {"processed_dir": str(tmp_path), "dim_reduction_method": "glm_pca"},
        "training": {"batch_size": 2, "num_workers": 0},
    }

    try:
        create_dataloaders(config)
    except ValueError as error:
        assert "preprocessing metadata" in str(error)
    else:
        raise AssertionError("missing preprocessing provenance was accepted")


def test_training_config_binds_normalization_file(tmp_path: Path) -> None:

    stats_path = tmp_path / "normalization_stats.pkl"
    stats_path.write_bytes(b"stable training statistics")
    config = {"data": {"normalize": True, "normalization_stats_path": str(stats_path)}}

    _bind_normalization_stats(config)

    assert config["data"]["normalization_stats_sha256"] == (
        "88010e540524159630dc5e3776bafffe5f53acd3197a70f89f79fc614b777ac4"
    )
