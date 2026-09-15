from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.special import expit


def evaluate_genotypes(
    prepared_dir: Path, synthetic_factors: np.ndarray, real_factors: np.ndarray, *, seed: int,
) -> tuple[dict, np.ndarray]:
    with (prepared_dir / "glm_pca_parameters.pkl").open("rb") as handle:
        parameters = pickle.load(handle)
    if any(item.get("family") != "binomial" or item.get("trials") != 2 for item in parameters):
        raise ValueError("Genotype evaluation requires a Binomial(2,p) decoder")
    with np.load(prepared_dir / "genotypes.npz") as data:
        calls, offsets = data["calls"], data["offsets"]
    with np.load(prepared_dir / "dataset.npz") as data:
        indices = data["val_indices"][:len(synthetic_factors)]
    real = calls[indices]
    def decode(factors: np.ndarray, draw_seed: int) -> np.ndarray:
        probabilities = np.concatenate([
            expit(factors[:, gene] @ item["loadings"].T + item["intercept"])
            for gene, item in enumerate(parameters)
        ], axis=1)
        return np.random.default_rng(draw_seed).binomial(2, probabilities).astype(np.int8)

    synthetic = decode(synthetic_factors, seed)
    reconstruction = decode(real_factors, seed + 1)

    def metrics(generated: np.ndarray) -> dict:
        valid = np.isfinite(real)
        counts = valid.sum(axis=0)
        usable = counts > 0
        real_af = np.nansum(real, axis=0) / (2 * np.maximum(counts, 1))
        generated_af = generated.mean(axis=0) / 2
        errors = []
        for start, end in zip(offsets[:-1], offsets[1:]):
            for distance in (1, 2, 4):
                for left in range(int(start), int(end) - distance):
                    right = left + distance
                    rows = valid[:, left] & valid[:, right]
                    if rows.sum() < 4:
                        continue
                    r = real[rows][:, [left, right]]
                    g = generated[rows][:, [left, right]]
                    errors.append(float(abs(np.cov(r, rowvar=False, ddof=0)[0, 1]
                                            - np.cov(g, rowvar=False, ddof=0)[0, 1])))
        return {
            "af_mae": float(np.abs(real_af[usable] - generated_af[usable]).mean()),
            "local_dosage_covariance_mae": float(np.mean(errors)) if errors else None,
            "local_pairs_evaluated": len(errors),
            "valid_genotype_fraction": float(np.isin(generated, [0, 1, 2]).mean()),
            "unique_individual_fraction": len(np.unique(generated, axis=0)) / len(generated),
        }

    return {
        "synthetic": metrics(synthetic),
        "real_latent_decoder_reference": metrics(reconstruction),
        "samples": len(synthetic),
        "variants": synthetic.shape[1],
        "limitation": "One stochastic independent-binomial decoder draw; unphased local dosage covariance, not haplotype LD",
    }, synthetic
