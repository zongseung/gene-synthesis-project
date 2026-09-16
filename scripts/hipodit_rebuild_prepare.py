# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# ─── How to run ───
# Imported by hipodit_rebuild_check.py; use that file's prepare command.

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import time
from collections import Counter
from pathlib import Path


class DataPreparationError(RuntimeError):
    pass


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _labels_and_split(panel_path: Path, seed: int):
    import numpy as np

    from src.preprocessing.labels import compute_split_indices

    with panel_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    samples = [row["sample"] for row in rows]
    population_values = [row["pop"] for row in rows]
    superpopulation_values = [row["super_pop"] for row in rows]
    populations = sorted(set(population_values))
    pop_to_idx = {pop: index for index, pop in enumerate(populations)}
    labels = np.asarray([pop_to_idx[pop] for pop in population_values], dtype=np.int64)
    split = compute_split_indices(labels, val_ratio=0.1, test_ratio=0.1, seed=seed)
    superpops = sorted(set(superpopulation_values))
    super_to_idx = {pop: index for index, pop in enumerate(superpops)}
    mapping = {
        pop_to_idx[pop]: super_to_idx[superpop]
        for pop, superpop in zip(population_values, superpopulation_values)
    }
    hierarchy = {
        "pop_to_idx": pop_to_idx,
        "idx_to_pop": {index: pop for pop, index in pop_to_idx.items()},
        "superpop_to_idx": super_to_idx,
        "idx_to_superpop": {index: pop for pop, index in super_to_idx.items()},
        "pop_sizes": dict(Counter(population_values)),
        "pop_to_superpop": mapping,
    }
    return samples, labels, split, mapping, hierarchy


def _gene_matrix(vcf, gene: dict, train_indices, max_variants: int, owned_sites: set):
    import numpy as np

    columns, variants = [], []
    for variant in vcf(f"17:{gene['start'] + 1}-{gene['end']}"):
        if len(variant.ALT) != 1 or len(variant.REF) != 1 or len(variant.ALT[0]) != 1:
            continue
        if (variant.POS, variant.REF, variant.ALT[0]) in owned_sites:
            continue
        types = variant.gt_types
        dosage = types.astype(np.float32)
        dosage[types == 3], dosage[types == 2] = 2.0, np.nan
        train = dosage[train_indices]
        if not np.isfinite(train).any():
            continue
        train_mean = float(np.nanmean(train))
        maf = min(train_mean / 2.0, 1.0 - train_mean / 2.0)
        if maf < 0.01:
            continue
        columns.append(dosage)
        variants.append(
            {
                "id": variant.ID or f"17:{variant.POS}:{variant.REF}:{variant.ALT[0]}",
                "position": int(variant.POS),
                "ref": variant.REF,
                "alt": variant.ALT[0],
                "train_maf": maf,
            }
        )
        if len(columns) == max_variants:
            break
    return (np.stack(columns, axis=1), variants) if columns else (None, [])


def prepare(args: argparse.Namespace) -> None:
    import numpy as np
    from cyvcf2 import VCF

    from src.preprocessing.config import PANEL_PATH, REFGENE_PATH, VCF_PATH
    from src.preprocessing.gene_annotation import load_refgene
    from src.preprocessing.binomial_glm_pca import fit_binomial_glm_pca, information_schedule
    from src.preprocessing.tokenizer import (
        apply_normalization,
        fit_normalization_stats,
        invert_normalization,
    )
    from hipodit_genotype_check import provenance

    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sample_ids, labels, split, pop_mapping, hierarchy = _labels_and_split(
        Path(PANEL_PATH), args.seed
    )
    train_indices, val_indices, test_indices = split
    vcf = VCF(VCF_PATH, strict_gt=True)
    if sample_ids != list(vcf.samples):
        raise DataPreparationError("VCF and panel sample order differs")
    features, parameters, gene_map, reconstruction, genotypes, fitted = [], [], [], [], [], []
    owned_sites = set()
    try:
        for gene in load_refgene(REFGENE_PATH)["17"]:
            matrix, variants = _gene_matrix(vcf, gene, train_indices, args.max_variants, owned_sites)
            if matrix is None or matrix.shape[1] <= args.components:
                continue
            result = fit_binomial_glm_pca(matrix, train_indices, args.components,
                                          max_iter=args.glm_iterations)
            transformed = result.factors.astype(np.float32)
            loadings, intercept = result.loadings, result.intercept
            dosage_mean = 2 * result.probabilities(result.factors[val_indices])
            reconstruction.append(
                {
                    "gene": gene["name"],
                    "validation_dosage_mean_rmse": float(
                        np.sqrt(np.nanmean((dosage_mean - matrix[val_indices]) ** 2))
                    ),
                    "validation_dosage_mean_max": float(dosage_mean.max()),
                    "optimizer_converged": result.converged,
                    "objective_initial": result.objective_initial,
                    "objective_final": result.objective_final,
                }
            )
            features.append(transformed)
            genotypes.append(matrix)
            fitted.append(result)
            owned_sites.update((item["position"], item["ref"], item["alt"]) for item in variants)
            parameters.append(
                {"gene": gene["name"], "loadings": loadings, "intercept": intercept,
                 "family": "binomial", "trials": 2, "link": "logit"}
            )
            gene_map.append(
                {
                    "gene": gene["name"],
                    "start": gene["start"],
                    "end": gene["end"],
                    "variants": variants,
                }
            )
            if len(features) == args.genes:
                break
    finally:
        vcf.close()
    if len(features) != args.genes:
        raise DataPreparationError(f"only {len(features)} eligible chr17 genes found")
    raw = np.stack(features, axis=1)
    stats = fit_normalization_stats(raw[train_indices], clip=None)
    information = np.stack([fit.fisher_diagonal(fit.factors[train_indices], stats["std"][index])
                            for index, fit in enumerate(fitted)])
    np.save(args.output_dir / "fisher_information.npy", information)
    np.save(args.output_dir / "feature_schedule.npy", information_schedule(information))
    np.savez_compressed(args.output_dir / "genotypes.npz", calls=np.concatenate(genotypes, axis=1),
                        offsets=np.cumsum([0] + [matrix.shape[1] for matrix in genotypes]))
    values_digest = hashlib.sha256(
        stats["mean"].tobytes() + stats["std"].tobytes()
    ).hexdigest()
    normalized = apply_normalization(raw, stats)
    roundtrip = float(np.max(np.abs(invert_normalization(normalized, stats) - raw)))
    np.savez_compressed(
        args.output_dir / "dataset.npz",
        x=normalized,
        y=labels,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )
    for name, value in (
        ("normalization_stats.pkl", stats),
        ("glm_pca_parameters.pkl", parameters),
    ):
        with (args.output_dir / name).open("wb") as handle:
            pickle.dump(value, handle, protocol=4)
    for name, indices in (
        ("train", train_indices),
        ("val", val_indices),
        ("test", test_indices),
    ):
        with (args.output_dir / f"{name}_data.pkl").open("wb") as handle:
            pickle.dump((normalized[indices], labels[indices]), handle, protocol=4)
    stats_digest = hashlib.sha256(
        (args.output_dir / "normalization_stats.pkl").read_bytes()
    ).hexdigest()
    with (args.output_dir / "label_hierarchy.pkl").open("wb") as handle:
        pickle.dump(hierarchy, handle, protocol=4)
    _atomic_json(
        args.output_dir / "gene_variant_map.json", {"chromosome": 17, "genes": gene_map}
    )
    _atomic_json(
        args.output_dir / "preprocessing_metadata.json",
        {
            "dim_reduction_method": "glm_pca",
            "family": "binomial",
            "trials": 2,
            "fit_split": "train",
            "chromosome": 17,
            "gene_count": args.genes,
            "normalization_fingerprint": stats_digest,
            "normalization_values_sha256": values_digest,
        },
    )
    split_manifest = {
        "seed": args.seed,
        "method": "population-stratified 80/10/10",
        "train_indices": train_indices.tolist(),
        "validation_indices": val_indices.tolist(),
        "test_indices": test_indices.tolist(),
        "train_sample_ids": [sample_ids[index] for index in train_indices],
        "validation_sample_ids": [sample_ids[index] for index in val_indices],
        "test_sample_ids": [sample_ids[index] for index in test_indices],
    }
    _atomic_json(args.output_dir / "split_manifest.json", split_manifest)
    report = {
        "status": "prepared",
        "seed": args.seed,
        "source_vcf": str(VCF_PATH),
        "source_vcf_bytes": Path(VCF_PATH).stat().st_size,
        "source_panel": str(PANEL_PATH),
        "chromosome": 17,
        "samples": len(raw),
        "genes": args.genes,
        "components": args.components,
        # The per-gene variant cap actually used; without it the panel cannot be rebuilt.
        "max_variants": args.max_variants,
        "glm_family": "binomial",
        "glm_iterations": args.glm_iterations,
        "maf_filter": "train-only >= 0.01",
        "missing_imputation": "none; missing calls excluded from likelihood",
        "fit_split": "train",
        "split_sizes": {
            "train": len(train_indices),
            "validation": len(val_indices),
            "test": len(test_indices),
        },
        "pop_to_superpop": pop_mapping,
        "normalization_fingerprint": stats_digest,
        "normalization_values_sha256": values_digest,
        "normalization_roundtrip_max_abs": roundtrip,
        "reconstruction": reconstruction,
        "runtime_seconds": time.monotonic() - started, **provenance(),
    }
    _atomic_json(args.output_dir / "prepare_report.json", report)
    print(json.dumps(report, indent=2))
