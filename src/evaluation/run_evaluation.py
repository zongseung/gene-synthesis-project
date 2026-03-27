#!/usr/bin/env python3
"""Full evaluation pipeline for synthetic genotype quality assessment.

Runs all evaluation metrics in parallel using ProcessPoolExecutor:
- Fidelity: PCA channel-wise Wasserstein, AF correlation
- Structure: PCA overlap (Silhouette), Sliced Wasserstein Distance
- Utility: Recovery Rate, Augmentation effect
- Privacy: NNAA, DUPI, Membership inference AUC
- Robustness: Pop-size vs quality correlation, Per-pop DUPI gap

Saves results to JSON and CSV, prints summary table.

Usage:
    python src/evaluation/run_evaluation.py \\
        --config configs/default.yaml \\
        --syn_dir outputs/run_001/synthetic_samples \\
        --output_dir outputs/run_001/evaluation
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.evaluation.metrics import (
    af_correlation,
    augmentation_effect,
    dupi,
    membership_inference_auc,
    nnaa,
    pca_overlap_silhouette,
    recovery_rate,
    robustness_correlation,
    sliced_wasserstein,
    wasserstein_per_channel,
)
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────
# Data loading
# ───────────────────────────────────────────────────────────────────

def load_real_data(
    test_path: str,
    train_path: str | None = None,
) -> dict[str, np.ndarray]:
    """Load real test (and optionally train) data from pickle files.

    Args:
        test_path: Path to test_data.pkl containing (x_test, y_test).
        train_path: Optional path to train_data.pkl.

    Returns:
        Dict with 'test_x', 'test_y', and optionally 'train_x', 'train_y'.
    """
    with open(test_path, "rb") as f:
        x_test, y_test = pickle.load(f)

    result = {
        "test_x": np.asarray(x_test, dtype=np.float32),
        "test_y": np.asarray(y_test, dtype=np.int64),
    }

    if train_path and Path(train_path).exists():
        with open(train_path, "rb") as f:
            x_train, y_train = pickle.load(f)
        result["train_x"] = np.asarray(x_train, dtype=np.float32)
        result["train_y"] = np.asarray(y_train, dtype=np.int64)

    return result


def load_synthetic_data(syn_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load all synthetic samples from a directory of .pt files.

    Each .pt file contains (genome_tensor, label_tensor).

    Args:
        syn_dir: Directory containing sample_pop*.pt files.

    Returns:
        (syn_x, syn_y): Stacked arrays of synthetic samples and labels.
    """
    syn_path = Path(syn_dir)
    pt_files = sorted(syn_path.glob("sample_pop*.pt"))

    if not pt_files:
        raise FileNotFoundError(f"No sample_pop*.pt files found in {syn_dir}")

    samples = []
    labels = []

    for pt_file in pt_files:
        genome, label = torch.load(pt_file, map_location="cpu", weights_only=True)
        samples.append(genome.numpy())
        labels.append(label.item() if label.dim() == 0 else label.numpy())

    syn_x = np.stack(samples).astype(np.float32)  # (M, K, gene_size)
    syn_y = np.array(labels, dtype=np.int64)  # (M,)

    logger.info(f"Loaded {len(syn_x)} synthetic samples from {syn_dir}")
    return syn_x, syn_y


# ───────────────────────────────────────────────────────────────────
# Metric wrappers (for parallel execution)
# ───────────────────────────────────────────────────────────────────

def _eval_fidelity(real_x: np.ndarray, syn_x: np.ndarray) -> dict:
    """Compute fidelity metrics."""
    wass = wasserstein_per_channel(real_x, syn_x)
    af = af_correlation(real_x, syn_x)
    return {
        "wasserstein_per_channel": wass,
        "af_correlation": af,
    }


def _eval_structure(real_x: np.ndarray, syn_x: np.ndarray) -> dict:
    """Compute structure metrics."""
    pca_sil = pca_overlap_silhouette(real_x, syn_x)
    swd = sliced_wasserstein(real_x, syn_x)
    return {
        "pca_overlap_silhouette": pca_sil,
        "sliced_wasserstein": swd,
    }


def _eval_utility(
    real_train_x: np.ndarray | None,
    real_train_y: np.ndarray | None,
    syn_x: np.ndarray,
    syn_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> dict:
    """Compute utility metrics."""
    rr = recovery_rate(
        syn_x, syn_y, test_x, test_y,
        real_train_x=real_train_x,
        real_train_y=real_train_y,
    )
    result = {"recovery_rate": rr}

    if real_train_x is not None:
        aug = augmentation_effect(
            real_train_x, real_train_y,
            syn_x, syn_y,
            test_x, test_y,
            ratios=[0.05, 0.5],
        )
        result["augmentation_effect"] = aug

    return result


def _eval_privacy(
    real_train_x: np.ndarray | None,
    real_test_x: np.ndarray,
    syn_x: np.ndarray,
) -> dict:
    """Compute privacy metrics."""
    result = {}

    if real_train_x is not None:
        nnaa_result = nnaa(real_train_x, real_test_x, syn_x)
        result["nnaa"] = nnaa_result

    dupi_result = dupi(real_test_x, syn_x, k=1)
    result["dupi"] = dupi_result

    if real_train_x is not None:
        mi_result = membership_inference_auc(real_train_x, syn_x, real_test_x)
        result["membership_inference"] = mi_result

    return result


def _eval_robustness(
    real_x: np.ndarray,
    real_y: np.ndarray,
    syn_x: np.ndarray,
    syn_y: np.ndarray,
) -> dict:
    """Compute per-population quality and robustness metrics."""
    per_pop_quality = {}

    for pop_idx in np.unique(real_y):
        real_mask = real_y == pop_idx
        syn_mask = syn_y == pop_idx

        if real_mask.sum() == 0 or syn_mask.sum() == 0:
            continue

        real_pop = real_x[real_mask]
        syn_pop = syn_x[syn_mask]

        # Per-population AF correlation
        real_mean = real_pop.mean(axis=0).flatten()
        syn_mean = syn_pop.mean(axis=0).flatten()
        nonzero = (real_mean != 0) | (syn_mean != 0)

        if nonzero.sum() >= 2:
            r = float(np.corrcoef(real_mean[nonzero], syn_mean[nonzero])[0, 1])
        else:
            r = 0.0

        per_pop_quality[int(pop_idx)] = {
            "n_real": int(real_mask.sum()),
            "n_syn": int(syn_mask.sum()),
            "af_correlation": r,
        }

    # Size vs quality correlation
    pop_sizes = {p: v["n_real"] for p, v in per_pop_quality.items()}
    rob_corr = robustness_correlation(per_pop_quality, pop_sizes)

    # Per-population DUPI
    per_pop_dupi = {}
    for pop_idx in per_pop_quality:
        real_mask = real_y == pop_idx
        syn_mask = syn_y == pop_idx
        n_real = int(real_mask.sum())
        n_syn = int(syn_mask.sum())

        if n_real >= 5 and n_syn >= 5:
            pop_dupi = dupi(real_x[real_mask], syn_x[syn_mask], k=1)
            per_pop_dupi[int(pop_idx)] = {
                "dupi": pop_dupi["dupi"],
                "benchmark": pop_dupi["benchmark"],
                "gap": pop_dupi["gap"],
            }

    # DUPI gap analysis
    if len(per_pop_dupi) >= 5:
        gaps = [v["gap"] for v in per_pop_dupi.values()]
        sizes = [per_pop_quality[p]["n_real"] for p in per_pop_dupi.keys()]
        dupi_gap_corr = float(np.corrcoef(sizes, gaps)[0, 1])
    else:
        dupi_gap_corr = float("nan")

    return {
        "per_pop_quality": per_pop_quality,
        "size_quality_correlation": rob_corr,
        "per_pop_dupi": per_pop_dupi,
        "dupi_gap_correlation": dupi_gap_corr,
    }


# ───────────────────────────────────────────────────────────────────
# Main evaluation
# ───────────────────────────────────────────────────────────────────

def evaluate_all(
    real_data: dict[str, np.ndarray],
    syn_x: np.ndarray,
    syn_y: np.ndarray,
    max_workers: int = 6,
) -> dict[str, Any]:
    """Run all evaluation metrics in parallel.

    Args:
        real_data: Dict with test_x, test_y, and optionally train_x, train_y.
        syn_x: (M, K, gene_size) synthetic data.
        syn_y: (M,) synthetic labels.
        max_workers: Number of parallel workers.

    Returns:
        Nested dict of all evaluation results.
    """
    test_x = real_data["test_x"]
    test_y = real_data["test_y"]
    train_x = real_data.get("train_x")
    train_y = real_data.get("train_y")

    # Transpose if stored as (N, gene_size, K) -> need (N, K, gene_size)
    if test_x.ndim == 3 and test_x.shape[2] < test_x.shape[1]:
        test_x = np.transpose(test_x, (0, 2, 1))
    if train_x is not None and train_x.ndim == 3 and train_x.shape[2] < train_x.shape[1]:
        train_x = np.transpose(train_x, (0, 2, 1))
    if syn_x.ndim == 3 and syn_x.shape[2] < syn_x.shape[1]:
        syn_x = np.transpose(syn_x, (0, 2, 1))

    logger.info(
        f"Evaluation: real_test={test_x.shape}, syn={syn_x.shape}, "
        f"real_train={'None' if train_x is None else str(train_x.shape)}"
    )

    results = {}
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            "fidelity": executor.submit(_eval_fidelity, test_x, syn_x),
            "structure": executor.submit(_eval_structure, test_x, syn_x),
            "utility": executor.submit(
                _eval_utility, train_x, train_y, syn_x, syn_y, test_x, test_y
            ),
            "privacy": executor.submit(_eval_privacy, train_x, test_x, syn_x),
            "robustness": executor.submit(
                _eval_robustness, test_x, test_y, syn_x, syn_y
            ),
        }

        for name, future in futures.items():
            try:
                results[name] = future.result()
                logger.info(f"  {name}: done")
            except Exception as e:
                logger.error(f"  {name}: FAILED - {e}")
                results[name] = {"error": str(e)}

    elapsed = time.time() - t_start
    results["evaluation_time_sec"] = round(elapsed, 2)

    return results


# ───────────────────────────────────────────────────────────────────
# Output formatting
# ───────────────────────────────────────────────────────────────────

def _make_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to Python native for JSON."""
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return str(obj)
    return obj


def save_results(
    results: dict[str, Any],
    output_dir: str,
) -> None:
    """Save evaluation results to JSON and CSV.

    Args:
        results: Nested results dict from evaluate_all.
        output_dir: Directory for output files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save full results as JSON
    json_path = os.path.join(output_dir, "summary_metrics.json")
    serializable = _make_json_serializable(results)
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Full results saved to {json_path}")

    # Save per-population metrics as CSV
    robustness = results.get("robustness", {})
    per_pop = robustness.get("per_pop_quality", {})
    per_pop_dupi = robustness.get("per_pop_dupi", {})

    if per_pop:
        csv_path = os.path.join(output_dir, "per_population_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "pop_idx", "n_real", "n_syn", "af_correlation",
                "dupi", "dupi_benchmark", "dupi_gap",
            ])
            for pop_idx in sorted(per_pop.keys()):
                info = per_pop[pop_idx]
                dupi_info = per_pop_dupi.get(pop_idx, {})
                writer.writerow([
                    pop_idx,
                    info["n_real"],
                    info["n_syn"],
                    f"{info['af_correlation']:.6f}",
                    f"{dupi_info.get('dupi', ''):.6f}" if "dupi" in dupi_info else "",
                    f"{dupi_info.get('benchmark', ''):.6f}" if "benchmark" in dupi_info else "",
                    f"{dupi_info.get('gap', ''):.6f}" if "gap" in dupi_info else "",
                ])
        logger.info(f"Per-population CSV saved to {csv_path}")


def print_summary(results: dict[str, Any]) -> None:
    """Print a human-readable summary table of key metrics."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    # Fidelity
    fidelity = results.get("fidelity", {})
    af = fidelity.get("af_correlation", {})
    wass = fidelity.get("wasserstein_per_channel", {})
    print(f"\nFidelity:")
    print(f"  AF Correlation (Pearson r):       {af.get('pearson_r', 'N/A'):.4f}" if isinstance(af.get('pearson_r'), (int, float)) else f"  AF Correlation: N/A")
    print(f"  Wasserstein (mean per-channel):   {wass.get('mean', 'N/A'):.4f}" if isinstance(wass.get('mean'), (int, float)) else f"  Wasserstein: N/A")

    # Structure
    structure = results.get("structure", {})
    sil = structure.get("pca_overlap_silhouette", {})
    swd = structure.get("sliced_wasserstein", {})
    print(f"\nStructure:")
    print(f"  Silhouette Score:                 {sil.get('silhouette_score', 'N/A'):.4f}" if isinstance(sil.get('silhouette_score'), (int, float)) else f"  Silhouette: N/A")
    print(f"  Sliced Wasserstein Distance:      {swd.get('swd', 'N/A'):.4f}" if isinstance(swd.get('swd'), (int, float)) else f"  SWD: N/A")

    # Utility
    utility = results.get("utility", {})
    rr = utility.get("recovery_rate", {})
    print(f"\nUtility:")
    print(f"  Recovery Rate (syn acc):          {rr.get('syn_accuracy', 'N/A'):.4f}" if isinstance(rr.get('syn_accuracy'), (int, float)) else f"  Recovery Rate: N/A")
    if "ratio" in rr:
        print(f"  Recovery Ratio (syn/real):        {rr['ratio']:.4f}" if isinstance(rr['ratio'], (int, float)) else "")

    # Privacy
    privacy = results.get("privacy", {})
    nnaa_r = privacy.get("nnaa", {})
    dupi_r = privacy.get("dupi", {})
    mi_r = privacy.get("membership_inference", {})
    print(f"\nPrivacy:")
    if "nnaa" in nnaa_r:
        print(f"  NNAA:                             {nnaa_r['nnaa']:.4f} (ideal=0.5)")
    if "dupi" in dupi_r:
        print(f"  DUPI:                             {dupi_r['dupi']:.4f} (benchmark={dupi_r.get('benchmark', '?'):.4f})")
    if "auc" in mi_r:
        print(f"  Membership Inference AUC:         {mi_r['auc']:.4f} (ideal=0.5)")

    # Robustness
    robustness = results.get("robustness", {})
    corr = robustness.get("size_quality_correlation", {})
    print(f"\nRobustness:")
    if isinstance(corr.get("abs_correlation"), (int, float)):
        print(f"  |r| (size vs quality):            {corr['abs_correlation']:.4f} (ideal=0)")
    dupi_gap = robustness.get("dupi_gap_correlation", None)
    if dupi_gap is not None and not (isinstance(dupi_gap, float) and np.isnan(dupi_gap)):
        print(f"  DUPI gap correlation:             {dupi_gap:.4f}")

    elapsed = results.get("evaluation_time_sec", "?")
    print(f"\nEvaluation time: {elapsed}s")
    print("=" * 70 + "\n")


# ───────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic genotype quality"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--syn_dir", type=str, required=True,
        help="Directory containing synthetic sample .pt files",
    )
    parser.add_argument(
        "--real_test_path", type=str, default=None,
        help="Path to real test data pkl (default: data/processed/test_data.pkl)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--max_workers", type=int, default=6,
        help="Number of parallel workers for metric computation",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Resolve paths
    real_test_path = args.real_test_path or "data/processed/test_data.pkl"
    real_train_path = "data/processed/train_data.pkl"

    if args.output_dir is None:
        save_dir = config.get("save_dir", "outputs/default")
        args.output_dir = os.path.join(save_dir, "evaluation")

    # Load data
    logger.info("Loading real data...")
    real_data = load_real_data(real_test_path, real_train_path)

    logger.info("Loading synthetic data...")
    syn_x, syn_y = load_synthetic_data(args.syn_dir)

    # Run evaluation
    logger.info("Starting evaluation...")
    results = evaluate_all(real_data, syn_x, syn_y, max_workers=args.max_workers)

    # Save and print
    save_results(results, args.output_dir)
    print_summary(results)


if __name__ == "__main__":
    main()
