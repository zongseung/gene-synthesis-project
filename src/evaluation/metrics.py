"""Individual evaluation metric implementations for synthetic genotype quality.

All metrics operate on numpy arrays in PCA space. Inputs are typically:
- real: (N, K, gene_size) or (N, D) after flattening
- syn:  (M, K, gene_size) or (M, D) after flattening
- labels: (N,) or (M,) population indices

Metric categories:
- Fidelity: wasserstein_per_channel, af_correlation
- Structure: pca_overlap_silhouette, sliced_wasserstein
- Utility: recovery_rate, augmentation_effect
- Privacy: nnaa, dupi, membership_inference_auc
- Robustness: robustness_correlation
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import comb
from scipy.stats import wasserstein_distance

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────
# Fidelity
# ───────────────────────────────────────────────────────────────────

def wasserstein_per_channel(
    real: np.ndarray,
    syn: np.ndarray,
) -> dict[str, Any]:
    """Per-channel 1D Wasserstein distance between real and synthetic data.

    Args:
        real: (N, K, gene_size) real PCA features.
        syn: (M, K, gene_size) synthetic PCA features.

    Returns:
        Dict with per_channel distances and mean distance.
    """
    assert real.ndim == 3 and syn.ndim == 3, (
        f"Expected 3D arrays, got real={real.ndim}D, syn={syn.ndim}D"
    )
    n_channels = real.shape[1]
    distances = []

    for k in range(n_channels):
        # Flatten gene dimension for each channel
        real_flat = real[:, k, :].flatten()
        syn_flat = syn[:, k, :].flatten()
        dist = wasserstein_distance(real_flat, syn_flat)
        distances.append(float(dist))

    return {
        "per_channel": distances,
        "mean": float(np.mean(distances)),
        "std": float(np.std(distances)),
    }


def af_correlation(
    real: np.ndarray,
    syn: np.ndarray,
) -> dict[str, float]:
    """Gene-level mean correlation between real and synthetic data.

    Computes mean across samples for each (gene, channel) position,
    then measures Pearson correlation between real and synthetic means.

    Args:
        real: (N, K, gene_size) real PCA features.
        syn: (M, K, gene_size) synthetic PCA features.

    Returns:
        Dict with pearson_r and p_value.
    """
    real_mean = real.mean(axis=0).flatten()  # (K * gene_size,)
    syn_mean = syn.mean(axis=0).flatten()

    # Remove positions where both are zero (padding)
    nonzero = (real_mean != 0) | (syn_mean != 0)
    if nonzero.sum() < 2:
        return {"pearson_r": 0.0}

    r = float(np.corrcoef(real_mean[nonzero], syn_mean[nonzero])[0, 1])
    return {"pearson_r": r}


# ───────────────────────────────────────────────────────────────────
# Structure
# ───────────────────────────────────────────────────────────────────

def pca_overlap_silhouette(
    real: np.ndarray,
    syn: np.ndarray,
) -> dict[str, float]:
    """PCA overlap measured by silhouette score on 2D PCA projection.

    Projects combined real+synthetic data to 2D PCA, then computes
    silhouette score using real/syn as cluster labels. A score near 0
    means the distributions overlap well (ideal).

    Args:
        real: (N, K, gene_size) real data.
        syn: (M, K, gene_size) synthetic data.

    Returns:
        Dict with silhouette_score.
    """
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    real_flat = real.reshape(real.shape[0], -1).astype(np.float64)
    syn_flat = syn.reshape(syn.shape[0], -1).astype(np.float64)

    combined = np.vstack([real_flat, syn_flat])
    labels = np.array([0] * len(real_flat) + [1] * len(syn_flat))

    # Reduce dimensionality for silhouette computation
    n_components = min(2, combined.shape[1], combined.shape[0] - 1)
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(combined)

    score = float(silhouette_score(projected, labels))
    return {
        "silhouette_score": score,
        "explained_variance": pca.explained_variance_ratio_.tolist(),
    }


def sliced_wasserstein(
    real: np.ndarray,
    syn: np.ndarray,
    n_projections: int = 100,
    seed: int = 42,
) -> dict[str, float]:
    """Sliced Wasserstein Distance via random projections.

    Projects both distributions onto random 1D directions and averages
    the 1D Wasserstein distances.

    Args:
        real: (N, K, gene_size) real data.
        syn: (M, K, gene_size) synthetic data.
        n_projections: Number of random projection directions.
        seed: Random seed.

    Returns:
        Dict with swd (mean), std, and per_projection list.
    """
    rng = np.random.RandomState(seed)
    real_flat = real.reshape(real.shape[0], -1).astype(np.float64)
    syn_flat = syn.reshape(syn.shape[0], -1).astype(np.float64)
    dim = real_flat.shape[1]

    distances = []
    for _ in range(n_projections):
        direction = rng.randn(dim)
        direction /= np.linalg.norm(direction) + 1e-12

        proj_real = real_flat @ direction
        proj_syn = syn_flat @ direction
        dist = wasserstein_distance(proj_real, proj_syn)
        distances.append(float(dist))

    return {
        "swd": float(np.mean(distances)),
        "std": float(np.std(distances)),
    }


# ───────────────────────────────────────────────────────────────────
# Utility
# ───────────────────────────────────────────────────────────────────

def recovery_rate(
    syn_x: np.ndarray,
    syn_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    real_train_x: np.ndarray | None = None,
    real_train_y: np.ndarray | None = None,
) -> dict[str, float]:
    """Recovery Rate: accuracy ratio of classifier trained on syn vs real.

    Trains a simple classifier on synthetic data, tests on real test data.
    If real_train_x is provided, also trains on real for the denominator.

    Args:
        syn_x: (M, K, gene_size) synthetic training data.
        syn_y: (M,) synthetic labels.
        test_x: (N_test, K, gene_size) real test data.
        test_y: (N_test,) real test labels.
        real_train_x: Optional (N_train, K, gene_size) real train data.
        real_train_y: Optional (N_train,) real train labels.

    Returns:
        Dict with syn_accuracy, real_accuracy (if provided), and ratio.
    """
    from sklearn.linear_model import LogisticRegression

    syn_flat = syn_x.reshape(syn_x.shape[0], -1)
    test_flat = test_x.reshape(test_x.shape[0], -1)

    # Train on synthetic, test on real
    clf_syn = LogisticRegression(max_iter=500, solver="lbfgs", n_jobs=-1)
    clf_syn.fit(syn_flat, syn_y)
    syn_acc = float(clf_syn.score(test_flat, test_y))

    result: dict[str, float] = {"syn_accuracy": syn_acc}

    # Train on real, test on real (baseline)
    if real_train_x is not None and real_train_y is not None:
        real_flat = real_train_x.reshape(real_train_x.shape[0], -1)
        clf_real = LogisticRegression(max_iter=500, solver="lbfgs", n_jobs=-1)
        clf_real.fit(real_flat, real_train_y)
        real_acc = float(clf_real.score(test_flat, test_y))
        result["real_accuracy"] = real_acc
        result["ratio"] = syn_acc / max(real_acc, 1e-8)
    else:
        result["ratio"] = syn_acc

    return result


def augmentation_effect(
    real_x: np.ndarray,
    real_y: np.ndarray,
    syn_x: np.ndarray,
    syn_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    ratios: list[float] | None = None,
) -> dict[str, Any]:
    """Measure accuracy improvement when augmenting real data with synthetic.

    Tests multiple mixing ratios of real + synthetic training data.

    Args:
        real_x: (N, K, gene_size) real training data.
        real_y: (N,) real training labels.
        syn_x: (M, K, gene_size) synthetic data.
        syn_y: (M,) synthetic labels.
        test_x: (N_test, K, gene_size) real test data.
        test_y: (N_test,) real test labels.
        ratios: List of real data fractions [0.05, 0.5, 1.0].

    Returns:
        Dict mapping each ratio to its accuracy and improvement over baseline.
    """
    from sklearn.linear_model import LogisticRegression

    if ratios is None:
        ratios = [0.05, 0.5, 1.0]

    real_flat = real_x.reshape(real_x.shape[0], -1)
    syn_flat = syn_x.reshape(syn_x.shape[0], -1)
    test_flat = test_x.reshape(test_x.shape[0], -1)

    # Baseline: real only
    clf_base = LogisticRegression(max_iter=500, solver="lbfgs", n_jobs=-1)
    clf_base.fit(real_flat, real_y)
    base_acc = float(clf_base.score(test_flat, test_y))

    results: dict[str, Any] = {"baseline_real_only": base_acc}

    for ratio in ratios:
        n_real = max(1, int(len(real_flat) * ratio))
        # Subsample real data
        rng = np.random.RandomState(42)
        idx = rng.choice(len(real_flat), n_real, replace=False)

        # Mix real subset + all synthetic
        mix_x = np.vstack([real_flat[idx], syn_flat])
        mix_y = np.concatenate([real_y[idx], syn_y])

        clf = LogisticRegression(max_iter=500, solver="lbfgs", n_jobs=-1)
        clf.fit(mix_x, mix_y)
        acc = float(clf.score(test_flat, test_y))

        results[f"ratio_{ratio}"] = {
            "accuracy": acc,
            "improvement": acc - base_acc,
            "n_real_used": n_real,
            "n_syn_used": len(syn_flat),
        }

    return results


# ───────────────────────────────────────────────────────────────────
# Privacy
# ───────────────────────────────────────────────────────────────────

def nnaa(
    train_real: np.ndarray,
    test_real: np.ndarray,
    syn: np.ndarray,
    metric: str = "euclidean",
) -> dict[str, float]:
    """Nearest Neighbor Adversarial Accuracy.

    AA = 0.5 * [P(real->real closer) + P(syn->syn closer)]
    Ideal value: 0.5

    Args:
        train_real: (N_train, ...) real training data (flattened).
        test_real: (N_test, ...) real test data (flattened).
        syn: (M, ...) synthetic data (flattened).
        metric: Distance metric for cdist.

    Returns:
        Dict with nnaa, real_term, syn_term, gap.
    """
    X_real = train_real.reshape(train_real.shape[0], -1).astype(np.float64)
    X_test = test_real.reshape(test_real.shape[0], -1).astype(np.float64)
    Y_syn = syn.reshape(syn.shape[0], -1).astype(np.float64)

    # Combine train+test real for the "real" pool
    X_all = np.vstack([X_real, X_test])

    dist_rr = cdist(X_all, X_all, metric=metric)
    np.fill_diagonal(dist_rr, np.inf)
    dist_rs = cdist(X_all, Y_syn, metric=metric)

    dist_ss = cdist(Y_syn, Y_syn, metric=metric)
    np.fill_diagonal(dist_ss, np.inf)
    dist_sr = cdist(Y_syn, X_all, metric=metric)

    # Term 1: each real point, is nearest real closer than nearest syn?
    d_r2r = np.min(dist_rr, axis=1)
    d_r2s = np.min(dist_rs, axis=1)
    term1 = float(np.mean(d_r2r < d_r2s))

    # Term 2: each syn point, is nearest syn closer than nearest real?
    d_s2s = np.min(dist_ss, axis=1)
    d_s2r = np.min(dist_sr, axis=1)
    term2 = float(np.mean(d_s2s < d_s2r))

    nnaa_val = 0.5 * (term1 + term2)

    return {
        "nnaa": nnaa_val,
        "real_term": term1,
        "syn_term": term2,
        "ideal": 0.5,
        "gap": abs(nnaa_val - 0.5),
    }


def dupi(
    real: np.ndarray,
    syn: np.ndarray,
    k: int = 1,
    metric: str = "euclidean",
) -> dict[str, Any]:
    """DUPI (Data Utility and Privacy Index) from Jeong et al. (2023).

    DUPI^{<k>} = fraction of real points whose k-th nearest syn neighbor
    is closer than k-th nearest real neighbor (leave-one-out).

    Ideal value: benchmark = m / (n + m - 1) for k=1.

    Args:
        real: (N, ...) real data.
        syn: (M, ...) synthetic data.
        k: Neighbor order (default 1).
        metric: Distance metric.

    Returns:
        Dict with dupi, benchmark, gap, ui_pi decomposition.
    """
    X = real.reshape(real.shape[0], -1).astype(np.float64)
    Y = syn.reshape(syn.shape[0], -1).astype(np.float64)
    n, m = X.shape[0], Y.shape[0]

    # Distances
    dist_xy = cdist(X, Y, metric=metric)
    dist_xx = cdist(X, X, metric=metric)
    np.fill_diagonal(dist_xx, np.inf)

    # k-th nearest distances
    if k == 1:
        d_y_k = np.min(dist_xy, axis=1)
        d_x_k = np.min(dist_xx, axis=1)
    else:
        d_y_k = np.partition(dist_xy, k - 1, axis=1)[:, k - 1]
        d_x_k = np.partition(dist_xx, k - 1, axis=1)[:, k - 1]

    indicators = (d_y_k <= d_x_k).astype(np.float64)
    dupi_val = float(np.mean(indicators))

    # Theoretical benchmark
    if k == 1:
        benchmark = m / (n + m - 1)
    else:
        denom = comb(n - 1 + m, m, exact=True)
        total = 0.0
        for s in range(k, 2 * k):
            total += comb(s - 1, k - 1, exact=True) * comb(
                n - 1 + m - s, m - k, exact=True
            )
        benchmark = total / denom if denom > 0 else 0.5

    gap = abs(dupi_val - benchmark)

    # UI/PI decomposition (tau=5)
    tau = 5.0
    if dupi_val <= benchmark:
        g = dupi_val / (2.0 * benchmark) if benchmark > 0 else 0.0
    else:
        g = (dupi_val - benchmark) / (2.0 * (1.0 - benchmark) + 1e-12) + 0.5

    ui = float(np.arctan(tau * g) / np.arctan(tau))
    pi = float(np.arctan(tau - tau * g) / np.arctan(tau))

    return {
        "dupi": dupi_val,
        "benchmark": float(benchmark),
        "gap": gap,
        "k": k,
        "n_real": n,
        "n_syn": m,
        "ui": ui,
        "pi": pi,
        "ui_pi_product": ui * pi,
    }


def membership_inference_auc(
    train_real: np.ndarray,
    syn: np.ndarray,
    holdout_real: np.ndarray | None = None,
) -> dict[str, float]:
    """Membership inference attack AUC.

    Trains a binary classifier to distinguish between real training data
    (members) and holdout data (non-members) using distances to synthetic
    data as features. AUC near 0.5 means the attack cannot distinguish
    members from non-members, indicating good privacy.

    Args:
        train_real: (N_train, ...) real training data (members).
        syn: (M, ...) synthetic data.
        holdout_real: (N_hold, ...) real holdout data (non-members).
            If None, a random subset of train_real is held out.

    Returns:
        Dict with auc, ideal (0.5), and gap.
    """
    from sklearn.metrics import roc_auc_score

    X_train = train_real.reshape(train_real.shape[0], -1).astype(np.float64)
    Y_syn = syn.reshape(syn.shape[0], -1).astype(np.float64)

    if holdout_real is not None:
        X_hold = holdout_real.reshape(holdout_real.shape[0], -1).astype(np.float64)
    else:
        # Split train_real into member/non-member
        n = len(X_train)
        rng = np.random.RandomState(42)
        perm = rng.permutation(n)
        split = n // 2
        X_hold = X_train[perm[split:]]
        X_train = X_train[perm[:split]]

    # Feature: min distance to synthetic data for each point
    dist_member = cdist(X_train, Y_syn, metric="euclidean")
    feat_member = np.min(dist_member, axis=1)

    dist_nonmember = cdist(X_hold, Y_syn, metric="euclidean")
    feat_nonmember = np.min(dist_nonmember, axis=1)

    # Labels: 1 = member, 0 = non-member
    y_true = np.concatenate([
        np.ones(len(feat_member)),
        np.zeros(len(feat_nonmember)),
    ])
    # Score: negative distance (closer = more likely to be member)
    scores = np.concatenate([-feat_member, -feat_nonmember])

    try:
        auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        auc = 0.5

    return {
        "auc": auc,
        "ideal": 0.5,
        "gap": abs(auc - 0.5),
    }


# ───────────────────────────────────────────────────────────────────
# Robustness
# ───────────────────────────────────────────────────────────────────

def robustness_correlation(
    metrics_per_pop: dict[int, dict[str, float]],
    pop_sizes: dict[int, int],
) -> dict[str, float]:
    """Correlation between population size and generation quality.

    |r| near 0 means quality is independent of population size (robust).

    Args:
        metrics_per_pop: {pop_idx: {"af_correlation": float, ...}}.
        pop_sizes: {pop_idx: n_samples}.

    Returns:
        Dict with correlation (|r|) and raw_r.
    """
    common_pops = set(metrics_per_pop.keys()) & set(pop_sizes.keys())
    if len(common_pops) < 5:
        return {"abs_correlation": float("nan"), "raw_r": float("nan")}

    sizes = [pop_sizes[p] for p in sorted(common_pops)]
    qualities = [
        metrics_per_pop[p].get("af_correlation", 0.0) for p in sorted(common_pops)
    ]

    r = float(np.corrcoef(sizes, qualities)[0, 1])
    return {
        "abs_correlation": abs(r),
        "raw_r": r,
        "n_populations": len(common_pops),
    }
