"""Throwaway prototypes: remedies for B3 marginal-AF drift. Dev split only; no repo change.

usage: python af_drift_prototypes.py OUT.json [quick]
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/user/gene-synthesis-project")
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from hipodit_genotype_check import genotype_metrics  # noqa: E402
from src.models.genotype_decoder import GenotypeDecoder  # noqa: E402
from src.preprocessing.tokenizer import invert_normalization, load_normalization_stats  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(int(sys.argv[4]) if len(sys.argv) > 4 else 24)
QUICK = len(sys.argv) > 2 and sys.argv[2] == "quick"
OUT = Path(sys.argv[1])
P = ROOT / "outputs/diagnostics/hipodit_fisher_20260915_unique"
O = ROOT / "outputs/diagnostics/hipodit_ld_oracle_20260915"
LAM, SINKHORN_IT, DRAWS = 1.0, 50, 20
SEEDS = [20260915] if QUICK else [20260915 + k for k in range(5)]
MAX_ITER = 20 if QUICK else 250

# ---------------- frozen panel ----------------
geno = np.load(P / "genotypes.npz")
calls, offsets = geno["calls"].astype(np.float64), geno["offsets"]
ds = np.load(P / "dataset.npz")
y = ds["y"].astype(np.int64)
tr, va = ds["train_indices"], ds["val_indices"]
params = pickle.load(open(P / "glm_pca_parameters.pkl", "rb"))
stats = load_normalization_stats(P / "normalization_stats.pkl", expected_shape=(8, 4))
z = invert_normalization(ds["x"], stats).astype(np.float64)
eta0 = np.concatenate([z[:, k] @ p["loadings"].T + p["intercept"] for k, p in enumerate(params)], 1)
genes = json.load(open(P / "gene_variant_map.json"))["genes"]
positions = np.array([v["position"] for g in genes for v in g["variants"]], dtype=np.float64)
train_maf = np.array([v["train_maf"] for g in genes for v in g["variants"]])
maf_bin = np.searchsorted((0.05, 0.2), train_maf, side="right")
bins = GenotypeDecoder.load(O / "decoder_B3.npz").bins
bin_of, edges = bins.bin_of_snp, bins.edges
NB, C, J = len(edges) + 1, 26, calls.shape[1]
linked = bin_of >= 0
lj = np.flatnonzero(linked)
W = torch.as_tensor(np.bincount(y[tr], minlength=C) / len(tr))
hier = pickle.load(open(P / "label_hierarchy.pkl", "rb"))
pop2super = np.array([hier["pop_to_superpop"][k] for k in range(C)], dtype=np.int64)
S_N = int(pop2super.max()) + 1
sup_of_row = pop2super[y]
LOGC = torch.log(torch.tensor([1.0, 2.0, 1.0]))
G3 = torch.arange(3.0)

rng_perm = np.random.default_rng(20260916)
perm = np.arange(J)
for s, e in zip(offsets[:-1], offsets[1:]):
    perm[s:e] = s + rng_perm.permutation(e - s)

# ---------------- models ----------------
def shapes(spec):
    out = {}
    if spec.get("U"):
        out["U"] = (C, J)
    if spec.get("A"):
        out["A"] = (NB, 3, 3)
    if spec.get("delta") == "af":
        out["d"] = (J,)
    if spec.get("delta") == "class":
        out["dc"] = (J, 2)
    if spec.get("theta"):
        out["th"] = {"snp": (J,), True: (J,), "cohort": (C, J), "superpop": (S_N, J)}[spec["theta"]]
    return out


def penalty(spec, pr):
    total = torch.zeros(())
    if "U" in pr:
        total = total + ((pr["U"] - (W @ pr["U"])[None]) ** 2).sum()
    if "A" in pr:
        total = total + (centered_A(pr["A"]) ** 2).sum() if spec.get("family") == "mp" else total + (pr["A"] ** 2).sum()
    for key in ("d", "dc", "th"):
        if key in pr:
            total = total + (pr[key] ** 2).sum()
    return total


def centered_A(A):
    return A - A.mean(2, keepdim=True) - A.mean(1, keepdim=True) + A.mean((1, 2), keepdim=True)


def table(spec, pr, eta, lab):
    """(N,J,3,3) log P(G_j=g | G_{j-1}=h); rows identical where the chain is reset."""
    N = eta.shape[0]
    lin = torch.as_tensor(eta)
    if "U" in pr:
        lin = lin + (pr["U"] - (W @ pr["U"])[None])[torch.as_tensor(lab)]
    if spec.get("family") != "mp":
        if "d" in pr:
            lin = lin + pr["d"][None]
        base = G3 * lin[..., None] + LOGC
        if "dc" in pr:
            base = base + torch.cat([torch.zeros(J, 1), pr["dc"]], 1)[None]
        full = base[:, :, None, :].expand(N, J, 3, 3)
        if "A" in pr:
            mask = torch.as_tensor(linked, dtype=torch.float64)[None, :, None, None]
            full = full + mask * pr["A"][torch.as_tensor(np.maximum(bin_of, 0))][None]
        return torch.log_softmax(full, -1)
    # marginal-preserving family: AF-exact heterozygosity tilt + Sinkhorn-projected association
    if "th" not in pr:
        theta = torch.zeros(1, J)
    elif pr["th"].dim() == 1:
        theta = pr["th"][None]
    elif spec["theta"] == "cohort":
        theta = pr["th"][torch.as_tensor(lab)]
    else:
        theta = pr["th"][torch.as_tensor(pop2super[lab])]
    t = torch.exp(theta)
    q_log = torch.nn.functional.logsigmoid(-lin.abs())  # log min(p, 1-p)
    q = torch.exp(q_log)
    disc = torch.sqrt((t * (1 - 2 * q)) ** 2 + 4 * q * (1 - q))
    log_f = np.log(2.0) + q_log - torch.log(t * (1 - 2 * q) + disc)  # root for q <= 1/2
    log_x = torch.where(lin <= 0, log_f, -log_f)  # x(p) = 1/x(1-p) by g <-> 2-g symmetry
    logits = torch.stack([torch.zeros_like(log_x), np.log(2.0) + theta.expand_as(log_x) + log_x, 2 * log_x], -1)
    log_pi = torch.log_softmax(logits, -1)
    full = log_pi[:, :, None, :].expand(N, J, 3, 3).clone()
    if "A" in pr:
        Ab = centered_A(pr["A"])[torch.as_tensor(bin_of[lj])][None]  # (1,Jl,3,3)
        lp, lc = log_pi[:, lj - 1], log_pi[:, lj]
        f = torch.zeros_like(lp)
        for _ in range(SINKHORN_IT):
            g = lc - torch.logsumexp(Ab + f[..., :, None], dim=-2)
            f = lp - torch.logsumexp(Ab + g[..., None, :], dim=-1)
        log_k = Ab + f[..., :, None] + g[..., None, :]
        full[:, lj] = log_k - torch.logsumexp(log_k, -1, keepdim=True)
        table.col_err = float((torch.logsumexp(log_k, -2) - lc).detach().abs().max())
    return full


def per_call_nll(L, obs):
    g = torch.as_tensor(obs.astype(np.int64))
    h = torch.zeros_like(g)
    h[:, 1:] = g[:, :-1]
    h[:, ~torch.as_tensor(linked)] = 0
    N = g.shape[0]
    return -L[torch.arange(N)[:, None], torch.arange(J)[None, :], h, g]


def fit(spec, c, e):
    pr = {k: torch.zeros(s, requires_grad=True) for k, s in shapes(spec).items()}
    if not pr:
        return pr, 0.0, 0
    n_obs = len(tr) * J
    opt = torch.optim.LBFGS(list(pr.values()), lr=1, max_iter=MAX_ITER, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-10, tolerance_change=1e-14, history_size=50)
    evals = [0]

    def closure():
        opt.zero_grad()
        loss = (per_call_nll(table(spec, pr, e[tr], y[tr]), c[tr]).sum() + LAM * penalty(spec, pr)) / n_obs
        loss.backward()
        evals[0] += 1
        return loss

    opt.step(closure)
    grad = max(float(p.grad.abs().max()) for p in pr.values())
    return pr, grad, evals[0]


def sample(L, u):
    cum = np.cumsum(np.exp(L), -1)
    cum[..., -1] = 1.0
    N = L.shape[0]
    out = np.zeros((N, J), dtype=np.int8)
    prev = np.zeros(N, dtype=np.int64)
    for j in range(J):
        row = cum[np.arange(N), j, prev if linked[j] else 0]
        prev = (u[:, j][:, None] > row).sum(1)
        out[:, j] = prev
    return out


KEYS = ("af_mae", "cohort_af_mae", "heterozygosity_mae", "genotype_proportion_tv", "ld_r2_mae",
        "local_dosage_covariance_mae")

MODELS = {
    "B0": {},
    "T_snp": {"U": 1, "theta": "snp", "family": "mp"},
    "T_superpop": {"U": 1, "theta": "superpop", "family": "mp"},
    "T_cohort": {"U": 1, "theta": "cohort", "family": "mp"},
    "T_snp_noU": {"theta": "snp", "family": "mp"},
    "B1": {"U": 1},
    "B3": {"U": 1, "A": 1},
    "E1_B3+dAF": {"U": 1, "A": 1, "delta": "af"},
    "B1+dClass": {"U": 1, "delta": "class"},
    "E2_B3+dClass": {"U": 1, "A": 1, "delta": "class"},
    "T_B1+tilt": {"U": 1, "theta": 1, "family": "mp"},
    "MP0_B1+SinkhornA": {"U": 1, "A": 1, "family": "mp"},
    "MP_B1+tilt+SinkhornA": {"U": 1, "A": 1, "theta": 1, "family": "mp"},
    "E2_order_shuffle": {"U": 1, "A": 1, "delta": "class", "shuffle": 1},
    "MP_order_shuffle": {"U": 1, "A": 1, "theta": 1, "family": "mp", "shuffle": 1},
}
if QUICK:
    MODELS = {k: MODELS[k] for k in ("B0", "B3", "MP_B1+tilt+SinkhornA")}
if len(sys.argv) > 3 and sys.argv[3] != "all":
    MODELS = {k: MODELS[k] for k in ["B0", *sys.argv[3].split(",")]}

# overfit hypothesis for the Binomial heterozygote deficit
p0 = 1 / (1 + np.exp(-eta0[tr]))
het_pred = 2 * p0 * (1 - p0)
is_het = calls[tr] == 1
results = {"diagnostic": {
    "train_pred_het_given_het": float(het_pred[is_het].mean()),
    "train_pred_het_given_hom": float(het_pred[~is_het].mean()),
    "train_real_het_fraction": float(is_het.mean()),
    "train_pred_het_fraction": float(het_pred.mean())}, "models": {}}
print(json.dumps(results["diagnostic"]), flush=True)

uniforms = {s: np.random.default_rng(s).random((DRAWS, len(va), J)) for s in SEEDS}
for name, spec in MODELS.items():
    started = time.monotonic()
    c, e = (calls[:, perm], eta0[:, perm]) if spec.get("shuffle") else (calls, eta0)
    pr, grad, evals = fit(spec, c, e)
    table.col_err = None
    with torch.no_grad():
        L_train = table(spec, pr, e[tr], y[tr])
        train_nll = float(per_call_nll(L_train, c[tr]).mean())
        del L_train
        L_dev = table(spec, pr, e[va], y[va])
        dev_nll = float(per_call_nll(L_dev, c[va]).mean())
        L_dev = L_dev.numpy()
    per_seed = []
    for s in SEEDS:
        gen = np.concatenate([sample(L_dev, uniforms[s][d]) for d in range(DRAWS)])
        if spec.get("shuffle"):
            restored = np.empty_like(gen)
            restored[:, perm] = gen
            gen = restored
        m = genotype_metrics(calls[va], gen, offsets=offsets, positions=positions, edges=edges,
                             real_labels=y[va], gen_labels=np.tile(y[va], DRAWS), maf_bin_of_snp=maf_bin)
        per_seed.append({k: float(m[k]) for k in KEYS} | {
            "ld_r2_mae_by_bin": [float(v) for v in m["ld_r2_mae_by_bin"]],
            "af_mae_by_maf_bin": [float(v) for v in m["af_mae_by_maf_bin"]]})
    summary = {k: [float(np.mean([r[k] for r in per_seed])), float(np.std([r[k] for r in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0] for k in KEYS}
    summary["ld_r2_mae_by_bin"] = np.mean([r["ld_r2_mae_by_bin"] for r in per_seed], 0).round(5).tolist()
    summary["af_mae_by_maf_bin"] = np.mean([r["af_mae_by_maf_bin"] for r in per_seed], 0).round(5).tolist()
    extra = {}
    if "A" in pr:
        A = (centered_A(pr["A"]) if spec.get("family") == "mp" else pr["A"]).detach().numpy()
        extra["A_diag_minus_offdiag_by_bin"] = [float(np.trace(a) / 3 - (a.sum() - np.trace(a)) / 6) for a in A]
    if "th" in pr:
        th = pr["th"].detach().numpy()
        extra["theta_shape"] = list(th.shape)
        extra["theta_quantiles_5_50_95"] = np.quantile(th, [0.05, 0.5, 0.95]).round(3).tolist()
    results["models"][name] = {"spec": spec, "train_nll": train_nll, "dev_nll": dev_nll, "max_grad": grad,
                               "evals": evals, "sinkhorn_col_err": table.col_err, "summary": summary,
                               "per_seed": per_seed, "runtime_s": time.monotonic() - started, **extra}
    print(f"{name:24s} dev_nll={dev_nll:.6f} " + " ".join(f"{k}={summary[k][0]:.5f}±{summary[k][1]:.5f}" for k in KEYS)
          + f" grad={grad:.1e} evals={evals} colerr={table.col_err} t={time.monotonic() - started:.0f}s", flush=True)
    OUT.write_text(json.dumps(results, indent=1))

# per-seed Gate-1 style comparison against B0 (same uniforms)
b0 = results["models"]["B0"]
for name, r in results["models"].items():
    if name == "B0":
        continue
    verdicts = [(r["dev_nll"] < b0["dev_nll"], s["ld_r2_mae"] < s0["ld_r2_mae"], s["cohort_af_mae"] <= 1.05 * s0["cohort_af_mae"])
                for s, s0 in zip(r["per_seed"], b0["per_seed"])]
    r["gate1_like_pass_seeds"] = sum(all(v) for v in verdicts)
    r["cohort_af_ratio_by_seed"] = [round(s["cohort_af_mae"] / s0["cohort_af_mae"], 4) for s, s0 in zip(r["per_seed"], b0["per_seed"])]
    print(f"{name:24s} gate1-like pass {r['gate1_like_pass_seeds']}/{len(SEEDS)} cohortAF ratio by seed {r['cohort_af_ratio_by_seed']}")
OUT.write_text(json.dumps(results, indent=1))
