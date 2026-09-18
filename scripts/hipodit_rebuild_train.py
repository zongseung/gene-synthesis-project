# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# ─── How to run ───
# Imported by hipodit_rebuild_check.py; use that file's train command.

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path


class DiagnosticRuntimeError(RuntimeError):
    pass


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def train(args: argparse.Namespace) -> None:
    import numpy as np
    import torch
    import yaml

    from src.models import GaussianDiffusion, HybridCNNDiTFiLM
    from src.preprocessing.tokenizer import (
        invert_normalization,
        load_normalization_stats,
    )
    from hipodit_genotype_check import evaluate_genotypes, provenance

    started = time.monotonic()
    run_dir = args.run_dir or args.output_dir
    if run_dir != args.output_dir:
        run_dir.mkdir(parents=True, exist_ok=False)
    elif (run_dir / "checkpoint_diagnostic.pt").exists():
        raise DiagnosticRuntimeError("Run already contains a checkpoint; choose a new --run-dir")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise DiagnosticRuntimeError("CUDA requested but unavailable in this Python environment")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    data = np.load(args.output_dir / "dataset.npz")
    x = torch.from_numpy(data["x"]).permute(0, 2, 1).contiguous()
    y = torch.from_numpy(data["y"]).long()
    train_indices = torch.from_numpy(data["train_indices"]).long()
    eval_indices = data[f"{args.eval_split}_indices"]
    prepared = json.loads((args.output_dir / "prepare_report.json").read_text())
    pop_mapping = {int(key): int(value) for key, value in prepared["pop_to_superpop"].items()}
    stats_path = args.output_dir / "normalization_stats.pkl"
    feature_schedule = (np.load(args.output_dir / "feature_schedule.npy").T.tolist()
                        if args.schedule == "fisher" else None)
    config = {
        "data": {
            "gene_size": prepared["genes"], "num_channels": prepared["components"],
            "num_classes": 26, "normalize": True, "enforce_zeros": False,
            "processed_dir": str(args.output_dir), "dim_reduction_method": "glm_pca",
            "preprocessing_metadata_path": str(args.output_dir / "preprocessing_metadata.json"),
            "zero_mask_path": str(args.output_dir / "no_zero_mask.pt"),
            "label_hierarchy_path": str(args.output_dir / "label_hierarchy.pkl"),
            "normalization_stats_path": str(stats_path),
            "normalization_stats_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
        },
        "model": {
            "name": "HybridCNNDiTFiLM", "base_channels": 64,
            "channel_mult": [1, 2, 2], "kernel_size": 3, "d_model": 256,
            "n_dit_blocks": 4, "n_heads": 4, "mlp_ratio": 4.0, "dropout": 0.0,
            "patch_size": 2, "n_pops": 26, "n_superpops": 5,
            "pop_to_superpop": pop_mapping,
        },
        "diffusion": {
            "max_timesteps": 1000, "noise_schedule": "linear", "sample_clip": 6.0,
            "guidance_type": "classifier_free", "guidance_weight": 0.0,
            "cfg_dropout_rate": 0.1, "sampling_timesteps": args.ddim_steps, "ddim_eta": 0.0,
            "feature_schedule": feature_schedule,
        },
        "training": {"batch_size": args.batch_size, "epochs": 1, "lr": 2e-4},
        "save_dir": str(run_dir),
    }
    model = HybridCNNDiTFiLM(config).to(device)
    diffusion = GaussianDiffusion(
        timesteps=1000, enforce_zeros=False, null_class=26, cfg_dropout_rate=0.1,
        schedule_type="linear", sample_clip=6.0,
        feature_schedule=feature_schedule,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    probe_indices = train_indices[: min(args.batch_size, len(train_indices))]
    probe_x, probe_y = x[probe_indices].to(device), y[probe_indices].to(device)
    probe_t = torch.full((len(probe_indices),), 500, device=device, dtype=torch.long)
    probe_noise = torch.randn_like(probe_x)
    model.eval()
    with torch.no_grad():
        probe_before = float(diffusion.p_losses(
            model, probe_x, probe_t, probe_y, noise=probe_noise,
            use_min_snr=False, cfg_training=False,
        )["loss"])
    losses, attention_grad_l2 = [], 0.0
    model.train()
    for _ in range(args.steps):
        chosen = train_indices[torch.randint(
            len(train_indices), (args.batch_size,), generator=generator,
        )]
        batch_x, batch_y = x[chosen].to(device), y[chosen].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda",
        ):
            loss = diffusion.p_losses(
                model, batch_x, torch.randint(1000, (len(chosen),), device=device), batch_y,
            )["loss"]
        loss.backward()
        attention_grad_l2 = sum(
            float(parameter.grad.float().square().sum())
            for block in model.dit.blocks for parameter in block.attn.parameters()
            if parameter.grad is not None
        ) ** 0.5
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        probe_after = float(diffusion.p_losses(
            model, probe_x, probe_t, probe_y, noise=probe_noise,
            use_min_snr=False, cfg_training=False,
        )["loss"])
    checkpoint = run_dir / "checkpoint_diagnostic.pt"
    torch.save({
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "config": config, "epoch": 0, "global_step": args.steps,
        "seed": args.seed, "steps": args.steps,
    }, checkpoint)
    (run_dir / "diagnostic_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    count = min(args.samples, len(eval_indices))
    sample_labels = y[torch.from_numpy(eval_indices[:count])].to(device)
    samples_norm = diffusion.sample_ddim(
        model, (count, prepared["components"], prepared["genes"]), sample_labels,
        device, ddim_steps=args.ddim_steps, eta=0.0, guidance_scale=0.0,
    ).permute(0, 2, 1).cpu().numpy()
    stats = load_normalization_stats(
        stats_path, expected_shape=(prepared["genes"], prepared["components"]),
    )
    samples = invert_normalization(samples_norm, stats)
    real = invert_normalization(data["x"][eval_indices[:count]], stats)
    np.save(run_dir / "synthetic_samples_original.npy", samples)
    genotype_metrics = None
    if prepared["glm_family"] == "binomial":
        genotype_metrics, genotype_calls = evaluate_genotypes(
            args.output_dir, samples, real, seed=args.seed + 1,
            decoder_dir=args.decoder_dir, split=args.eval_split)
        np.save(run_dir / "synthetic_genotypes.npy", genotype_calls["B0"])
        if "T" in genotype_calls:
            np.save(run_dir / "synthetic_genotypes_T.npy", genotype_calls["T"])
    labels_generated = sample_labels.cpu().tolist()
    samples_dir = run_dir / "synthetic_samples"
    samples_dir.mkdir(exist_ok=False)
    for index, (sample, label) in enumerate(zip(samples, labels_generated)):
        torch.save(
            (torch.from_numpy(sample.T.copy()), torch.tensor(label, dtype=torch.long)),
            samples_dir / f"sample_pop{label}_{index:04d}.pt",
        )
    labels_probe = sample_labels[: min(8, count)]
    times_probe = torch.zeros(len(labels_probe), device=device, dtype=torch.long)
    with torch.no_grad():
        params = model.film_gen(model.pop_embedding(labels_probe), times_probe)[2]
        gates = torch.cat([value.chunk(6, dim=-1)[2].flatten() for value in params])
    gate_nonzero = int(torch.count_nonzero(gates).cpu())
    checks = {
        "finite_outputs": bool(np.isfinite(samples).all()),
        "finite_losses": bool(np.isfinite(losses).all()),
        "finite_fixed_probe": bool(np.isfinite([probe_before, probe_after]).all()),
        "attention_gradient_positive": args.steps < 2 or attention_grad_l2 > 0.0,
        "attention_gate_open": args.steps < 2 or gate_nonzero > 0,
    }
    report = {
        "status": "complete" if all(checks.values()) else "failed",
        "diagnostic_only": True,
        "schedule": args.schedule,
        "genotype_evaluation": genotype_metrics,
        "eval_split": args.eval_split,
        "decoder_dir": str(args.decoder_dir) if args.decoder_dir else None,
        "cohort_label_source": f"{args.eval_split} labels of the first {count} individuals in "
                               "split order; natural composition, no balanced sampling",
        "label_composition": {label: labels_generated.count(label)
                              for label in sorted(set(labels_generated))},
        "limitation": f"{prepared['genes']} chr17 genes and one training seed; diagnostic only",
        "seed": args.seed, "device": str(device), "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "steps": args.steps, "ddim_steps": args.ddim_steps,
        "samples_generated": count, "sample_space": "original GLM-PCA factor scale",
        "checks": checks, "loss_first": losses[0], "loss_last": losses[-1],
        "loss_min": min(losses), "fixed_denoising_probe_before": probe_before,
        "fixed_denoising_probe_after": probe_after,
        "attention_gradient_l2_last_step": attention_grad_l2,
        "attention_gate_nonzero": gate_nonzero,
        "attention_gate_max_abs": float(gates.abs().max().cpu()),
        # Plan Task 8 step 4: the sequence attention actually sees. One token means no
        # token-to-token long-range attention can be claimed for this panel.
        "dit_latent_size": model.latent_size, "attention_tokens": model.patchify.n_tokens,
        "same_scale_evaluation": {
            "mean_rmse": float(np.sqrt(np.mean((samples.mean(0) - real.mean(0)) ** 2))),
            "std_rmse": float(np.sqrt(np.mean((samples.std(0) - real.std(0)) ** 2))),
        },
        "normalization_roundtrip_max_abs": prepared["normalization_roundtrip_max_abs"],
        "data_split": prepared["split_sizes"], "checkpoint": str(checkpoint),
        "runtime_seconds": time.monotonic() - started, **provenance(),
    }
    _atomic_json(run_dir / "diagnostic_report.json", report)
    _atomic_json(samples_dir / "generation_meta.json", {
        "sample_space": "original", "stats_path": str(stats_path),
        "stats_fingerprint": prepared["normalization_fingerprint"],
        "labels": labels_generated,
    })
    if report["status"] != "complete":
        raise DiagnosticRuntimeError(f"diagnostic checks failed: {checks}")
    print(json.dumps(report, indent=2))
