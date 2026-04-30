#!/usr/bin/env python3
"""Run generation + metric evaluation for multiple CFG guidance weights."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def guidance_tag(weight: float) -> str:
    text = f"{weight:g}".replace("-", "m").replace(".", "p")
    return f"gw_{text}"


def sample_count(syn_dir: Path) -> int:
    return sum(1 for _ in syn_dir.glob("sample_pop*_*.pt"))


def completed_generation(syn_dir: Path) -> bool:
    meta_path = syn_dir / "generation_meta.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return sample_count(syn_dir) == int(meta.get("total_samples", -1))


def run_command(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--model-path",
        default="outputs/autoresearch/autoresearch_runs/baseline/best_model.pth",
    )
    parser.add_argument("--run-dir", default="outputs/guidance_sweep")
    parser.add_argument("--weights", nargs="+", type=float, default=[0.0, 1.0, 2.0, 3.0, 5.0, 7.0])
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--seed", type=int, default=20260327)
    parser.add_argument("--max-per-pop", type=int, default=None)
    parser.add_argument("--sampling-timesteps", type=int, default=None)
    parser.add_argument("--ddim-eta", type=float, default=None)
    parser.add_argument("--batch-gen-size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    rows = []
    for weight in args.weights:
        weight_dir = run_dir / guidance_tag(weight)
        syn_dir = weight_dir / "synthetic_samples"
        eval_dir = weight_dir / "evaluation_metrics"
        syn_dir.mkdir(parents=True, exist_ok=True)

        if args.force or not completed_generation(syn_dir):
            gen_cmd = [
                sys.executable,
                "src/inference/generator.py",
                "--config",
                args.config,
                "--model_path",
                args.model_path,
                "--output_dir",
                str(syn_dir),
                "--guidance_weight",
                str(weight),
                "--seed",
                str(args.seed),
                "--batch_gen_size",
                str(args.batch_gen_size),
            ]
            if args.max_per_pop is not None:
                gen_cmd += ["--max_per_pop", str(args.max_per_pop)]
            if args.sampling_timesteps is not None:
                gen_cmd += ["--sampling_timesteps", str(args.sampling_timesteps)]
            if args.ddim_eta is not None:
                gen_cmd += ["--ddim_eta", str(args.ddim_eta)]
            run_command(gen_cmd, env=env)
        else:
            print(f"Skipping generation for {weight_dir}: existing complete samples", flush=True)

        eval_cmd = [
            sys.executable,
            "scripts/evaluate_synthetic_metrics.py",
            "--syn-dir",
            str(syn_dir),
            "--out-dir",
            str(eval_dir),
        ]
        run_command(eval_cmd)

        summary = load_summary(eval_dir / "summary_metrics.json")
        rows.append({
            "guidance_weight": weight,
            "n_synthetic": summary["counts"]["n_synthetic"],
            "dupi": summary["dupi"]["dupi"],
            "dupi_benchmark": summary["dupi"]["dupi_benchmark"],
            "dupi_abs_error": summary["dupi"]["dupi_abs_error"],
            "utility_index": summary["dupi"]["utility_index"],
            "privacy_index": summary["dupi"]["privacy_index"],
            "centroid_distance": summary["distribution_distances"]["centroid_distance"],
            "gaussian_w2_distance": summary["distribution_distances"]["gaussian_w2_distance"],
            "mmd_rbf_biased": summary["distribution_distances"]["mmd_rbf_biased"],
            "syn_dir": str(syn_dir),
            "eval_dir": str(eval_dir),
        })

    summary_path = run_dir / "guidance_sweep_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved sweep summary: {summary_path}")


if __name__ == "__main__":
    main()
