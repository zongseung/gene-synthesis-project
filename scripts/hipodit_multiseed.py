#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# ─── How to run ───
# Use the existing CUDA training interpreter: python scripts/hipodit_multiseed.py
# --prepared-dir outputs/diagnostics/hipodit_fisher_20260915_unique --output-dir NEW_DIR
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.hipodit_multiseed_summary import (
    DECODER_SEEDS_REQUIRED, ExperimentError, SCHEDULES, summarize, summarize_decoder,
    verify_fingerprints, write_summary, write_summary_decoder,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paired standard/Fisher diagnostic on one frozen prepared split")
    parser.add_argument("--mode", choices=("schedule", "decoder"), default="schedule",
                        help="schedule: paired standard/Fisher runs. decoder: plan rev.2 Gate 3', "
                             "one run per seed decoded by both the B0 and T arms")
    parser.add_argument("--decoder-dir", type=Path, help="Gate 1'-passing oracle directory (decoder mode)")
    parser.add_argument("--eval-split", choices=("val", "test"), default="test")
    parser.add_argument("--prepared-dir", type=Path,
                        default=PROJECT_ROOT / "outputs/diagnostics/hipodit_fisher_20260915_unique")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[20260327, 20260328, 20260329, 20260330, 20260331])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples", type=int, default=251)
    parser.add_argument("--ddim-steps", type=int, default=100)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def run_experiment(args: argparse.Namespace) -> None:
    """Write the declaration before running every pair sequentially, stopping on failure."""
    decoder_mode = args.mode == "decoder"
    prepared_dir, output = args.prepared_dir.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    if output.is_relative_to(prepared_dir):
        raise ExperimentError("Output must be outside the frozen prepared directory")
    if decoder_mode and args.decoder_dir is None:
        raise ExperimentError("Decoder mode needs the Gate 1'-passing --decoder-dir")
    required_seeds = DECODER_SEEDS_REQUIRED if decoder_mode else 2
    if len(args.seeds) < required_seeds or len(set(args.seeds)) != len(args.seeds):
        raise ExperimentError(f"At least {required_seeds} distinct seeds are required")
    if any(seed < 0 or seed > 2**32 - 1 for seed in args.seeds):
        raise ExperimentError("Seeds must be between 0 and 2**32 - 1")
    parameters = {name: getattr(args, name) for name in
                  ("steps", "batch_size", "samples", "ddim_steps", "device")}
    if any(parameters[name] < 1 for name in ("steps", "batch_size", "samples", "ddim_steps")):
        raise ExperimentError("Steps, batch size, samples and DDIM steps must be positive")
    if args.ddim_steps > 1000:
        raise ExperimentError("DDIM steps cannot exceed the 1000 diffusion timesteps")
    prepared = json.loads((prepared_dir / "prepare_report.json").read_text())
    if (prepared["status"] != "prepared" or prepared["glm_family"] != "binomial"
            or prepared["fit_split"] != "train"):
        raise ExperimentError("Expected train-only prepared binomial data")
    split = {"val": "validation", "test": "test"}[args.eval_split] if decoder_mode else "validation"
    if args.samples > prepared["split_sizes"][split]:
        raise ExperimentError(f"Requested samples exceed the frozen {split} split")
    source_paths = sorted([*(PROJECT_ROOT / "src").rglob("*.py"),
                           *(PROJECT_ROOT / "scripts").glob("hipodit*.py")])
    # Decoder mode trains once per seed; the same generated latents feed both the B0 and T arms.
    schedules = ("standard",) if decoder_mode else SCHEDULES
    runs = []
    for seed in args.seeds:
        for schedule in schedules:
            command = [sys.executable, str(PROJECT_ROOT / "scripts/hipodit_rebuild_check.py"),
                       "train", "--output-dir", str(prepared_dir), "--run-dir",
                       str(output / f"seed_{seed}" / schedule), "--seed", str(seed),
                       "--schedule", schedule]
            for name, value in parameters.items():
                command.extend(["--" + name.replace("_", "-"), str(value)])
            if decoder_mode:
                command.extend(["--decoder-dir", str(args.decoder_dir.resolve()),
                                "--eval-split", args.eval_split])
            runs.append({"seed": seed, "schedule": schedule, "command": command})
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "diagnostic_only": True,
        "mode": args.mode,
        "prepared_dir": str(prepared_dir), "prepared": prepared, "seeds": args.seeds,
        "parameters": parameters, "runs": runs, "python": sys.executable,
        "prepared_sha256": {str(path.relative_to(prepared_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in sorted(prepared_dir.rglob("*")) if path.is_file()},
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        "evaluation": "First requested validation individuals; paired labels and decoder seeds. "
        "Synthetic AF MAE and local dosage covariance MAE, seed-level sample SD. "
        "No fixed-t probe comparison, significance claim, or selective exclusion.",
    }
    if decoder_mode:
        decoder_dir = args.decoder_dir.resolve()
        manifest.update({
            "decoder_dir": str(decoder_dir), "eval_split": args.eval_split,
            "decoder_sha256": hashlib.sha256((decoder_dir / "decoder_T.npz").read_bytes()).hexdigest(),
            "gate": "gate3_prime", "plan_revision": "rev2-§9",
            "evaluation": "One training run per seed; the same generated latents decoded by the "
            f"frozen B0 and T arms on the {args.eval_split} split. Paired T-minus-B0 differences "
            "with 95% percentile bootstrap intervals; no selective exclusion.",
        })
    output.mkdir(parents=True, exist_ok=False)
    with (output / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with (output / "runs.jsonl").open("x") as ledger:
        for declared in runs:
            verify_fingerprints(output)
            started = time.monotonic()
            event = {**declared, "status": "started", "started_utc": datetime.now(timezone.utc).isoformat()}
            ledger.write(json.dumps(event) + "\n")
            ledger.flush()
            print(json.dumps(event), flush=True)
            result = subprocess.run(declared["command"], cwd=PROJECT_ROOT, check=False)
            event.update({"status": "complete" if result.returncode == 0 else "failed",
                          "returncode": result.returncode, "runtime_seconds": time.monotonic() - started,
                          "ended_utc": datetime.now(timezone.utc).isoformat()})
            ledger.write(json.dumps(event) + "\n")
            ledger.flush()
            result.check_returncode()
    if decoder_mode:
        summary = summarize_decoder(output)
        write_summary_decoder(output, summary)
        print(json.dumps(summary["gate3_prime"], sort_keys=True, allow_nan=False), flush=True)
        return
    summary = summarize(output)
    write_summary(output, summary)
    print(json.dumps({"status": "complete", "summary": str(output / "summary.json")}), flush=True)


if __name__ == "__main__":
    run_experiment(build_parser().parse_args())
