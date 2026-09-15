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
    ExperimentError, SCHEDULES, summarize, verify_fingerprints, write_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paired standard/Fisher diagnostic on one frozen prepared split")
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
    prepared_dir, output = args.prepared_dir.resolve(), args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    if output.is_relative_to(prepared_dir):
        raise ExperimentError("Output must be outside the frozen prepared directory")
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        raise ExperimentError("At least two distinct seeds are required")
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
    if args.samples > prepared["split_sizes"]["validation"]:
        raise ExperimentError("Requested samples exceed the frozen validation split")
    source_paths = sorted([*(PROJECT_ROOT / "src").rglob("*.py"),
                           *(PROJECT_ROOT / "scripts").glob("hipodit*.py")])
    runs = []
    for seed in args.seeds:
        for schedule in SCHEDULES:
            command = [sys.executable, str(PROJECT_ROOT / "scripts/hipodit_rebuild_check.py"),
                       "train", "--output-dir", str(prepared_dir), "--run-dir",
                       str(output / f"seed_{seed}" / schedule), "--seed", str(seed),
                       "--schedule", schedule]
            for name, value in parameters.items():
                command.extend(["--" + name.replace("_", "-"), str(value)])
            runs.append({"seed": seed, "schedule": schedule, "command": command})
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "diagnostic_only": True,
        "prepared_dir": str(prepared_dir), "prepared": prepared, "seeds": args.seeds,
        "parameters": parameters, "runs": runs, "python": sys.executable,
        "prepared_sha256": {str(path.relative_to(prepared_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in sorted(prepared_dir.rglob("*")) if path.is_file()},
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        "evaluation": "First requested validation individuals; paired labels and decoder seeds. "
        "Synthetic AF MAE and local dosage covariance MAE, seed-level sample SD. "
        "No fixed-t probe comparison, significance claim, or selective exclusion.",
    }
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
    summary = summarize(output)
    write_summary(output, summary)
    print(json.dumps({"status": "complete", "summary": str(output / "summary.json")}), flush=True)


if __name__ == "__main__":
    run_experiment(build_parser().parse_args())
