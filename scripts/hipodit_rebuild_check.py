#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
# ─── How to run ───
# Prepare with the project environment, then train with a CUDA-12-compatible
# torch environment. See --help; both stages write only below --output-dir.
"""Run a bounded, reproducible HiPoDiT rebuild on fresh real chr17 data.

This is a diagnostic integration run, not a benchmark or novelty claim.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hipodit_rebuild_prepare import prepare
from hipodit_rebuild_train import train


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def aligned_gene_count(value: str) -> int:
    parsed = positive_int(value)
    if parsed % 8:
        raise argparse.ArgumentTypeError("expected a gene count divisible by 8")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    prepare_parser = sub.add_parser(
        "prepare", help="build a fresh train-only chr17 dataset"
    )
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--genes", type=aligned_gene_count, default=32)
    prepare_parser.add_argument("--components", type=positive_int, default=4)
    prepare_parser.add_argument("--max-variants", type=positive_int, default=128)
    prepare_parser.add_argument("--glm-iterations", type=positive_int, default=1000)
    prepare_parser.add_argument("--seed", type=int, default=20260327)
    train_parser = sub.add_parser(
        "train", help="train, sample, and evaluate the prepared data"
    )
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--run-dir", type=Path)
    train_parser.add_argument("--schedule", choices=("standard", "fisher"), default="standard")
    train_parser.add_argument("--steps", type=positive_int, default=100)
    train_parser.add_argument("--batch-size", type=positive_int, default=64)
    train_parser.add_argument("--samples", type=positive_int, default=64)
    train_parser.add_argument("--ddim-steps", type=positive_int, default=20)
    train_parser.add_argument("--seed", type=int, default=20260327)
    train_parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    train_parser.add_argument("--decoder-dir", type=Path,
                              help="Gate 1'-passing oracle directory whose decoder_T.npz also decodes the generated latents")
    train_parser.add_argument("--eval-split", choices=("val", "test"), default="val")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.stage == "prepare":
        prepare(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
