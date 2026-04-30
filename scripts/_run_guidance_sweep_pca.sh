#!/usr/bin/env bash
# Generate ep440 synthetic samples at multiple guidance weights + plot PCA each.
set -euo pipefail

CKPT="outputs/20260428_glmpca_k4_gpu1/checkpoint_epoch440.pth"
RUN_DIR="outputs/20260428_glmpca_k4_gpu1"
FIG_DIR="$RUN_DIR/figures"
mkdir -p "$FIG_DIR"

for W in 1.0 3.0 5.0; do
  TAG="gw${W//./p}"
  SYN="$RUN_DIR/synthetic_samples_ep440_${TAG}"
  echo "==== guidance_weight=$W ===="
  CUDA_VISIBLE_DEVICES=1 PYTHONHASHSEED=0 .venv/bin/python src/inference/generator.py \
    --config configs/default.yaml \
    --model_path "$CKPT" \
    --output_dir "$SYN" \
    --max_per_pop 30 \
    --guidance_weight "$W"
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/plot_pca.py \
    --run-dir "$RUN_DIR" \
    --syn-dir "$SYN" \
    --real-splits test \
    --guidance-weight "$W" \
    --out "$FIG_DIR/pca_ep440_${TAG}.png"
done

echo "DONE_GUIDANCE_SWEEP"
