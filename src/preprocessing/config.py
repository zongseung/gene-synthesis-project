"""Preprocessing pipeline configuration constants."""

from __future__ import annotations

import os
from multiprocessing import cpu_count

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Single merged VCF with tabix index (chr1-22)
VCF_PATH = os.path.join(DATA_DIR, "ALL.autosomes.phase3.genotypes.vcf.gz")
VCF_TBI_PATH = VCF_PATH + ".tbi"
PANEL_PATH = os.path.join(
    DATA_DIR, "integrated_call_samples_v3.20130502.ALL.panel"
)
REFGENE_PATH = os.path.join(DATA_DIR, "refGene.txt.gz")

# Per-chromosome VCF files (for Rust parser — avoids full-file scan)
PER_CHROM_VCF_DIR = os.path.expanduser("~/GeneDiffusion")
PER_CHROM_VCF_PATTERN = "ALL.chr{chrom}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"

CHROMOSOMES = list(range(1, 23))
PREPROCESS_SEED = 20260327

# PCA grid search
PCA_CANDIDATES = [4, 6, 8, 10, 12, 16]
MARGINAL_GAIN_THRESHOLD = 0.03
MARGINAL_GAIN_DECAY_RATIO = 0.5
PCA_SAMPLE_GENES = 500

# Per-gene dimensionality reduction backend.
#   'pca'      — Gaussian PCA (sklearn). Fast (~10 ms/gene); misspecified for
#                 Binomial(2, p) genotype dosage data.
#   'glm_pca'  — GLM-PCA (Townes et al. 2019). Statistically correct for
#                 dosage; ~100× slower per gene (estimate 7-35 h for full 22-chr
#                 run on commodity CPU). Recommended for final analysis after
#                 PCA-based ablation has been verified.
# Override at runtime: HIPODIT_DIM_RED=glm_pca python src/preprocessing/run_pipeline.py
DIM_RED_METHOD = os.environ.get("HIPODIT_DIM_RED", "pca")
GLM_PCA_FAMILY = os.environ.get("HIPODIT_GLM_FAMILY", "poi")  # 'poi' enables Rust path; 'mult' | 'nb' fall back to glmpca-py
GLM_PCA_MAX_ITER = int(os.environ.get("HIPODIT_GLM_MAX_ITER", "100"))

# Gene size alignment (CNN downsampling x3 + patch_size 16 -> 128)
GENE_SIZE_ALIGNMENT = 128

# MAF filter
MAF_THRESHOLD = 0.01
MAX_VARIANTS_PER_GENE = 500
