"""Preprocessing pipeline configuration constants."""

from __future__ import annotations

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# Override to keep a new representation's artifacts out of the existing directory:
#   HIPODIT_PROCESSED_DIR=data/processed_binom2 python src/preprocessing/run_pipeline.py
PROCESSED_DIR = os.environ.get(
    "HIPODIT_PROCESSED_DIR", os.path.join(DATA_DIR, "processed")
)

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
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Per-gene component count used by run_pipeline.py.
PCA_K = 4

# Per-gene dimensionality reduction backend: Poisson GLM-PCA (Townes et al.
# 2019), an explicit count-model approximation for bounded dosage, accelerated
# by glmpca-fast. Recorded in preprocessing_metadata.json and checked against
# configs/default.yaml's data.dim_reduction_method before training.
DIM_RED_METHOD = "glm_pca"

# Population-conditional prior arm folded into the normalization step
# (PriorGrad ICLR 2022 / ShiftDDPMs AAAI 2023 "Data-Normalization"). See
# src.preprocessing.tokenizer.CONDITIONAL_ARMS for the three values.
# Override at runtime: HIPODIT_CONDITIONAL_PRIOR=mean python src/preprocessing/run_pipeline.py
# To derive the conditional splits from an existing processed dir instead of
# re-running the VCF pass, use scripts/make_conditional_prior_data.py.
CONDITIONAL_PRIOR = os.environ.get("HIPODIT_CONDITIONAL_PRIOR", "none")
GLM_PCA_MAX_ITER = int(os.environ.get("HIPODIT_GLM_MAX_ITER", "100"))

# Observation model for the per-gene reduction.
#   'poi'     — Poisson working likelihood (accelerated Rust backend). An
#               approximation: it puts mass on dosage above two and assumes
#               Var = mean instead of 2p(1-p). Measured on 1KG chr22, its mean
#               exceeds two on 3.3% of predictions and its variance is off by
#               more than 2x on 15.6%.
#   'binom2'  — Binomial(2, p), the correctly specified likelihood for diploid
#               dosage (src/preprocessing/binomial_glm_pca.py).
# Override at runtime: HIPODIT_GLM_FAMILY=binom2 python src/preprocessing/run_pipeline.py
GLM_FAMILY = os.environ.get("HIPODIT_GLM_FAMILY", "poi")

# Gene size alignment (CNN downsampling x4 + patch_size 16 -> 256)
GENE_SIZE_ALIGNMENT = 256

# MAF filter
MAF_THRESHOLD = 0.01
MAX_VARIANTS_PER_GENE = 500
