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

CHROMOSOMES = list(range(1, 23))
PREPROCESS_SEED = 20260327

# PCA grid search
PCA_CANDIDATES = [4, 6, 8, 10, 12, 16]
MARGINAL_GAIN_THRESHOLD = 0.03
MARGINAL_GAIN_DECAY_RATIO = 0.5
PCA_SAMPLE_GENES = 500

# Gene size alignment (CNN downsampling x3 + patch_size 16 -> 128)
GENE_SIZE_ALIGNMENT = 128

# MAF filter
MAF_THRESHOLD = 0.01
MAX_VARIANTS_PER_GENE = 500
