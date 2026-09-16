# vcf_parser_rs

Rust-based VCF parser with gene boundary mapping, exposed to Python via PyO3.

Reads a bgzipped VCF file, filters biallelic SNPs by MAF, maps variants to gene boundaries (RefGene/GENCODE), and returns per-gene dosage matrices as NumPy arrays — ready for PCA or any downstream analysis.

## Why?

Existing tools handle VCF parsing and gene annotation separately. This crate combines both in a single pass:

```
VCF (bgzipped)  +  Gene annotations (RefGene)
        ↓
Per-gene variant matrices (NumPy arrays)
```

- **13x faster** than Python/cyvcf2 on 1000 Genomes Phase 3 data
- **Zero C library dependency** — pure Rust + flate2 (no htslib/libhts-dev required)
- **PyO3 binding** — `import vcf_parser_rs` in Python, returns `dict[str, np.ndarray]`
- **Gene-aware** — variants mapped to actual gene boundaries via bisect lookup, not arbitrary windows
- **Memory-safe** — process one chromosome at a time, no full-genome memory load

## Installation

Requires Rust toolchain and maturin:

```bash
# From the gene-synthesis-project root, into the project .venv (editable, maturin backend)
uv pip install -e ./vcf_parser_rs

# Or, with maturin directly
pip install maturin
maturin develop --release
```

The installed `.so` does not rebuild itself. After editing anything under `src/`, run the install command again, or Python keeps loading the old parser.

## Usage

### Python

```python
from vcf_parser_rs import process_one_chromosome_rs

# Gene annotations: list of dicts with name, start, end (sorted by start)
gene_list = [
    {"name": "DDX11L1", "start": 11873, "end": 14409},
    {"name": "WASH7P",  "start": 14361, "end": 29370},
    # ... (from RefGene, GENCODE, or any gene annotation)
]

args = (
    22,                          # chromosome number
    "path/to/file.vcf.gz",      # bgzipped VCF
    0.01,                        # MAF threshold
    500,                         # max variants per gene
    gene_list,                   # gene boundaries
    train_indices,               # list[int] of sample rows, or None for all samples
)

chrom_num, gene_matrices, sample_ids = process_one_chromosome_rs(args)

# gene_matrices: dict[str, np.ndarray]
#   key: gene name (e.g., "BRCA1")
#   value: float32 array, shape (n_samples, n_variants)
#
# sample_ids: list[str] — sample IDs from VCF header
```

### What it does per variant

1. **Biallelic SNP filter** — REF and ALT must be single bases (indels and multi-allelic sites excluded)
2. **Genotype to dosage** — `0/0`→0, `0/1`→1, `1/1`→2, `./.`→NaN
3. **MAF filter** — allele frequency is computed from the `train_indices` rows only (all samples when `None`); sites with MAF below the threshold are skipped
4. **Mean imputation** — NaN in *every* sample is filled with the mean of the `train_indices` rows, so held-out samples never shape the filter or the fill value
5. **Gene mapping** — binary search on gene starts, then a backward walk bounded by a prefix-max of gene ends
   - Intergenic variants (outside any gene) are skipped
   - Overlapping genes: variant assigned to all matching genes
6. **Per-gene cap** — only the first `max_variants` variants (VCF position order) are kept for each gene
7. **Matrix assembly** — genes with ≥2 variants are returned as `float32` arrays of shape `(n_samples, n_variants)`

`train_indices` mirrors the Python fallback's `_filter_and_impute_dosage` in `src/preprocessing/vcf_parser.py`, so both parsers produce the same matrices.

The parser prints one line per chromosome to stderr:

```
[chr1] 462879 genic variants, 479593 intergenic skipped, 2576 genes with >=2 variants
```

## Performance

Tested on 1000 Genomes Phase 3 (2,504 samples):

| Chromosome | Size | vcf_parser_rs | Python/cyvcf2 | Speedup |
|-----------|------|--------------|--------------|---------|
| chr22 | 197 MB | 14s | 187s | **13.1x** |
| chr1 | 1.1 GB | 83s | ~1,050s (est) | **~13x** |

## How gene mapping works

```
Gene annotation (RefGene):
  BRCA1: chr17:43,044,295 — 43,170,245

VCF variants on chr17:
  pos 43,091,434  →  inside BRCA1 range  ✓  → assigned to BRCA1
  pos 43,091,560  →  inside BRCA1 range  ✓  → assigned to BRCA1
  pos 43,500,000  →  outside any gene    ✗  → skipped (intergenic)
```

`GeneIndex::find_genes` mirrors Python's `_find_genes_for_position`:

1. `partition_point(|s| s < pos)` (Python `bisect_left`) finds the last gene whose start is below `pos`.
2. It walks backward from there and collects every gene with `start < pos <= end`.
3. It stops once the prefix-max of `end` over all earlier genes is below `pos`. A long gene that starts early and encloses `pos` is still found even when shorter genes sit between it and `pos`.

## Project structure

```
vcf_parser_rs/
├── Cargo.toml
├── pyproject.toml         # maturin build config
└── src/
    ├── lib.rs             # PyO3 module entry point
    ├── gene_index.rs      # Gene boundary index + bisect lookup
    └── vcf_processing.rs  # VCF byte-level parsing + dosage + gene mapping
```

## Dependencies

| Crate | Purpose |
|-------|---------|
| pyo3 0.22 | Python bindings |
| numpy 0.22 | NumPy array interop |
| flate2 1 | bgzf/gzip decompression |

No system C libraries required.

## Coordinate system

- VCF positions: **1-based** (standard)
- RefGene txStart/txEnd: **0-based** start, **0-based exclusive** end
- A gene `[txStart, txEnd)` covers 1-based positions `txStart + 1 … txEnd`, so the lookup tests `start < POS <= end`

## License

MIT
