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
# Install maturin
pip install maturin

# Build and install into current Python environment
maturin develop --release
```

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
)

chrom_num, gene_matrices, sample_ids = process_one_chromosome_rs(args)

# gene_matrices: dict[str, np.ndarray]
#   key: gene name (e.g., "BRCA1")
#   value: float32 array, shape (n_samples, n_variants)
#
# sample_ids: list[str] — sample IDs from VCF header
```

### What it does per variant

1. **Biallelic SNP filter** — REF and ALT must be single bases (indels excluded)
2. **Genotype to dosage** — `0/0`→0, `0/1`→1, `1/1`→2, `./.`→NaN→mean imputation
3. **MAF filter** — minor allele frequency below threshold is skipped
4. **Gene mapping** — O(log n) bisect lookup against gene boundaries
   - Intergenic variants (outside any gene) are skipped
   - Overlapping genes: variant assigned to all matching genes
5. **Matrix assembly** — genes with ≥2 variants are returned as NumPy arrays

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

Lookup uses sorted gene start positions + binary search (`partition_point`), equivalent to Python's `bisect.bisect_right`. Each variant is checked against all genes whose `start ≤ pos ≤ end`.

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
- Gene lookup compares 1-based VCF POS against gene boundaries as-is (matching the convention used by cyvcf2 and most VCF tools)

## License

MIT
