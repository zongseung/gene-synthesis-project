use pyo3::prelude::*;

mod gene_index;
mod vcf_processing;

/// Process a single chromosome from a tabix-indexed VCF file.
///
/// Args:
///     args: tuple of (chrom_num, vcf_path, maf_threshold, max_variants, gene_list, train_indices)
///         - chrom_num: int (1-22)
///         - vcf_path: str (path to merged bgzipped VCF with .tbi index)
///         - maf_threshold: float (e.g. 0.01)
///         - max_variants: int (e.g. 500)
///         - gene_list: list[dict] with keys "name" (str), "start" (int), "end" (int)
///         - train_indices: Optional[list[int]] — when given, the MAF filter and
///           mean-imputation value are computed from these sample rows only
///           (matches Python's `_filter_and_impute_dosage`); when None, all
///           samples are used.
///
/// Returns:
///     tuple of (chrom_num, gene_matrices, sample_ids)
///         - gene_matrices: dict[str, numpy.ndarray(float32, shape=(n_samples, n_variants))]
///         - sample_ids: list[str]
#[pyfunction]
fn process_one_chromosome_rs(
    py: Python<'_>,
    args: (i32, String, f64, usize, Vec<PyObject>, Option<Vec<usize>>),
) -> PyResult<(i32, PyObject, Vec<String>)> {
    let (chrom_num, vcf_path, maf_threshold, max_variants, gene_list, train_indices) = args;
    vcf_processing::process_chromosome(
        py,
        chrom_num,
        &vcf_path,
        maf_threshold,
        max_variants,
        &gene_list,
        train_indices,
    )
}

#[pymodule]
fn vcf_parser_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_one_chromosome_rs, m)?)?;
    Ok(())
}
