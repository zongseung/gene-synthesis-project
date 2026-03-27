/// Core VCF processing: read bgzipped VCF, filter, compute dosage, assign to genes.
///
/// Uses byte-level parsing for maximum throughput. No per-line String allocation.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

use flate2::read::MultiGzDecoder;
use numpy::PyArray2;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::gene_index::{Gene, GeneIndex};

fn parse_gene_list(py: Python<'_>, gene_list: &[PyObject]) -> PyResult<Vec<Gene>> {
    let mut genes = Vec::with_capacity(gene_list.len());
    for obj in gene_list {
        let dict = obj.downcast_bound::<PyDict>(py)?;
        let name: String = dict
            .get_item("name")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'name'"))?
            .extract()?;
        let start: i64 = dict
            .get_item("start")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'start'"))?
            .extract()?;
        let end: i64 = dict
            .get_item("end")?
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'end'"))?
            .extract()?;
        genes.push(Gene { name, start, end });
    }
    Ok(genes)
}

/// Parse genotype from bytes (e.g., b"0/0:..." or b"0|1:...") into dosage.
#[inline(always)]
fn parse_gt_dosage(field: &[u8]) -> Option<f32> {
    // Fast path: most fields start with "X/Y:" or "X|Y:" where X,Y are single digits
    if field.len() >= 3 {
        let (a1, sep, a2) = (field[0], field[1], field[2]);
        if sep == b'/' || sep == b'|' {
            if a1 == b'.' || a2 == b'.' {
                return None;
            }
            let d1 = if a1 > b'0' { 1u8 } else { 0 };
            let d2 = if a2 > b'0' { 1u8 } else { 0 };
            return Some((d1 + d2) as f32);
        }
    }

    // Slow path for unusual genotypes (multi-digit alleles, etc.)
    let gt_end = field.iter().position(|&b| b == b':').unwrap_or(field.len());
    let gt = &field[..gt_end];
    let mut alt_count: u8 = 0;
    for allele in gt.split(|&b| b == b'/' || b == b'|') {
        if allele == b"." {
            return None;
        }
        if allele.len() == 1 && allele[0] > b'0' && allele[0] <= b'9' {
            alt_count += 1;
        } else if let Ok(a) = std::str::from_utf8(allele)
            .unwrap_or("0")
            .parse::<u8>()
        {
            if a > 0 {
                alt_count += 1;
            }
        }
    }
    Some(alt_count as f32)
}

/// Fast i64 parser for position field (ASCII digits only, no sign).
#[inline(always)]
fn parse_pos(field: &[u8]) -> Option<i64> {
    let mut val: i64 = 0;
    for &b in field {
        if b < b'0' || b > b'9' {
            return None;
        }
        val = val * 10 + (b - b'0') as i64;
    }
    if field.is_empty() {
        None
    } else {
        Some(val)
    }
}

/// Split a byte slice by tabs, returning an iterator of slices.
struct TabSplit<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> TabSplit<'a> {
    fn new(data: &'a [u8]) -> Self {
        TabSplit { data, pos: 0 }
    }
}

impl<'a> Iterator for TabSplit<'a> {
    type Item = &'a [u8];

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos > self.data.len() {
            return None;
        }
        let start = self.pos;
        match self.data[start..].iter().position(|&b| b == b'\t') {
            Some(i) => {
                self.pos = start + i + 1;
                Some(&self.data[start..start + i])
            }
            None => {
                self.pos = self.data.len() + 1;
                Some(&self.data[start..])
            }
        }
    }
}

pub fn process_chromosome(
    py: Python<'_>,
    chrom_num: i32,
    vcf_path: &str,
    maf_threshold: f64,
    max_variants: usize,
    gene_list_py: &[PyObject],
) -> PyResult<(i32, PyObject, Vec<String>)> {
    let genes = parse_gene_list(py, gene_list_py)?;
    let gene_index = GeneIndex::new(genes);

    let file = File::open(vcf_path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!(
            "[chr{}] Failed to open VCF {}: {}",
            chrom_num, vcf_path, e
        ))
    })?;

    let decoder = MultiGzDecoder::new(BufReader::with_capacity(4 * 1024 * 1024, file));
    let mut reader = BufReader::with_capacity(2 * 1024 * 1024, decoder);

    let mut sample_ids: Vec<String> = Vec::new();
    let mut n_samples = 0usize;
    let mut gene_variants: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
    let mut n_variants: u64 = 0;
    let mut n_intergenic: u64 = 0;
    let mut line_buf = Vec::with_capacity(128 * 1024); // reusable buffer

    loop {
        line_buf.clear();
        let bytes_read = reader.read_until(b'\n', &mut line_buf).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("[chr{}] Read error: {}", chrom_num, e))
        })?;
        if bytes_read == 0 {
            break;
        }

        // Trim trailing newline
        let line = if line_buf.last() == Some(&b'\n') {
            &line_buf[..line_buf.len() - 1]
        } else {
            &line_buf[..]
        };

        if line.is_empty() {
            continue;
        }

        // Header lines
        if line[0] == b'#' {
            if line.starts_with(b"#CHROM") {
                let fields: Vec<&[u8]> = TabSplit::new(line).collect();
                if fields.len() > 9 {
                    sample_ids = fields[9..]
                        .iter()
                        .map(|f| String::from_utf8_lossy(f).to_string())
                        .collect();
                    n_samples = sample_ids.len();
                }
            }
            continue;
        }

        if n_samples == 0 {
            continue;
        }

        // Parse fields using zero-copy tab splitting
        let mut fields = TabSplit::new(line);

        let _chrom = match fields.next() {
            Some(f) => f,
            None => continue,
        };
        let pos_field = match fields.next() {
            Some(f) => f,
            None => continue,
        };
        let _id = fields.next(); // skip ID
        let ref_field = match fields.next() {
            Some(f) => f,
            None => continue,
        };
        let alt_field = match fields.next() {
            Some(f) => f,
            None => continue,
        };

        // Biallelic SNP filter
        if ref_field.len() != 1 || alt_field.len() != 1 {
            continue;
        }
        // Check for multi-allelic (comma in ALT)
        if alt_field.contains(&b',') {
            continue;
        }

        let pos = match parse_pos(pos_field) {
            Some(p) => p,
            None => continue,
        };

        // Skip QUAL, FILTER, INFO, FORMAT (fields 5-8)
        let _qual = fields.next();
        let _filter = fields.next();
        let _info = fields.next();
        let _format = fields.next();

        // Parse sample genotypes
        let mut dosage = vec![f32::NAN; n_samples];
        let mut valid_sum: f64 = 0.0;
        let mut valid_count: u32 = 0;

        for i in 0..n_samples {
            let gt_field = match fields.next() {
                Some(f) => f,
                None => break,
            };
            if let Some(d) = parse_gt_dosage(gt_field) {
                dosage[i] = d;
                valid_sum += d as f64;
                valid_count += 1;
            }
        }

        if valid_count == 0 {
            continue;
        }

        // MAF filter
        let af = valid_sum / (2.0 * valid_count as f64);
        let maf = af.min(1.0 - af);
        if maf < maf_threshold {
            continue;
        }

        // Mean imputation
        let mean_val = (valid_sum / valid_count as f64) as f32;
        for d in dosage.iter_mut() {
            if d.is_nan() {
                *d = mean_val;
            }
        }

        // Gene lookup
        let matched = gene_index.find_genes(pos);
        if matched.is_empty() {
            n_intergenic += 1;
            continue;
        }

        for gene_name in matched {
            let variants = gene_variants.entry(gene_name.to_string()).or_default();
            if variants.len() < max_variants {
                variants.push(dosage.clone());
            }
        }

        n_variants += 1;
    }

    // Convert to numpy dict (genes with >= 2 variants)
    let dict = PyDict::new_bound(py);
    let mut n_genes_out = 0u32;

    for (gene_name, variants) in &gene_variants {
        if variants.len() < 2 {
            continue;
        }
        let n_vars = variants.len();

        let rows: Vec<Vec<f32>> = (0..n_samples)
            .map(|row| (0..n_vars).map(|col| variants[col][row]).collect())
            .collect();

        let array = PyArray2::<f32>::from_vec2_bound(py, &rows).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Failed to create numpy array for {}: {}",
                gene_name, e
            ))
        })?;
        dict.set_item(gene_name, array)?;
        n_genes_out += 1;
    }

    eprintln!(
        "[chr{}] {} genic variants, {} intergenic skipped, {} genes with >=2 variants",
        chrom_num, n_variants, n_intergenic, n_genes_out
    );

    Ok((chrom_num, dict.into(), sample_ids))
}
