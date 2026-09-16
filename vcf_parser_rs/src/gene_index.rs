/// Gene boundary index with O(log n) position lookup.

pub struct Gene {
    pub name: String,
    pub start: i64,
    pub end: i64,
}

pub struct GeneIndex {
    genes: Vec<Gene>,
    starts: Vec<i64>,
    /// Prefix-maximum of `end` through each index (genes sorted by start).
    /// Mirrors the Python reference's `max_ends`, so the backward walk can
    /// stop as soon as no earlier (smaller-start) gene could still enclose
    /// `pos`, even under overlapping genes.
    max_ends: Vec<i64>,
}

impl GeneIndex {
    pub fn new(mut genes: Vec<Gene>) -> Self {
        genes.sort_by_key(|g| g.start);
        let starts = genes.iter().map(|g| g.start).collect();
        let mut max_ends = Vec::with_capacity(genes.len());
        let mut running_max = i64::MIN;
        for g in &genes {
            running_max = running_max.max(g.end);
            max_ends.push(running_max);
        }
        GeneIndex { genes, starts, max_ends }
    }

    /// Find all gene names matching zero-based RefGene bounds `start < pos <= end`.
    /// Mirrors Python's `_find_genes_for_position` exactly (bisect_left - 1,
    /// then walk backward, breaking only when the prefix-max end can no
    /// longer reach `pos`).
    pub fn find_genes(&self, pos: i64) -> Vec<&str> {
        // bisect_left(starts, pos): count of starts strictly less than pos.
        let bisect_left = self.starts.partition_point(|&s| s < pos);
        if bisect_left == 0 {
            return vec![];
        }
        let idx = bisect_left - 1;

        let mut matched = Vec::new();
        for i in (0..=idx).rev() {
            let g = &self.genes[i];
            if g.start > pos {
                continue;
            }
            if g.end >= pos {
                matched.push(g.name.as_str());
            }
            if i == 0 || self.max_ends[i - 1] < pos {
                break;
            }
        }
        matched
    }
}
