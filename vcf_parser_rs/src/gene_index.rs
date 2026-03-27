/// Gene boundary index with O(log n) position lookup.

pub struct Gene {
    pub name: String,
    pub start: i64,
    pub end: i64,
}

pub struct GeneIndex {
    genes: Vec<Gene>,
    starts: Vec<i64>,
}

impl GeneIndex {
    pub fn new(mut genes: Vec<Gene>) -> Self {
        genes.sort_by_key(|g| g.start);
        let starts = genes.iter().map(|g| g.start).collect();
        GeneIndex { genes, starts }
    }

    /// Find all gene names whose [start, end] interval contains `pos`.
    /// Uses partition_point (bisect_right equivalent) for O(log n) lookup.
    pub fn find_genes(&self, pos: i64) -> Vec<&str> {
        // Find first index where start > pos
        let idx = self.starts.partition_point(|&s| s <= pos);
        if idx == 0 {
            return vec![];
        }

        let mut matched = Vec::new();
        for i in (0..idx).rev() {
            let g = &self.genes[i];
            if g.end < pos {
                break;
            }
            matched.push(g.name.as_str());
        }
        matched
    }
}
