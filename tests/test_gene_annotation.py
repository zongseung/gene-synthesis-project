import gzip

from src.preprocessing.gene_annotation import load_refgene


def _row(name: str, chrom: str, start: int, end: int) -> str:
    fields = ["0", "NM_1", chrom, "+", str(start), str(end), str(start), str(end),
              "1", f"{start},", f"{end},", "0", name, "cmpl", "cmpl", "0,"]
    return "\t".join(fields) + "\n"


def test_distant_loci_of_one_name_stay_separate_and_unique(tmp_path):
    path = tmp_path / "refGene.txt.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(_row("RNU1", "chr1", 100, 200))
        handle.write(_row("RNU1", "chr1", 5_000_000, 5_000_100))  # distant copy
        handle.write(_row("RNU1", "chr2", 300, 400))              # other chromosome
        handle.write(_row("GENE", "chr1", 1_000, 2_000))
        handle.write(_row("GENE", "chr1", 1_500, 3_000))          # overlapping isoform

    coords = load_refgene(str(path))

    chr1 = {g["name"]: (g["start"], g["end"]) for g in coords["1"]}
    assert chr1["GENE"] == (1_000, 3_000)
    assert chr1["RNU1@chr1-100"] == (100, 200)
    assert chr1["RNU1@chr1-5000000"] == (5_000_000, 5_000_100)
    assert [g["name"] for g in coords["2"]] == ["RNU1@chr2-300"]
