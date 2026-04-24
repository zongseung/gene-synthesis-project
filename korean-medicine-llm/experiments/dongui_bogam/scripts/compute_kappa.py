from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute Cohen's kappa from two reviewer CSV files.")
    p.add_argument("--a", type=Path, required=True)
    p.add_argument("--b", type=Path, required=True)
    p.add_argument("--resolve", type=Path, default=None)
    p.add_argument("--summary", type=Path, default=None)
    return p.parse_args()


def read_sheet(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {row["id"]: row for row in reader}


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    n = len(labels_a)
    if n == 0:
        return 0.0
    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    categories = set(counts_a) | set(counts_b)
    expected = sum((counts_a[c] / n) * (counts_b[c] / n) for c in categories)
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def main() -> int:
    args = parse_args()
    sheet_a = read_sheet(args.a)
    sheet_b = read_sheet(args.b)
    common_ids = sorted(set(sheet_a) & set(sheet_b))
    labels_a = [sheet_a[row_id].get("label", "").strip() for row_id in common_ids]
    labels_b = [sheet_b[row_id].get("label", "").strip() for row_id in common_ids]
    kappa = cohen_kappa(labels_a, labels_b)
    disagree = [row_id for row_id, la, lb in zip(common_ids, labels_a, labels_b) if la != lb]

    if args.resolve:
        args.resolve.parent.mkdir(parents=True, exist_ok=True)
        with args.resolve.open("w", encoding="utf-8", newline="") as f:
            fieldnames = ["id", "label_a", "label_b", "final_label", "question", "response", "notes_a", "notes_b"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row_id in common_ids:
                row_a = sheet_a[row_id]
                row_b = sheet_b[row_id]
                final_label = row_a["label"] if row_a["label"] == row_b["label"] else ""
                writer.writerow(
                    {
                        "id": row_id,
                        "label_a": row_a.get("label", ""),
                        "label_b": row_b.get("label", ""),
                        "final_label": final_label,
                        "question": row_a.get("question", ""),
                        "response": row_a.get("response", ""),
                        "notes_a": row_a.get("notes", ""),
                        "notes_b": row_b.get("notes", ""),
                    }
                )

    summary = {
        "rows": len(common_ids),
        "kappa": round(kappa, 4),
        "agreement": round(sum(1 for a, b in zip(labels_a, labels_b) if a == b) / max(len(common_ids), 1), 4),
        "disagreement_count": len(disagree),
        "disagreement_ids": disagree,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.summary.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
