from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare duplicate CSV sheets for manual review.")
    p.add_argument("--probes", nargs="+", type=Path, required=True)
    p.add_argument("--out-a", type=Path, required=True)
    p.add_argument("--out-b", type=Path, required=True)
    return p.parse_args()


def load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    row["_source_file"] = path.name
                    rows.append(row)
    return rows


def write_sheet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "category",
        "question",
        "expected",
        "response",
        "_source_file",
        "label",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row.get("id", ""),
                    "category": row.get("category", ""),
                    "question": row.get("question", ""),
                    "expected": row.get("expected", ""),
                    "response": row.get("response", ""),
                    "_source_file": row.get("_source_file", ""),
                    "label": "",
                    "notes": "",
                }
            )


def main() -> int:
    args = parse_args()
    rows = load_rows(args.probes)
    write_sheet(args.out_a, rows)
    write_sheet(args.out_b, rows)
    print(f"[prep_manual_review] rows={len(rows)} out_a={args.out_a} out_b={args.out_b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
