from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two probe runs for determinism.")
    p.add_argument("run_a", type=Path)
    p.add_argument("run_b", type=Path)
    return p.parse_args()


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def main() -> int:
    args = parse_args()
    run_a = load_jsonl(args.run_a)
    run_b = load_jsonl(args.run_b)
    common = sorted(set(run_a) & set(run_b))
    exact = 0
    changed = []
    for row_id in common:
        hash_a = stable_hash(run_a[row_id].get("response", ""))
        hash_b = stable_hash(run_b[row_id].get("response", ""))
        if hash_a == hash_b:
            exact += 1
        else:
            changed.append(row_id)
    ratio = exact / max(len(common), 1)
    print(json.dumps({"common_ids": len(common), "exact_match": exact, "match_rate": round(ratio, 4), "changed_ids": changed}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
