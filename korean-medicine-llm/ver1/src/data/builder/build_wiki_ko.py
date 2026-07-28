"""wiki_ko general-replay corpus 빌더 (ver4 §R1).

mix 재설계(claudedocs/research_hanmed_cpt_methodology_20260421.md §0)에 따라
일반 한국어 replay 코퍼스를 확보한다. CPT 학습 중 wiki_ko 에서 실제 샘플될
토큰 상한은 cap_tokens(20.4M) × mix(0.15) = 3.06M tokens 이므로, 5M 을
target 으로 잡아 cycling 여유 + 향후 ablation(0.20 비율) 재사용성을 동시에
확보한다.

출력 포맷: preprocess.py stage1 입력 스키마와 동일.
    {"text": <한국어 본문>, "book_id": "wiki_ko", "source_id": "<page_id>"}

book_id 통일 이유: §5.2 의 "한 sequence = 단일 book" 제약은 고전 원전의
도메인 경계 보존(《傷寒論》↔《東醫寶鑑》 혼합 방지)을 위해 설계된 것.
Wikipedia general replay 는 semantic 경계가 training objective 상 무관하므로
모든 문서를 공유 book_id "wiki_ko" 로 묶어 정상 packing(~4–5× compression)을
얻는다. 원본 page id 는 `source_id` 로 보존해 audit trail 유지.

Usage:
    PYTHONHASHSEED=0 .venv/bin/python src/data/builder/build_wiki_ko.py \\
        --target-tokens 5000000 \\
        --min-chars 500 \\
        --output data/cpt/wiki_ko.jsonl

Notes:
- 도메인 중립성: 한의학 기사는 소수라 general anchor 효과 유지. 과도한
  의료 overlap 방지를 위해 카테고리 필터는 적용하지 않음 (wikipedia 덤프의
  자연 분포가 곧 general representation).
- tokenizer-agnostic 대략 토큰 추정을 위해 char/3 heuristic 사용 (한국어
  NFC 기준 한 토큰 ≈ 2~3 chars; 보수적으로 3).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        default="wikimedia/wikipedia",
        help="HF dataset id (default: wikimedia/wikipedia)",
    )
    ap.add_argument(
        "--config",
        default="20231101.ko",
        help="dataset config (default: 20231101.ko)",
    )
    ap.add_argument(
        "--target-tokens",
        type=int,
        default=5_000_000,
        help="대략 목표 토큰 (char/3 heuristic). CPT cap_tokens 20.4M × mix 0.15 = 3.06M 소비 예상; 5M 이면 여유 포함.",
    )
    ap.add_argument(
        "--min-chars",
        type=int,
        default=500,
        help="문서 본문 최소 문자 수 (stub 문서 제거).",
    )
    ap.add_argument(
        "--max-chars",
        type=int,
        default=50_000,
        help="preprocess.py stage1 상한과 동일 (50,000).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("data/cpt/wiki_ko.jsonl"),
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="shuffle seed (streaming shuffle buffer).",
    )
    ap.add_argument(
        "--shuffle-buffer",
        type=int,
        default=10_000,
        help="스트리밍 셔플 버퍼 크기 (메모리 ~ buffer × avg_doc).",
    )
    args = ap.parse_args()

    from datasets import load_dataset  # 지연 import

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[wiki_ko] streaming {args.dataset}:{args.config}")
    ds = load_dataset(args.dataset, args.config, split="train", streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    char_budget = args.target_tokens * 3  # char/3 heuristic
    kept = 0
    total_chars = 0
    skipped_short = 0
    skipped_long = 0
    with args.output.open("w", encoding="utf-8") as fout:
        for rec in ds:
            text = (rec.get("text") or "").strip()
            if not text:
                continue
            n = len(text)
            if n < args.min_chars:
                skipped_short += 1
                continue
            if n > args.max_chars:
                skipped_long += 1
                continue
            page_id = rec.get("id") or str(kept)
            fout.write(json.dumps(
                {
                    "text": text,
                    "book_id": "wiki_ko",
                    "source_id": str(page_id),
                },
                ensure_ascii=False,
            ) + "\n")
            kept += 1
            total_chars += n
            if kept % 500 == 0:
                print(
                    f"  kept={kept:,}  chars={total_chars:,} "
                    f"(~{total_chars // 3:,} tok)  "
                    f"skip[short={skipped_short:,} long={skipped_long:,}]",
                    flush=True,
                )
            if total_chars >= char_budget:
                break

    est_tokens = total_chars // 3
    print(
        f"\n✓ wiki_ko raw: {args.output}\n"
        f"  docs={kept:,}  chars={total_chars:,}  est_tokens~{est_tokens:,}\n"
        f"  target={args.target_tokens:,} tok  (char_budget {char_budget:,})\n"
        f"  next: PYTHONHASHSEED=0 .venv/bin/python src/data/builder/preprocess.py \\\n"
        f"        --corpora wiki_ko --stage all --allow-missing-eval"
    )


if __name__ == "__main__":
    main()
