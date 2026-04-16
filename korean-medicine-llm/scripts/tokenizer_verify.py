"""Extended Bllossom tokenizer 검증 — §04a §A.3 / §A.4 정합성 확인.

기획서 주장과 실제 tokenizer 동작이 일치하는지 5 가지 검증:
  V1. Special token 4 개가 single token 으로 encode 되며 id 128256~128259 할당
  V2. Llama-3 chat template (`<|start_header_id|>`, `<|eot_id|>`) 기존 보존
  V3. Bilingual block encode → decode round-trip (tag 손실 없음)
  V4. 실측 tok/char (한문·한국어) — §A.3 R3.2 수치 재확인
  V5. 실제 corpus (data/cpt/hanmed_bilingual.jsonl) 에서 샘플 10 개 encode → 통계

결과를 data/stats/tokenizer_verify.json 에 저장해 감사 가능하게 한다.

Usage:
    PYTHONHASHSEED=0 .venv/bin/python scripts/tokenizer_verify.py \\
        --tokenizer data/tokenizer/hanmed_bllossom_ext
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.seed import set_global_seed  # noqa: E402


EXPECTED_SPECIAL_IDS = {
    "<ZH>": 128256,
    "</ZH>": 128257,
    "<KO>": 128258,
    "</KO>": 128259,
}

LLAMA3_CHAT_TOKENS = [
    "<|begin_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
]


def v1_special_token_ids(tok) -> dict:
    """V1: special token 4 개 → single token + id 할당 확인."""
    results = {}
    for sp, expected_id in EXPECTED_SPECIAL_IDS.items():
        ids = tok.encode(sp, add_special_tokens=False)
        results[sp] = {
            "n_tokens": len(ids),
            "ids": ids,
            "expected_id": expected_id,
            "ok": len(ids) == 1 and ids[0] == expected_id,
        }
    return results


def v2_llama3_chat_tokens(tok) -> dict:
    """V2: Llama-3 chat template token 보존."""
    results = {}
    for t in LLAMA3_CHAT_TOKENS:
        ids = tok.encode(t, add_special_tokens=False)
        results[t] = {
            "n_tokens": len(ids),
            "ids": ids,
            "ok": len(ids) == 1,  # single token 기대
        }
    return results


def v3_block_roundtrip(tok) -> dict:
    """V3: bilingual block encode → decode 무손실."""
    block = "<ZH>東醫寶鑑者 我國醫學之大全也</ZH>\n<KO>동의보감은 우리나라 의학의 집대성이다.</KO>\n\n"
    ids = tok.encode(block, add_special_tokens=False)
    decoded = tok.decode(ids, skip_special_tokens=False)
    return {
        "input_len_chars": len(block),
        "encoded_len_tokens": len(ids),
        "decoded": decoded,
        "input_hash": hash(block),
        "decoded_hash": hash(decoded),
        "tag_preserved": all(tag in decoded for tag in EXPECTED_SPECIAL_IDS),
        "ok": all(tag in decoded for tag in EXPECTED_SPECIAL_IDS),
    }


def v4_tok_per_char(tok) -> dict:
    """V4: 실측 tok/char — §A.3 R3.2 수치 (zh 1.040, ko 0.745) 재확인."""
    zh = "乾鑿度云, 天形出乎乾, 有太易太初太始太素"
    ko = "《건착도》에, 하늘[天]의 형(形)은 건(乾)에서 나오니"
    zh_ids = tok.encode(zh, add_special_tokens=False)
    ko_ids = tok.encode(ko, add_special_tokens=False)
    return {
        "zh": {
            "chars": len(zh),
            "tokens": len(zh_ids),
            "tok_per_char": len(zh_ids) / len(zh),
        },
        "ko": {
            "chars": len(ko),
            "tokens": len(ko_ids),
            "tok_per_char": len(ko_ids) / len(ko),
        },
    }


def v5_corpus_sample(tok, jsonl_path: Path, n_sample: int = 10) -> dict:
    """V5: 실제 corpus 샘플 10 개 encode 통계."""
    with open(jsonl_path, encoding="utf-8") as f:
        lines = f.readlines()
    rng = random.Random(42)
    rng.shuffle(lines)
    samples = []
    for line in lines[:n_sample]:
        rec = json.loads(line)
        text = rec["text"]
        ids = tok.encode(text, add_special_tokens=False)
        samples.append(
            {
                "book_id": rec.get("book_id"),
                "chars": len(text),
                "tokens": len(ids),
                "tok_per_char": len(ids) / max(len(text), 1),
                "first_5_tokens": ids[:5],
                "contains_zh_tag": 128256 in ids,
                "contains_ko_tag": 128258 in ids,
            }
        )
    total_chars = sum(s["chars"] for s in samples)
    total_tokens = sum(s["tokens"] for s in samples)
    return {
        "jsonl_path": str(jsonl_path),
        "n_sample": len(samples),
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "avg_tok_per_char": total_tokens / max(total_chars, 1),
        "all_contain_both_tags": all(
            s["contains_zh_tag"] and s["contains_ko_tag"] for s in samples
        ),
        "samples": samples,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("data/tokenizer/hanmed_bllossom_ext"),
        help="extended Bllossom tokenizer path",
    )
    ap.add_argument(
        "--bilingual-jsonl",
        type=Path,
        default=Path("data/cpt/hanmed_bilingual.jsonl"),
        help="corpus sample 용",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("data/stats/tokenizer_verify.json"),
    )
    args = ap.parse_args()

    set_global_seed(42)

    print(f"[verify] tokenizer: {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(str(args.tokenizer))
    print(f"[verify] vocab_size: {len(tok):,}")

    report = {
        "verify_date": datetime.now(timezone.utc).isoformat(),
        "tokenizer_path": str(args.tokenizer),
        "vocab_size": len(tok),
        "V1_special_token_ids": v1_special_token_ids(tok),
        "V2_llama3_chat_tokens": v2_llama3_chat_tokens(tok),
        "V3_block_roundtrip": v3_block_roundtrip(tok),
        "V4_tok_per_char": v4_tok_per_char(tok),
    }

    if args.bilingual_jsonl.exists():
        report["V5_corpus_sample"] = v5_corpus_sample(tok, args.bilingual_jsonl)
    else:
        report["V5_corpus_sample"] = {"skipped": f"{args.bilingual_jsonl} not found"}

    # 요약 출력
    print("\n[V1 — Special tokens]")
    for sp, r in report["V1_special_token_ids"].items():
        mark = "✅" if r["ok"] else "❌"
        print(f"  {mark} {sp:10s} ids={r['ids']}  (expected {r['expected_id']})")

    print("\n[V2 — Llama-3 chat tokens]")
    for t, r in report["V2_llama3_chat_tokens"].items():
        mark = "✅" if r["ok"] else "⚠"
        print(f"  {mark} {t:24s} ids={r['ids']}")

    print("\n[V3 — Block round-trip]")
    r3 = report["V3_block_roundtrip"]
    mark = "✅" if r3["ok"] else "❌"
    print(f"  {mark} input {r3['input_len_chars']} chars → {r3['encoded_len_tokens']} tokens; tag_preserved={r3['tag_preserved']}")

    print("\n[V4 — tok/char]")
    v4 = report["V4_tok_per_char"]
    print(f"  한문: {v4['zh']['tokens']:>3d} tokens / {v4['zh']['chars']:>2d} chars = {v4['zh']['tok_per_char']:.3f}")
    print(f"  한글: {v4['ko']['tokens']:>3d} tokens / {v4['ko']['chars']:>2d} chars = {v4['ko']['tok_per_char']:.3f}")
    print("  §A.3 기대: zh 1.040, ko 0.745 (R3.2 실측 기준)")

    if "samples" in report["V5_corpus_sample"]:
        print("\n[V5 — Corpus sample 10]")
        v5 = report["V5_corpus_sample"]
        print(f"  total {v5['total_chars']} chars → {v5['total_tokens']} tokens (avg {v5['avg_tok_per_char']:.3f})")
        print(f"  all_contain_both_tags = {v5['all_contain_both_tags']}")

    # 전체 pass/fail 판정
    all_ok = (
        all(r["ok"] for r in report["V1_special_token_ids"].values())
        and report["V3_block_roundtrip"]["ok"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[verify] report → {args.output}")
    if not all_ok:
        print("[verify] ❌ some checks failed")
        sys.exit(1)
    print("[verify] ✅ all critical checks passed")


if __name__ == "__main__":
    main()
