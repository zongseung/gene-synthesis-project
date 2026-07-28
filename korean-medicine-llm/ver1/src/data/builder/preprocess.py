"""CPT 전처리 파이프라인 — §04a §A (Stage 1 clean) + §A.3/§C.3 (Stage 2 pack).

입력:  data/cpt/{hanmed_bilingual, hanmed_zh_only, hanmed_ko_only}.jsonl
       (extract_corpora.py 출력 — NFC + 공백 + 길이필터 1차 적용)

처리:
  Stage 1 — 정제 (tokenizer 무관)
    F1: Exact dedup (NFC 후 SHA-1)
    F2: 길이 5 ≤ n ≤ 50000
    F3: 동일 문자 run-length > 0.5 → drop  (R3.4 docstring drift fix)
    F4: 언어 비율
        - bilingual: hanja ≥ 0.1 AND hangul ≥ 0.1
        - zh_only:   hanja ≥ 0.4
        - ko_only:   hangul ≥ 0.4
    F5: Eval contamination (SHA-256) — eval/hashes/heldout_*.txt 매치 record drop.
        B4: eval_dir 부재 시 sys.exit(2), drop ratio > 0.5% 시 sys.exit(3).
        --allow-missing-eval 플래그로만 skip 허용.
        Hash 단위 = `rec['text']` 전체 (§A.2 F5 명시).

  Stage 2 — 시퀀스 패킹 (tokenizer 필요)
    - Bllossom tokenizer (§A.3 R3.2 primary) 로 encode
    - greedy pack up to seq_len, 블록 사이 EOS 삽입
    - seq_len default 2048 (§C.3 R3.4 B3 fix)
    - special token 사용 시 extended tokenizer (data/tokenizer/hanmed_bllossom_ext) 필수

출력:
  data/cpt_processed/
    ├── {name}_clean.jsonl       # Stage 1 결과
    ├── {name}_packed_{seq}.jsonl # Stage 2 결과
    └── preprocess_stats.json

Usage (§11.2 W3):
  PYTHONHASHSEED=0 .venv/bin/python src/data/builder/preprocess.py \\
      --stage 1 --input data/cpt --output data/cpt_processed \\
      --eval-hash-dir eval/hashes

  PYTHONHASHSEED=0 .venv/bin/python src/data/builder/preprocess.py \\
      --stage 2 --input data/cpt --output data/cpt_processed \\
      --tokenizer data/tokenizer/hanmed_bllossom_ext --seq-len 2048
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# 프로젝트 루트 import
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.seed import set_global_seed  # noqa: E402


# B4 (§A.2 F5 / §D G1,G5): contamination gate exit codes
EXIT_EVAL_DIR_MISSING = 2
EXIT_CONTAMINATION_OVER_THRESHOLD = 3
CONTAMINATION_DROP_THRESHOLD = 0.005  # 0.5% (§04a §D G5)

# ──────────────────────────────────────────────────────────────────────
# Stage 1 — 정제 (tokenizer 무관)
# ──────────────────────────────────────────────────────────────────────

HANJA_RE = re.compile(r"[\u3400-\u9FFF\uF900-\uFAFF]")
HANGUL_RE = re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]")
ENGLISH_RE = re.compile(r"[A-Za-z]")

# M6 (R3.4): `\s+` 전체 치환은 bilingual block `<ZH>...</ZH>\n<KO>...</KO>\n\n` 의
# 개행 경계를 소실시킴. 공백류(스페이스·탭·전각공백) 만 단일 공백으로 줄이고
# 라인브레이크는 보존. 연속 개행은 최대 2개로 캡핑 (빈 줄 하나만 허용).
HSPACE_RE = re.compile(r"[ \t\u3000]+")
NL_RE = re.compile(r"\n{3,}")

# ver4 §5.3: bilingual block 에서 한문 구간만 추출하기 위한 패턴
ZH_BLOCK_RE = re.compile(r"<ZH>(.*?)</ZH>", re.DOTALL)


def normalize(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = HSPACE_RE.sub(" ", s)
    s = NL_RE.sub("\n\n", s)
    return s.strip()


def text_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def language_ratio(text: str) -> dict[str, float]:
    if not text:
        return {"hanja": 0, "hangul": 0, "english": 0, "other": 1.0}
    n = len(text)
    h = len(HANJA_RE.findall(text))
    k = len(HANGUL_RE.findall(text))
    e = len(ENGLISH_RE.findall(text))
    return {
        "hanja": h / n,
        "hangul": k / n,
        "english": e / n,
        "other": (n - h - k - e) / n,
    }


def char_repeat_ratio(text: str) -> float:
    """동일 문자 최장 run length / 전체 길이. 0.3 이상이면 의심."""
    if not text:
        return 0.0
    max_run = cur = 1
    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 1
    return max_run / len(text)


def quality_check(text: str, corpus_kind: str) -> tuple[bool, str | None]:
    """
    Returns (keep, reason_if_dropped)
    corpus_kind ∈ {bilingual, zh_only, ko_only, en_only}
    """
    n = len(text)
    if n < 5:
        return False, "too_short"
    if n > 50000:
        return False, "too_long"

    rep = char_repeat_ratio(text)
    if rep > 0.5:
        return False, f"char_repeat({rep:.2f})"

    lr = language_ratio(text)

    if corpus_kind == "zh_only":
        # 한문만 — 한자 비율 ≥ 0.4 (구두점·숫자 고려)
        if lr["hanja"] < 0.4:
            return False, f"low_hanja({lr['hanja']:.2f})"
    elif corpus_kind == "ko_only":
        # 한국어만 — 한글 비율 ≥ 0.4 (한자 병기 허용)
        if lr["hangul"] < 0.4:
            return False, f"low_hangul({lr['hangul']:.2f})"
    elif corpus_kind == "en_only":
        if lr["english"] < 0.5:
            return False, f"low_english({lr['english']:.2f})"
    elif corpus_kind == "bilingual":
        # bilingual — 한자와 한글 모두 어느 정도 있어야
        if lr["hanja"] < 0.1:
            return False, f"low_hanja_in_bilingual({lr['hanja']:.2f})"
        if lr["hangul"] < 0.1:
            return False, f"low_hangul_in_bilingual({lr['hangul']:.2f})"
    return True, None


def extract_zh_for_hash(text: str, corpus_kind: str) -> str:
    """ver4 §5.3: corpus 종류별 한문-전용 정규화 문자열을 반환.

    eval/hashes/heldout_*.txt 는 "한문 원문"의 SHA-256 을 담도록 설계됨.
    bilingual 의 경우 `<ZH>...</ZH>\n<KO>...</KO>\n\n` 래핑 전체를 해싱하면
    영영 매치할 수 없으므로 `<ZH>...</ZH>` 구간만 추출해 정규화 후 반환.

    - zh_only:   이미 순수 한문 → normalize() 그대로
    - bilingual: `<ZH>(.*?)</ZH>` 매치 → NFC + HSPACE 축약 + strip.
                 매치 실패 시 방어적으로 full text 반환.
    - ko_only / en_only: 한문 없음 → 빈 문자열 (hash skip)
    """
    if not text:
        return ""
    if corpus_kind == "zh_only":
        return normalize(text)
    if corpus_kind == "bilingual":
        m = ZH_BLOCK_RE.search(text)
        if not m:
            return normalize(text)  # fallback (방어적)
        zh = unicodedata.normalize("NFC", m.group(1))
        zh = HSPACE_RE.sub(" ", zh)
        return zh.strip()
    # ko_only, en_only: 한문 없음 → contamination 해시 skip
    return ""


def load_contamination_hashes(eval_dir: Path | None) -> set[str]:
    """eval/hashes/heldout_*.txt 에서 SHA256 hash set 로드. 없으면 빈 set."""
    hashes = set()
    if not eval_dir or not eval_dir.exists():
        return hashes
    for hash_file in eval_dir.glob("heldout_*.txt"):
        with open(hash_file, encoding="utf-8") as f:
            for line in f:
                h = line.strip()
                if h and len(h) >= 32:
                    hashes.add(h)
    return hashes


_HASH_BASIS_DESC = {
    "zh_only": "zh_normalized",
    "bilingual": "ZH-extract",
    "ko_only": "skipped",
    "en_only": "skipped",
}


def stage1_clean(
    input_path: Path,
    output_path: Path,
    corpus_kind: str,
    contam_hashes: set[str],
) -> dict:
    seen_hashes: set[str] = set()
    stats = {
        "input": 0,
        "kept": 0,
        "dropped_dedup": 0,
        "dropped_quality": Counter(),
        "dropped_contamination": 0,
        "out_chars": 0,
        # ver4 §5.3: hash 계산 basis 기록
        "hash_basis": _HASH_BASIS_DESC.get(corpus_kind, "unknown"),
    }
    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            stats["input"] += 1
            text = normalize(rec.get("text", ""))
            if not text:
                stats["dropped_quality"]["empty_after_norm"] += 1
                continue

            h_sha1 = text_hash(text)
            if h_sha1 in seen_hashes:
                stats["dropped_dedup"] += 1
                continue

            ok, reason = quality_check(text, corpus_kind)
            if not ok:
                stats["dropped_quality"][reason] += 1
                continue

            if contam_hashes:
                # ver4 §5.3: bilingual 은 <ZH>...</ZH> 만 해시. ko_only 는 skip.
                zh_for_hash = extract_zh_for_hash(text, corpus_kind)
                if zh_for_hash:
                    h_sha256 = hashlib.sha256(zh_for_hash.encode("utf-8")).hexdigest()
                    if h_sha256 in contam_hashes:
                        stats["dropped_contamination"] += 1
                        continue

            seen_hashes.add(h_sha1)
            rec["text"] = text
            rec["text_hash_sha1"] = h_sha1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["kept"] += 1
            stats["out_chars"] += len(text)

    stats["dropped_quality"] = dict(stats["dropped_quality"])
    return stats


# ──────────────────────────────────────────────────────────────────────
# Stage 2 — 시퀀스 패킹 (tokenizer 필요)
# ──────────────────────────────────────────────────────────────────────

def stage2_pack(
    input_path: Path,
    output_path: Path,
    tokenizer_name: str,
    seq_len: int,
    record_sep: str = "eos",
) -> dict:
    """record_sep:
      "eos"  (default, 기존 동작) — record 사이에 `<|eot_id|>` 삽입.
      "none" — record 사이 sep 삽입 없이 직접 연결. round_2 진단
               "EOS-as-separator 학습 → F3 loop 환각" 대응 (ver4 §08).
    """
    from transformers import AutoTokenizer

    print(f"  loading tokenizer: {tokenizer_name}")
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # round_2 실측: eos_token_id 를 sep 로 쓰면 모델이 EOS 를 "종결" 이 아닌
    # "문단 전환" 으로 학습 → inference 에서 멈추지 못함 (F3 loop).
    sep_id = tok.eos_token_id if record_sep == "eos" else None
    bos_id = tok.bos_token_id          # §C.3 sequence 시작 BOS 1회
    pad_id = tok.pad_token_id
    print(f"  record_sep={record_sep} (sep_id={sep_id})")

    if bos_id is None:
        raise RuntimeError(
            "tokenizer.bos_token_id is None — §04a §C.3 요구 위반. "
            f"{tokenizer_name} 에서 BOS 가 정의돼야 함."
        )

    # ver4 §5.2: book_id 를 유지하면서 (book_id, text) 튜플로 읽는다.
    rows: list[tuple[object, str]] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rows.append((rec.get("book_id"), rec["text"]))

    print(f"  tokenizing {len(rows)} records …")
    texts = [r[1] for r in rows]
    book_ids = [r[0] for r in rows]
    encs = tok(texts, add_special_tokens=False, truncation=False)["input_ids"]

    stats = {
        "input_records": len(rows),
        "input_total_tokens": sum(len(e) for e in encs),
        "input_avg_tokens": sum(len(e) for e in encs) / len(encs) if encs else 0,
        "input_max_tokens": max((len(e) for e in encs), default=0),
        "packed_sequences": 0,
        "packed_total_tokens": 0,
        "padding_tokens": 0,
        "bos_tokens": 0,
        "sep_tokens": 0,
        "too_long_truncated": 0,
        "seq_len": seq_len,
        "bos_token_id": bos_id,
        "sep_token_id": sep_id,
        "pad_token_id": pad_id,
        # ver4 §5.2: book 경계 관찰용
        "book_boundary_flushes": 0,
        "sequences_per_book": defaultdict(int),
        "unique_books": 0,
    }

    # seq_len 에 맞게 BOS 자리 (+1) 를 남기고 record 를 채운 뒤 pad
    effective_len = seq_len - 1  # BOS 1 자리 예약

    def _flush(cur: list[int], cur_book_ids: list[object], fout) -> None:
        """cur 에 BOS + pad 붙여 정확히 seq_len 로 만든 뒤 기록.

        ver4 §5.2: flush 시점에 모든 포함 record 의 book_id 가 동일한지 검증.
        """
        # 단일 truncated doc 은 cur_book_ids 길이 1 → 자동 pass.
        if cur_book_ids:
            uniq = {b for b in cur_book_ids}
            assert len(uniq) == 1, (
                f"book boundary violation in pack flush: book_ids={sorted(uniq)} "
                "— ver4 §5.2 요구 위반 (한 sequence 는 단일 book 만 포함)"
            )
            book_id_here = next(iter(uniq))
            stats["sequences_per_book"][book_id_here] += 1
        payload = [bos_id] + cur
        if len(payload) > seq_len:
            payload = payload[:seq_len]
        pad_len = seq_len - len(payload)
        payload = payload + [pad_id] * pad_len
        assert len(payload) == seq_len, f"pack invariant broken: {len(payload)}"
        fout.write(json.dumps({"input_ids": payload}, ensure_ascii=False) + "\n")
        stats["packed_sequences"] += 1
        stats["packed_total_tokens"] += seq_len
        stats["padding_tokens"] += pad_len
        stats["bos_tokens"] += 1

    print("  packing …")
    with open(output_path, "w", encoding="utf-8") as fout:
        cur: list[int] = []
        cur_book_ids: list[object] = []
        cur_book_id: object = None  # sentinel: 미정
        for enc, bid in zip(encs, book_ids):
            # ver4 §5.2: book 전환 시 cur 를 강제 flush 하여 경계를 고정.
            if cur and bid != cur_book_id:
                _flush(cur, cur_book_ids, fout)
                stats["book_boundary_flushes"] += 1
                cur = []
                cur_book_ids = []
            if len(enc) >= effective_len:
                # 단일 doc 이 너무 길면 BOS + truncated 로 단독 seq
                if cur:
                    _flush(cur, cur_book_ids, fout)
                    cur = []
                    cur_book_ids = []
                truncated = enc[:effective_len]
                _flush(truncated, [bid], fout)
                stats["too_long_truncated"] += 1
                cur_book_id = bid
                continue
            # BOS 자리 + 현재 cur + (sep 1 if cur else 0) + enc <= seq_len
            # record_sep="none" 이면 sep 없이 직접 연결 (ver4 §08).
            need_sep = 1 if (cur and sep_id is not None) else 0
            if len(cur) + need_sep + len(enc) > effective_len:
                _flush(cur, cur_book_ids, fout)
                cur = list(enc)
                cur_book_ids = [bid]
            else:
                if cur and sep_id is not None:
                    cur.append(sep_id)
                    stats["sep_tokens"] += 1
                cur.extend(enc)
                cur_book_ids.append(bid)
            cur_book_id = bid
        if cur:
            _flush(cur, cur_book_ids, fout)

    # defaultdict → plain dict (키 stringify 로 json 직렬화 보장)
    spb = {str(k): v for k, v in stats["sequences_per_book"].items()}
    stats["sequences_per_book"] = spb
    stats["unique_books"] = len(spb)

    stats["padding_ratio"] = stats["padding_tokens"] / stats["packed_total_tokens"] if stats["packed_total_tokens"] else 0
    stats["compression"] = stats["input_records"] / stats["packed_sequences"] if stats["packed_sequences"] else 0
    return stats


# ──────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────

CORPORA_KINDS = {
    "hanmed_bilingual": "bilingual",
    "hanmed_zh_only": "zh_only",
    "hanmed_ko_only": "ko_only",
    "hanmed_en_only": "en_only",
    # ver4 §2.1.5: synth_facts 는 한국어 평서문 + 한자 병기. ko_only 필터 재사용.
    "hanmed_synth_facts": "ko_only",
    # ver4 §R1 (2026-04-21): general KO replay (Wikipedia-ko). 같은 ko_only
    # 필터(hangul ≥ 0.4) 재사용 — stub/한자편중 문서 자동 배제.
    "wiki_ko": "ko_only",
    # ver4 §8 (2026-04-23) Phase A' — book_008 전용 shards.
    # scripts/build_book008_splits.py 산출. 기존 hanmed_* 와 독립 CORPORA 로
    # 등록하여 Phase A' CPT 에서 mix 비율 개별 제어 가능.
    "book008_bilingual": "bilingual",
    "book008_ko_only": "ko_only",
    "book008_real_facts_identity": "ko_only",
    "book008_real_facts_context": "ko_only",
    "book008_prolog": "ko_only",
}


def _resolve_contamination(
    eval_hash_dir: Path, *, allow_missing: bool
) -> set[str]:
    """B4 (§04a §D G1): eval_dir 부재 시 hard-fail 기본.

    `--allow-missing-eval` 을 명시적으로 주면 warning 후 빈 set 반환.
    """
    if eval_hash_dir.exists():
        hashes = load_contamination_hashes(eval_hash_dir)
        print(
            f"[preprocess] loaded {len(hashes)} contamination hashes "
            f"from {eval_hash_dir}"
        )
        return hashes

    if not allow_missing:
        sys.stderr.write(
            f"[preprocess] ERROR: contamination hash dir not found: "
            f"{eval_hash_dir}\n"
            f"    §04a §D G1 요구 — Stage 1 CPT 착수 전 반드시 필요.\n"
            f"    테스트 목적으로 skip 하려면 --allow-missing-eval 명시.\n"
        )
        sys.exit(EXIT_EVAL_DIR_MISSING)

    sys.stderr.write(
        f"[preprocess] ⚠ --allow-missing-eval: skipping contamination "
        f"(eval/hashes not found at {eval_hash_dir})\n"
    )
    return set()


def _enforce_contamination_gate(stats_per_corpus: dict) -> None:
    """B4 (§04a §D G5): drop ratio > 0.5% 시 sys.exit(3)."""
    for corpus, cstats in stats_per_corpus.items():
        s1 = cstats.get("stage1")
        if not s1 or s1["input"] == 0:
            continue
        drop_ratio = s1["dropped_contamination"] / s1["input"]
        if drop_ratio > CONTAMINATION_DROP_THRESHOLD:
            sys.stderr.write(
                f"[preprocess] ERROR: {corpus} contamination drop ratio "
                f"{drop_ratio:.3%} > {CONTAMINATION_DROP_THRESHOLD:.1%} threshold.\n"
                f"    §04a §D G5 — held-out eval 재구성 또는 수기 확인 필요.\n"
            )
            sys.exit(EXIT_CONTAMINATION_OVER_THRESHOLD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("data/cpt"))
    ap.add_argument("--output", type=Path, default=Path("data/cpt_processed"))
    ap.add_argument(
        "--corpora",
        type=str,
        default="hanmed_bilingual,hanmed_zh_only,hanmed_ko_only",
        help="comma-separated corpus names",
    )
    ap.add_argument("--stage", choices=["1", "2", "all"], default="all")
    ap.add_argument(
        "--tokenizer",
        type=str,
        default="data/tokenizer/hanmed_bllossom_ext",
        help="for stage 2; §04a §A.3 R3.2 Bllossom primary",
    )
    # B3 (§04a §C.3 R3.4): default 2048
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument(
        "--eval-hash-dir",
        type=Path,
        default=Path("eval/hashes"),
        help="contamination check dir; missing → exit 2 unless --allow-missing-eval",
    )
    ap.add_argument(
        "--allow-missing-eval",
        action="store_true",
        help="skip contamination check when eval-hash-dir missing (dev/test only)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--record-sep",
        choices=["eos", "none"],
        default="eos",
        help=(
            "stage2 packing 시 record 사이 구분자. "
            "'eos' (기존 default) 는 <|eot_id|> 삽입 — 단 round_2 진단에서 "
            "EOS 를 '문단 전환' 으로 학습시켜 F3 loop 환각 유발. "
            "'none' 은 sep 없이 직접 연결 (ver4 §08)."
        ),
    )
    args = ap.parse_args()

    set_global_seed(args.seed)

    args.output.mkdir(parents=True, exist_ok=True)
    contam = _resolve_contamination(
        args.eval_hash_dir, allow_missing=args.allow_missing_eval
    )

    all_stats = {
        "preprocess_date": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input),
        "output_dir": str(args.output),
        "tokenizer": args.tokenizer if args.stage in ("2", "all") else None,
        "seq_len": args.seq_len if args.stage in ("2", "all") else None,
        "contamination_hashes_loaded": len(contam),
        "contamination_hash_dir": str(args.eval_hash_dir),
        "allow_missing_eval": args.allow_missing_eval,
        # ver4 §5.3: corpus 별 hash 계산 basis 기록
        "hash_basis_per_kind": dict(_HASH_BASIS_DESC),
        "corpora": {},
    }

    for corpus in args.corpora.split(","):
        corpus = corpus.strip()
        kind = CORPORA_KINDS.get(corpus)
        if not kind:
            print(f"⚠ unknown corpus '{corpus}', skipping")
            continue
        in_path = args.input / f"{corpus}.jsonl"
        # stage 1/all 은 raw input 필요. stage 2 only 는 `_clean.jsonl` 만 있어도 OK.
        if args.stage in ("1", "all") and not in_path.exists():
            print(f"⚠ {in_path} not found, skipping")
            continue
        if args.stage == "2":
            clean_path_check = args.output / f"{corpus}_clean.jsonl"
            if not clean_path_check.exists():
                print(f"⚠ {clean_path_check} not found, skipping")
                continue

        print(f"\n=== {corpus} ({kind}) ===")
        corpus_stats = {}

        if args.stage in ("1", "all"):
            clean_path = args.output / f"{corpus}_clean.jsonl"
            print(f"[Stage 1] {in_path} → {clean_path}")
            s1 = stage1_clean(in_path, clean_path, kind, contam)
            corpus_stats["stage1"] = s1
            print(
                f"  input={s1['input']:,}  kept={s1['kept']:,} "
                f"({s1['kept']/max(s1['input'],1)*100:.1f}%)"
            )
            print(
                f"  dropped: dedup={s1['dropped_dedup']:,}, "
                f"contamination={s1['dropped_contamination']:,}"
            )
            print(f"  quality drops: {s1['dropped_quality']}")
            print(f"  out_chars={s1['out_chars']:,}")

        if args.stage in ("2", "all"):
            clean_path = args.output / f"{corpus}_clean.jsonl"
            packed_path = args.output / f"{corpus}_packed_{args.seq_len}.jsonl"
            print(f"[Stage 2] {clean_path} → {packed_path}")
            s2 = stage2_pack(
                clean_path, packed_path, args.tokenizer, args.seq_len,
                record_sep=args.record_sep,
            )
            corpus_stats["stage2"] = s2
            print(
                f"  records={s2['input_records']:,}  "
                f"total_tokens={s2['input_total_tokens']:,}"
            )
            print(
                f"  avg_tokens/rec={s2['input_avg_tokens']:.0f}  "
                f"max={s2['input_max_tokens']:,}"
            )
            print(
                f"  packed_seqs={s2['packed_sequences']:,}  "
                f"compression={s2['compression']:.1f}x"
            )
            print(f"  padding_ratio={s2['padding_ratio']:.2%}")

        all_stats["corpora"][corpus] = corpus_stats

    # B4 (§04a §D G5): stage1 실행 시 drop ratio gate
    if args.stage in ("1", "all") and not args.allow_missing_eval and contam:
        _enforce_contamination_gate(all_stats["corpora"])

    stats_path = args.output / "preprocess_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f"\n✓ stats: {stats_path}")


if __name__ == "__main__":
    main()
