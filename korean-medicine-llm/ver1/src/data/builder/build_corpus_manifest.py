"""W4 — corpus_v1.json Manifest · Hash Pin builder.

참조:
    §04a §B.3 Manifest · Hash Pin         (스키마 정본)
    §04a §B.2 파일 포맷                    (input/clean/packed 구조)
    §04a §A.3 Tokenizer · §A.4 확장        (tokenizer 정보)
    §04a §A.5 전처리 결정론 (재현성)         (git_sha + seed + data hash pin)
    §11.2 W4                               (CLI, LoC, Exit gate)
    §11 pipeline_data_flow 단계 4          (cpt_trainer.py 가 본 manifest 를 소비)

단일 책임: Stage 1/2 산출물(이미 존재)을 읽어 `data/cpt_processed/corpus_v1.json`
          매니페스트를 생성. Stage 1/2 재실행 금지 (ls/read only).

Usage:
    PYTHONHASHSEED=0 .venv/bin/python src/data/builder/build_corpus_manifest.py

Exit (§11.2 W4):
    - corpus_v1.json 존재 + 모든 *_sha256 필드 64 자리 hex
    - git_sha = 현 HEAD
    - tokenizer_extended = true
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 프로젝트 루트 기준 import 를 위해 sys.path 조정 (tokenizer_extend.py 와 동일 패턴).
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.seed import set_global_seed  # noqa: E402


# §04a §B.3 — 3개 corpus (en_only 는 v1 scope 밖, §01.6).
CORPORA: tuple[str, ...] = ("hanmed_bilingual", "hanmed_zh_only", "hanmed_ko_only")

# §04a §A.3 R3.2 primary tokenizer.
DEFAULT_TOKENIZER_ID = "MLP-KTLim/llama-3-Korean-Bllossom-8B"

# §04a §A.4 확장 후 vocab (128,256 → 128,260).
EXPECTED_EXT_VOCAB = 128260

# §04a §C.3 fixed seq_len.
DEFAULT_SEQ_LEN = 2048

# §04a §B.1 디렉토리 레이아웃 기본 경로.
DEFAULT_PROCESSED_DIR = "data/cpt_processed"
DEFAULT_CPT_DIR = "data/cpt"
DEFAULT_TOKENIZER_DIR = "data/tokenizer/hanmed_bllossom_ext"
DEFAULT_EVAL_HASH_DIR = "eval/hashes"
DEFAULT_OUTPUT = "data/cpt_processed/corpus_v1.json"
DEFAULT_CONTAM_REPORT = "data/stats/contamination_drop.json"

# §B.3 crawl_scope 값 (현 시점 Core 14, Core 25 확장 시 "core_25").
CRAWL_SCOPE = "core_14"

# SHA-256 chunk size — large file 안전 read.
_HASH_CHUNK = 1 << 20  # 1 MiB


def sha256_file(path: Path) -> str:
    """파일 전체 바이트 SHA-256 (chunk read)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def count_jsonl_lines(path: Path) -> int:
    """jsonl 비어있지 않은 라인 수."""
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def git_sha(repo_root: Path) -> str:
    """현재 HEAD commit SHA. 실패 시 'unknown' (§A.5)."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("ascii").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_tokenizer_ext_manifest(tokenizer_dir: Path) -> dict[str, Any]:
    """data/tokenizer/hanmed_bllossom_ext/hanmed_ext_manifest.json 파싱.

    tokenizer_extend.py (W1) 가 생성한 메타에서 vocab/base 재활용.
    """
    mf_path = tokenizer_dir / "hanmed_ext_manifest.json"
    if not mf_path.exists():
        raise FileNotFoundError(
            f"tokenizer extend manifest not found: {mf_path} (W1 미실행 상태)"
        )
    with open(mf_path, encoding="utf-8") as f:
        return json.load(f)


def load_preprocess_stats(processed_dir: Path) -> dict[str, Any]:
    """preprocess_stats.json 파싱 (Stage 1/2 카운터).

    Stage 2 단독 실행 시 stage1 키가 누락될 수 있으므로 호출측에서 None-safe 접근.
    """
    stats_path = processed_dir / "preprocess_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"preprocess_stats.json not found: {stats_path} (W3 미실행 상태)"
        )
    with open(stats_path, encoding="utf-8") as f:
        return json.load(f)


def count_eval_hashes(eval_hash_dir: Path) -> int:
    """eval/hashes/heldout_*.txt 전체 라인 수 (중복 제거 전)."""
    if not eval_hash_dir.exists():
        return 0
    total = 0
    for p in sorted(eval_hash_dir.glob("heldout_*.txt")):
        with open(p, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    total += 1
    return total


def compute_contamination_drop(
    corpus_stats: dict[str, Any] | None,
) -> float | None:
    """stage1 카운터 존재 시 dropped_contamination / input 반환, 없으면 None.

    preprocess_stats.json stage1 이 없는 경우 (Stage 2 단독 실행) 는
    contamination-only drop 비율 산출 불가 → None (JSON null) 으로 기록.
    """
    if not corpus_stats:
        return None
    s1 = corpus_stats.get("stage1")
    if not s1:
        return None
    inp = s1.get("input", 0)
    if inp <= 0:
        return None
    dropped = s1.get("dropped_contamination", 0)
    return dropped / inp


def build_corpus_block(
    corpus: str,
    cpt_dir: Path,
    processed_dir: Path,
    seq_len: int,
    preprocess_stats: dict[str, Any],
) -> dict[str, Any]:
    """단일 corpus 의 manifest 블록 생성 (§B.3 스키마).

    - input_file      : data/cpt/{corpus}.jsonl         (raw extract)
    - clean_file      : data/cpt_processed/{corpus}_clean.jsonl
    - packed_file     : data/cpt_processed/{corpus}_packed_{seq}.jsonl
    - n_blocks_in     : raw jsonl 라인 수
    - n_blocks_clean  : clean jsonl 라인 수
    - n_packed_seqs   : packed jsonl 라인 수
    - total_tokens    : preprocess_stats.json stage2.input_total_tokens 재활용
    """
    input_file = cpt_dir / f"{corpus}.jsonl"
    clean_file = processed_dir / f"{corpus}_clean.jsonl"
    packed_file = processed_dir / f"{corpus}_packed_{seq_len}.jsonl"

    for p in (input_file, clean_file, packed_file):
        if not p.exists():
            raise FileNotFoundError(f"required artifact missing: {p}")

    cstats = preprocess_stats.get("corpora", {}).get(corpus, {})
    s2 = cstats.get("stage2", {})

    return {
        "input_file": str(input_file),
        "input_sha256": sha256_file(input_file),
        "clean_file": str(clean_file),
        "clean_sha256": sha256_file(clean_file),
        "packed_file": str(packed_file),
        "packed_sha256": sha256_file(packed_file),
        "n_blocks_in": count_jsonl_lines(input_file),
        "n_blocks_clean": count_jsonl_lines(clean_file),
        "n_packed_seqs": count_jsonl_lines(packed_file),
        "total_tokens": s2.get("input_total_tokens"),
    }


def build_contamination_block(
    corpora_out: dict[str, dict[str, Any]],
    preprocess_stats: dict[str, Any],
    eval_hash_dir: Path,
    contam_report: str,
) -> dict[str, Any]:
    """§B.3 contamination 섹션 생성."""
    hashes_loaded = preprocess_stats.get(
        "contamination_hashes_loaded",
        count_eval_hashes(eval_hash_dir),
    )
    drop_ratio_per_corpus: dict[str, float | None] = {}
    for corpus in corpora_out:
        cstats = preprocess_stats.get("corpora", {}).get(corpus, {})
        drop_ratio_per_corpus[corpus] = compute_contamination_drop(cstats)
    return {
        "hash_algo": "sha256",
        "hash_dir": str(eval_hash_dir),
        "hashes_loaded": hashes_loaded,
        "drop_ratio_per_corpus": drop_ratio_per_corpus,
        "report": contam_report,
    }


def build_manifest(
    processed_dir: Path,
    cpt_dir: Path,
    tokenizer_dir: Path,
    eval_hash_dir: Path,
    seq_len: int,
    contam_report: str,
    repo_root: Path,
) -> dict[str, Any]:
    """§B.3 corpus_v1 전체 매니페스트 조립."""
    ext_manifest = load_tokenizer_ext_manifest(tokenizer_dir)
    preprocess_stats = load_preprocess_stats(processed_dir)

    corpora_out: dict[str, dict[str, Any]] = {}
    for corpus in CORPORA:
        corpora_out[corpus] = build_corpus_block(
            corpus, cpt_dir, processed_dir, seq_len, preprocess_stats
        )

    return {
        "version": "corpus_v1",
        "build_date": datetime.now(timezone.utc).isoformat(),
        "crawl_scope": CRAWL_SCOPE,
        "tokenizer": ext_manifest.get("base_tokenizer", DEFAULT_TOKENIZER_ID),
        "tokenizer_extended": True,
        "tokenizer_ext_dir": str(tokenizer_dir),
        "tokenizer_ext_vocab_size": ext_manifest.get(
            "vocab_size_after", EXPECTED_EXT_VOCAB
        ),
        "seq_len": seq_len,
        "corpora": corpora_out,
        "contamination": build_contamination_block(
            corpora_out, preprocess_stats, eval_hash_dir, contam_report
        ),
        "git_sha": git_sha(repo_root),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build data/cpt_processed/corpus_v1.json manifest "
            "(§04a §B.3 / §11.2 W4)."
        )
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path(DEFAULT_PROCESSED_DIR),
        help="Stage 1/2 산출 디렉토리 (default: data/cpt_processed)",
    )
    parser.add_argument(
        "--cpt-dir",
        type=Path,
        default=Path(DEFAULT_CPT_DIR),
        help="raw cpt jsonl 디렉토리 (default: data/cpt)",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path(DEFAULT_TOKENIZER_DIR),
        help="확장 tokenizer 디렉토리 (default: data/tokenizer/hanmed_bllossom_ext)",
    )
    parser.add_argument(
        "--eval-hash-dir",
        type=Path,
        default=Path(DEFAULT_EVAL_HASH_DIR),
        help="contamination hash 디렉토리 (default: eval/hashes)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=DEFAULT_SEQ_LEN,
        help="Stage 2 packing seq_len (default: 2048)",
    )
    parser.add_argument(
        "--contam-report",
        type=str,
        default=DEFAULT_CONTAM_REPORT,
        help="contamination report path (manifest 기록용)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help="출력 manifest 경로 (default: data/cpt_processed/corpus_v1.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="전역 seed (§A.5 재현성, default: 42)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_global_seed(args.seed)

    manifest = build_manifest(
        processed_dir=args.processed_dir,
        cpt_dir=args.cpt_dir,
        tokenizer_dir=args.tokenizer_dir,
        eval_hash_dir=args.eval_hash_dir,
        seq_len=args.seq_len,
        contam_report=args.contam_report,
        repo_root=_ROOT,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # 감사용 요약 print (rank 0 only 는 CPT trainer 의 몫, 본 스크립트는 단일 프로세스).
    print(f"[manifest] wrote: {args.output}")
    print(f"[manifest] version: {manifest['version']}")
    print(f"[manifest] tokenizer: {manifest['tokenizer']}")
    print(f"[manifest] vocab: {manifest['tokenizer_ext_vocab_size']}")
    print(f"[manifest] seq_len: {manifest['seq_len']}")
    print(f"[manifest] corpora: {list(manifest['corpora'].keys())}")
    print(f"[manifest] git_sha: {manifest['git_sha']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
