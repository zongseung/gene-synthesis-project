"""§A.4 Special Token 추가 — Bllossom-8B tokenizer + <ZH>/</ZH>/<KO>/</KO>.

단일 책임: base tokenizer 를 로드해 4개 special token 을 추가하고 지정 경로에 저장.
확장 결과 vocab size 는 128,260 이며 id 128256~128259 가 순서대로 할당된다 (§A.4 R3.2).

§04a §A.3 Primary: MLP-KTLim/llama-3-Korean-Bllossom-8B (vocab 128,256).

Usage:
    PYTHONHASHSEED=0 .venv/bin/python src/data/builder/tokenizer_extend.py \\
        --base MLP-KTLim/llama-3-Korean-Bllossom-8B \\
        --output data/tokenizer/hanmed_bllossom_ext

Exit (§11.2 W1):
    - data/tokenizer/hanmed_bllossom_ext/tokenizer.json 존재
    - tok.encode("<ZH>") == [128256]
    - vocab size == 128,260
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

# 프로젝트 루트 기준 import 를 위해 sys.path 조정
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.seed import set_global_seed  # noqa: E402


SPECIAL_TOKENS = ["<ZH>", "</ZH>", "<KO>", "</KO>"]
EXPECTED_START_ID = 128256  # Bllossom Llama-3 BPE 예약 slot 바로 뒤


def extend_tokenizer(base_id: str, output_dir: Path) -> dict:
    """base tokenizer 에 special token 추가 후 output_dir 에 저장.

    Returns:
        manifest dict — 감사 가능한 메타.
    """
    tok = AutoTokenizer.from_pretrained(base_id)
    base_vocab = len(tok)

    added = tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    new_vocab = len(tok)

    if added != len(SPECIAL_TOKENS):
        raise RuntimeError(
            f"Expected to add {len(SPECIAL_TOKENS)} tokens, "
            f"but add_special_tokens returned {added}. "
            f"Likely already present in base tokenizer."
        )

    # id 할당 검증 — §A.4 는 128256~128259 를 명시
    for i, sp in enumerate(SPECIAL_TOKENS):
        ids = tok.encode(sp, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(
                f"{sp!r} not tokenized as single token: {ids}"
            )
        expected = EXPECTED_START_ID + i
        if ids[0] != expected:
            raise RuntimeError(
                f"{sp!r} id drift: expected {expected}, got {ids[0]}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(str(output_dir))

    manifest = {
        "build_date": datetime.now(timezone.utc).isoformat(),
        "base_tokenizer": base_id,
        "output_dir": str(output_dir),
        "special_tokens_added": SPECIAL_TOKENS,
        "added_count": added,
        "vocab_size_before": base_vocab,
        "vocab_size_after": new_vocab,
        "special_token_ids": {
            sp: tok.encode(sp, add_special_tokens=False)[0]
            for sp in SPECIAL_TOKENS
        },
    }
    manifest_path = output_dir / "hanmed_ext_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--base",
        default="MLP-KTLim/llama-3-Korean-Bllossom-8B",
        help="base tokenizer HF id (§A.3 primary)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("data/tokenizer/hanmed_bllossom_ext"),
        help="extended tokenizer 저장 경로",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_global_seed(args.seed)

    print(f"[tokenizer_extend] base={args.base}")
    print(f"[tokenizer_extend] output={args.output}")

    manifest = extend_tokenizer(args.base, args.output)

    print(
        f"[tokenizer_extend] OK — vocab {manifest['vocab_size_before']} "
        f"→ {manifest['vocab_size_after']} (added {manifest['added_count']})"
    )
    for sp, tid in manifest["special_token_ids"].items():
        print(f"    {sp!r:10s} → id {tid}")
    print(f"[tokenizer_extend] manifest: {args.output / 'hanmed_ext_manifest.json'}")


if __name__ == "__main__":
    main()
