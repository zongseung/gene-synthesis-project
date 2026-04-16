"""Packed jsonl 실측 검증 — Stage 2 결과가 의미 있게 파싱됐는지 확인.

7가지 체크:
    C1. 모든 sequence len == 2048
    C2. pad_token_id 분포 / EOS id 분포 / BOS id 분포
    C3. special token <ZH>=128256, </ZH>=128257, <KO>=128258, </KO>=128259 빈도
    C4. Llama-3 chat token (128000, 128009) 의도치 않은 등장 여부
    C5. 1개 sequence 를 decode 해서 원문과 round-trip 일치하는지
    C6. bilingual 에서 ZH-KO 짝 매칭 확인 (128256 개수 ≈ 128257, 128258 ≈ 128259)
    C7. Block 경계 `\\n` 보존 확인 (decode 시 `</ZH>\\n<KO>` 패턴)
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


def verify(packed_path: Path, clean_path: Path, tokenizer_dir: Path, n_show: int = 1):
    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    pad_id = tok.pad_token_id
    eos_id = tok.eos_token_id
    bos_id = tok.bos_token_id
    print(f"[{packed_path.name}]")
    print(f"  pad_id={pad_id}  eos_id={eos_id}  bos_id={bos_id}")

    # 특수 id
    ZH, ZH_CL, KO, KO_CL = 128256, 128257, 128258, 128259
    CHAT_EOT = 128009  # Llama-3 <|eot_id|>
    CHAT_BEGIN = 128000  # <|begin_of_text|>

    # ---- (C1~C4) 전체 스캔 ----
    counts = Counter()
    n_seqs = 0
    len_set = set()
    with open(packed_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ids = rec["input_ids"]
            len_set.add(len(ids))
            n_seqs += 1
            for i in ids:
                counts[i] += 1

    assert len_set == {2048}, f"seq_len drift: {len_set}"
    print(f"  C1 seq_len invariant: ALL 2048  ({n_seqs} seqs)  OK")
    print(f"  C2 pad token id {pad_id} count: {counts[pad_id]:,} ({counts[pad_id]/(n_seqs*2048)*100:.1f}%)")
    print(f"  C2 eos token id {eos_id} count: {counts[eos_id]:,}")
    print(f"  C2 bos token id {bos_id} count: {counts[bos_id]:,}")
    print(f"  C3 <ZH>=128256: {counts[ZH]:,}  </ZH>=128257: {counts[ZH_CL]:,}")
    print(f"  C3 <KO>=128258: {counts[KO]:,}  </KO>=128259: {counts[KO_CL]:,}")
    print(f"  C4 <|begin_of_text|>=128000: {counts[CHAT_BEGIN]:,}  <|eot_id|>=128009: {counts[CHAT_EOT]:,}")

    # C6 짝 매칭
    if counts[ZH] > 0:
        zh_pair_ratio = counts[ZH_CL] / counts[ZH]
        ko_pair_str = f"{counts[KO_CL]/counts[KO]:.3f}" if counts[KO] > 0 else "N/A"
        print(f"  C6 </ZH>/<ZH> ratio = {zh_pair_ratio:.3f}  </KO>/<KO> = {ko_pair_str}")

    # ---- (C5/C7) 첫 sequence round-trip decode ----
    print(f"\n  --- C5/C7: first {n_show} sequence(s) round-trip ---")
    with open(packed_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n_show:
                break
            rec = json.loads(line)
            ids = rec["input_ids"]
            # pad 제거 후 decode
            ids_no_pad = [t for t in ids if t != pad_id]
            decoded = tok.decode(ids_no_pad, skip_special_tokens=False)
            # 앞 300자만
            preview = decoded[:500].replace("\n", "\\n")
            print(f"  seq[{i}] non-pad tokens={len(ids_no_pad)}/{len(ids)}")
            print(f"  seq[{i}] decoded[:500]={preview!r}")
            # 블록 경계 보존?
            has_zh_nl_ko = "</ZH>\n<KO>" in decoded or "</ZH>" in decoded and "<KO>" in decoded
            print(f"  seq[{i}] </ZH>\\n<KO> pattern present: {'</ZH>\\n<KO>' in decoded}")
            # 원문과 비교: clean.jsonl 앞 N 개 record 의 text 를 연결하여 포함되는지
            ids_head = ids_no_pad[:50]
            head_decoded = tok.decode(ids_head, skip_special_tokens=False)
            print(f"  seq[{i}] head[:50 tokens] decode={head_decoded!r}")

    # ---- (C5 추가) clean.jsonl 첫 record 이 packed 안에 있는지 ----
    print(f"\n  --- C5b: clean 첫 record 원문이 packed 첫 seq 에 포함되는지 ---")
    with open(clean_path, encoding="utf-8") as f:
        first_clean = json.loads(f.readline())["text"]
    with open(packed_path, encoding="utf-8") as f:
        first_packed = json.loads(f.readline())["input_ids"]
    ids_no_pad = [t for t in first_packed if t != pad_id]
    decoded_full = tok.decode(ids_no_pad, skip_special_tokens=False)
    # 원문을 찾을 수 있는가
    clean_head = first_clean[:60]
    found = clean_head in decoded_full
    print(f"  clean[0] text[:60] = {clean_head!r}")
    print(f"  packed[0] 안에 포함? {found}")
    if not found:
        # partial: 첫 30자
        found2 = first_clean[:30] in decoded_full
        print(f"  clean[0] text[:30] 포함? {found2}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=Path, default=Path("data/tokenizer/hanmed_bllossom_ext"))
    ap.add_argument("--processed-dir", type=Path, default=Path("data/cpt_processed"))
    args = ap.parse_args()

    for corpus in ["hanmed_bilingual", "hanmed_zh_only", "hanmed_ko_only"]:
        packed = args.processed_dir / f"{corpus}_packed_2048.jsonl"
        clean = args.processed_dir / f"{corpus}_clean.jsonl"
        verify(packed, clean, args.tokenizer)
        print()


if __name__ == "__main__":
    main()
