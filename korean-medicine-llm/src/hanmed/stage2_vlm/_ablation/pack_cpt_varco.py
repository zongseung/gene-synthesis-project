"""한의학 한국어 CPT 코퍼스 → VARCO(Qwen3) 토크나이저로 2048 packing.

소스:
- 고전 32권: ver1/data/cpt/hanmed_ko_only.jsonl  (field: text)
- 의방유취 번역: data/cpt/uijongbo_ko.jsonl       (field: trans_ko_mt)

각 문서를 토큰화 → EOS로 구분 → 전부 이어붙여 2048 토큰 단위로 자른다.
출력: {input_ids: [...2048...]} jsonl. CPT trainer가 바로 읽음.

사용:
  PYTHONPATH=src .venv/bin/python -m hanmed.stage2_vlm._ablation.pack_cpt_varco \
      --model models/VARCO-VISION-2.0-14B --seq_len 2048 \
      --out data/cpt_varco/hanmed_ko_packed_varco_2048.jsonl
"""
from __future__ import annotations
import argparse, json, os, time


SOURCES = [
    ("ver1/data/cpt/hanmed_ko_only.jsonl", "text"),
    ("data/cpt/uijongbo_ko.jsonl", "trans_ko_mt"),
    ("data/cpt/taepyeong_ko.jsonl", "trans_ko_mt"),
]


def iter_docs(min_chars=5):
    for path, field in SOURCES:
        if not os.path.exists(path):
            print(f"[skip] {path} 없음"); continue
        cnt = 0
        for l in open(path):
            try:
                d = json.loads(l)
            except Exception:
                continue
            t = (d.get(field) or "").strip()
            if len(t) >= min_chars:
                yield t; cnt += 1
        print(f"  {path}: {cnt:,} 문서")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/VARCO-VISION-2.0-14B")
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_chars", type=int, default=5)
    ap.add_argument("--batch", type=int, default=1000)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    eos = tok.eos_token_id
    print(f"토크나이저 {type(tok).__name__} vocab={tok.vocab_size} eos={eos}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    t0 = time.time()

    # 문서를 배치 토큰화 → 토큰 스트림에 이어붙임(문서 사이 EOS)
    buf = []            # 토큰 누적 버퍼
    n_docs = 0; n_seq = 0; n_tok = 0
    fout = open(args.out, "w")

    def flush_batch(texts):
        nonlocal buf, n_seq, n_tok
        enc = tok(texts, add_special_tokens=False)["input_ids"]
        for ids in enc:
            buf.extend(ids); buf.append(eos)
        # 2048 단위로 잘라 기록
        while len(buf) >= args.seq_len:
            chunk = buf[:args.seq_len]; del buf[:args.seq_len]
            fout.write(json.dumps({"input_ids": chunk}, ensure_ascii=False) + "\n")
            n_seq += 1; n_tok += len(chunk)

    batch = []
    for doc in iter_docs(args.min_chars):
        batch.append(doc); n_docs += 1
        if len(batch) >= args.batch:
            flush_batch(batch); batch = []
            if n_docs % 20000 == 0:
                print(f"  진행 {n_docs:,} 문서 → {n_seq:,} 시퀀스 ({time.time()-t0:.0f}s)", flush=True)
    if batch:
        flush_batch(batch)
    # 남은 버퍼는 패딩 없이 마지막 부분 시퀀스로 기록(seq_len 미만이면 버림: CPT 관행)
    if len(buf) >= args.seq_len // 2:  # 절반 이상이면 살림
        fout.write(json.dumps({"input_ids": buf}, ensure_ascii=False) + "\n"); n_seq += 1; n_tok += len(buf)
    fout.close()

    print(f"\n완료: {n_docs:,} 문서 → {n_seq:,} 시퀀스(x{args.seq_len}) / {n_tok:,} 토큰 / {time.time()-t0:.0f}s")
    print(f"  → {args.out}")


if __name__ == "__main__":
    main()
