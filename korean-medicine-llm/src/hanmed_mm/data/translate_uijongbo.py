"""의방유취(book_056) 한문 원문 → 한국어 번역 (GPT API).

번역 환각 최소화 원칙:
- temperature 0, 직역. 원문에 없는 내용 추가/추론 금지.
- 출전 표기(《》), 약재·혈자리 한자어는 보존(한글 병기 허용).
- 불확실한 결자/오자는 [?]로만 표기, 임의 보완 금지.
- 1 레코드(original) → 1 한국어 문장열. trans_ko 기존값 있으면 건너뜀.

resume: out jsonl 에 (volume_id, content_seq) 기록 → 재실행 시 done 스킵.
concurrency: ThreadPoolExecutor. 증분 기록(중간 중단 안전).

사용:
  PYTHONPATH=src .venv/bin/python -m hanmed_mm.data.translate_uijongbo \
      --raw_dir ver1/data/raw/mediclassics_unified/book_056 \
      --out data/cpt/uijongbo_ko.jsonl --model gpt-4o-mini --workers 8 [--limit 20]
"""
from __future__ import annotations
import argparse, glob, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

SYS_PROMPT = (
    "당신은 한의학 고전 한문(漢文)을 현대 한국어로 번역하는 전문 번역가입니다. "
    "출력은 반드시 **한국어(한글) 문장**이어야 합니다. 절대 중국어(简体/繁體)나 한문 원문 그대로를 출력하지 마세요.\n"
    "원칙:\n"
    "1) 원문에 있는 내용만 직역합니다. 원문에 없는 설명·해석·추론을 덧붙이지 않습니다.\n"
    "2) 출전 표기(《》)와 인명·서명은 원문 한자대로 보존합니다.\n"
    "3) 약재명·혈자리·처방명 등 한의학 용어는 한국 한자음(한글)으로 옮기고, 필요시 한자를 괄호 병기합니다. 예: 素問→소문(素問).\n"
    "4) 글자가 깨졌거나 불확실하면 임의로 보완하지 말고 [?]로 표기합니다.\n"
    "5) 번역문(한국어)만 출력합니다. 머리말·해설·메타설명을 붙이지 않습니다."
)


def _hangul_ratio(s):
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    han = sum(1 for c in letters if "가" <= c <= "힣")
    return han / len(letters)


def load_records(raw_dir, min_chars=10):
    """번역 대상: 한문 있고 한국어 없음 + min_chars 이상(짧은 제목/편명 제외)."""
    recs = []
    for f in sorted(glob.glob(os.path.join(raw_dir, "vol_*.jsonl"))):
        for l in open(f):
            d = json.loads(l)
            zh = (d.get("original") or "").strip()
            ko = (d.get("trans_ko") or "").strip()
            if zh and not ko and len(zh) >= min_chars:
                recs.append({"volume_id": d.get("volume_id"), "content_seq": d.get("content_seq"),
                             "up_path_nm": d.get("up_path_nm"), "original": zh})
    return recs


def load_done(out_path):
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                d = json.loads(l); done.add((d["volume_id"], d["content_seq"]))
            except Exception:
                pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="테스트용 N건만")
    ap.add_argument("--env", default="/home/user/gene-synthesis-project/.env")
    args = ap.parse_args()

    # API 키 로드 (.env 의 API_key)
    key = None
    for line in open(args.env):
        if line.strip().startswith("API_key"):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit("API_key 없음 (.env)")

    from openai import OpenAI
    client = OpenAI(api_key=key)

    recs = load_records(args.raw_dir)
    done = load_done(args.out)
    todo = [r for r in recs if (r["volume_id"], r["content_seq"]) not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"번역 필요 {len(recs):,} | 완료 {len(done):,} | 이번 실행 {len(todo):,}", flush=True)

    lock = threading.Lock()
    cnt = {"ok": 0, "err": 0, "skip": 0}
    fout = open(args.out, "a")

    def _call(text, strict=False):
        msgs = [{"role": "system", "content": SYS_PROMPT}]
        if strict:
            msgs.append({"role": "system", "content": "직전 출력이 한국어가 아니었습니다. 반드시 한글 문장으로만 번역하세요."})
        msgs.append({"role": "user", "content": text})
        resp = client.chat.completions.create(model=args.model, temperature=0, messages=msgs)
        return resp.choices[0].message.content.strip()

    def work(r):
        try:
            ko = _call(r["original"])
            ratio = _hangul_ratio(ko)
            if ratio < 0.3:  # 중국어/한문으로 나온 경우 → 1회 강제 재시도
                ko2 = _call(r["original"], strict=True)
                if _hangul_ratio(ko2) >= ratio:
                    ko, ratio = ko2, _hangul_ratio(ko2)
            if ratio < 0.3:  # 그래도 한국어 아니면 제외(쓰레기 기록 금지)
                with lock:
                    cnt["skip"] += 1
                return
            out = {**r, "trans_ko_mt": ko, "mt_model": args.model, "hangul_ratio": round(ratio, 3)}
            with lock:
                fout.write(json.dumps(out, ensure_ascii=False) + "\n"); fout.flush()
                cnt["ok"] += 1
                if cnt["ok"] % 50 == 0:
                    print(f"  진행 {cnt['ok']:,}/{len(todo):,} (err {cnt['err']} skip {cnt['skip']})", flush=True)
        except Exception as e:
            with lock:
                cnt["err"] += 1
                if cnt["err"] <= 5:
                    print(f"  [err] seq{r.get('content_seq')}: {e}", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    fout.close()
    print(f"완료: ok {cnt['ok']:,} / err {cnt['err']} / {time.time()-t0:.0f}s → {args.out}", flush=True)


if __name__ == "__main__":
    main()
