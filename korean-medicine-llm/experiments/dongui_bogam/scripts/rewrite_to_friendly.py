"""ver8.2 § 4.3 — base Gemma3-12B-IT 로 ver8.1 답변을 친절체 rewrite.

base 만 사용 (ver8.1 LoRA 미적용) — base 의 친절체 표현력 활용.
gold seeds 의 (원답변, 친절답변) 페어를 in-context few-shot 로.

사용:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python \\
      experiments/dongui_bogam/scripts/rewrite_to_friendly.py \\
      --input data/sft/v8_1_rewrite_sample.jsonl \\
      --gold  data/sft/friendly_gold_v0.jsonl \\
      --output data/sft/friendly_rewrite_v0.jsonl \\
      --resume

자원: GPU 1장 (24 GB+), 10K rows ≈ 5 시간 (greedy 0.3, max 600).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]


REWRITE_SYSTEM = """당신은 한의학 고전 문헌의 답변을 친절한 어투로 풀어쓰는 작업을 합니다.

규칙:
1. 원답변에 포함된 발췌 (한문/한자체 본문) 는 한 글자도 바꾸지 말고 그대로 인용하세요.
   - 입력의 [원문 발췌] 블록은 답변에 반드시 그대로 포함하세요.
2. 발췌 앞에는 1-2 문장 자연스러운 도입을 한국어로 추가하세요.
3. 발췌 뒤에는 2-3 문장의 풀이를 자연스러운 현대 한국어 어투로 추가하세요.
   - "쉽게 말하면", "즉", "이는" 같은 자연스러운 연결어를 사용하세요.
   - 한문 직역체 ("오장의 양을 보한다") 는 일상적 한국어 ("오장의 양기를 보충합니다") 로 풀어쓰세요.
4. 발췌에 없는 약재명·처방명·수치·인용서는 절대 만들지 마세요. 사실 추가 금지.
5. 정형구 disclaimer ("...로 한정해 읽어 주십시오") 는 자연스러운 안내로 바꾸세요.
   예: "본 답변은 동의보감 본문 검색 결과이며 실제 복용은 한의사 상담이 필요합니다."
6. [N] 같은 출처 표시는 그대로 유지하세요.
7. 학술적 정확성과 친절한 어투를 동시에 유지하세요.
"""


EXCERPT_RE = re.compile(r"원문 발췌:\s*(.+)\s*$", re.S)


def extract_source_excerpt(question: str) -> str:
    """SFT question 의 `원문 발췌:` 이하 원문을 추출한다."""
    m = EXCERPT_RE.search(question or "")
    if not m:
        return ""
    return m.group(1).strip()


def load_few_shots(gold_path: Path, n: int = 3) -> list[dict]:
    """gold seeds 에서 few-shot example n개 추출 (길이 적당한 것 우선)."""
    gold = [json.loads(l) for l in gold_path.open()]
    # _source_assistant 필드 있는 것만 (rewrite 변환 example 로 적합)
    gold_with_source = [g for g in gold
                        if g.get("_source_assistant") or g.get("source_assistant")]
    if not gold_with_source:
        # source 없으면 그냥 짧은 것 우선
        gold_with_source = gold
    gold_with_source.sort(key=lambda r: abs(len(r.get("assistant", "")) - 600))
    return gold_with_source[:n]


def format_rewrite_input(question: str, answer: str, excerpt: str) -> str:
    parts = []
    if question:
        parts.append(f"[질문]\n{question}")
    if excerpt:
        parts.append(f"[원문 발췌]\n{excerpt}")
    parts.append(f"[원답변]\n{answer}")
    return "\n\n".join(parts)


def build_messages(
    few_shots: list[dict],
    original_question: str,
    original_answer: str,
    source_excerpt: str,
) -> list[dict]:
    msgs = [{"role": "system", "content": REWRITE_SYSTEM}]
    for fs in few_shots:
        src = fs.get("_source_assistant") or fs.get("source_assistant") or ""
        src_question = fs.get("_source_question") or fs.get("question") or ""
        src_excerpt = (
            fs.get("_source_excerpt")
            or fs.get("source_excerpt")
            or extract_source_excerpt(src_question)
        )
        if not src:
            src = fs.get("assistant", "")
        msgs.append({
            "role": "user",
            "content": (
                "다음 답변을 친절체로 풀어써 주세요:\n\n"
                f"{format_rewrite_input(src_question, src, src_excerpt)}"
            ),
        })
        msgs.append({"role": "assistant", "content": fs.get("assistant", "")})
    msgs.append({
        "role": "user",
        "content": (
            "다음 답변을 친절체로 풀어써 주세요:\n\n"
            f"{format_rewrite_input(original_question, original_answer, source_excerpt)}"
        ),
    })
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True,
                    help="ver8.1 rewrite sample jsonl")
    ap.add_argument("--gold", type=Path, required=True,
                    help="gold seeds (few-shot 용, _source_assistant 필드 권장)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--base", type=Path,
                    default=ROOT / "models" / "gemma-3-12b-it")
    ap.add_argument("--device", default="cuda",
                    help="cuda / cuda:0 / cpu")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16"])
    ap.add_argument("--temperature", type=float, default=0.3,
                    help="0.3 권장 (변형 약간 + 환각 억제)")
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--few-shot", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None,
                    help="처음 N rows 만 (디버그)")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="generation batch size")
    ap.add_argument("--resume", action="store_true",
                    help="output 기존 있으면 이어쓰기 (id 중복 skip)")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"input 없음: {args.input}")
    if not args.gold.exists():
        raise SystemExit(f"gold 없음: {args.gold}")

    print(f"[load] base = {args.base}")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    tok = AutoTokenizer.from_pretrained(str(args.base))
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.base), dtype=dtype, device_map=args.device
    ).eval()

    few_shots = load_few_shots(args.gold, n=args.few_shot)
    print(f"[few-shot] {len(few_shots)} examples loaded from {args.gold.name}")
    if len(few_shots) < args.few_shot:
        print(f"  [warn] requested {args.few_shot} but only {len(few_shots)} available")

    rows_in = [json.loads(l) for l in args.input.open()]
    if args.limit:
        rows_in = rows_in[:args.limit]
    print(f"[input] {args.input.name}: {len(rows_in)} rows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done_ids: set[str] = set()
    if args.resume and args.output.exists():
        for line in args.output.open():
            done_ids.add(json.loads(line)["id"])
        print(f"[resume] skipping {len(done_ids)} already-rewritten rows")

    mode = "a" if args.resume and args.output.exists() else "w"
    out = args.output.open(mode)

    t0 = time.time()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    pending_rows = [
        row for row in rows_in
        if row.get("id") not in done_ids and row.get("assistant")
    ]

    n_done_this_run = 0
    for start in range(0, len(pending_rows), max(1, args.batch_size)):
        batch_rows = pending_rows[start:start + max(1, args.batch_size)]
        prompts: list[str] = []
        source_meta: list[tuple[str, str]] = []
        for row in batch_rows:
            original = row.get("assistant", "")
            original_question = row.get("question", "")
            source_excerpt = extract_source_excerpt(original_question)
            msgs = build_messages(few_shots, original_question, original, source_excerpt)
            prompts.append(tok.apply_chat_template(msgs, tokenize=False,
                                                   add_generation_prompt=True))
            source_meta.append((original_question, source_excerpt))

        ids = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(
                **ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                pad_token_id=tok.pad_token_id,
            )
        prompt_len = ids.input_ids.shape[-1]
        for row, (original_question, source_excerpt), output_ids in zip(batch_rows, source_meta, outputs):
            original = row.get("assistant", "")
            rewritten = tok.decode(output_ids[prompt_len:], skip_special_tokens=True).strip()

            new_row = dict(row)
            new_row["_source_question"] = original_question
            if source_excerpt:
                new_row["_source_excerpt"] = source_excerpt
            new_row["_source_assistant"] = original
            new_row["assistant"] = rewritten
            new_row["tone"] = "friendly"
            new_row["_origin"] = "llm_rewrite_v0"
            # messages 마지막 assistant 도 update (학습 시 reflect)
            if isinstance(new_row.get("messages"), list):
                for m in reversed(new_row["messages"]):
                    if m.get("role") == "assistant":
                        m["content"] = rewritten
                        break
            out.write(json.dumps(new_row, ensure_ascii=False) + "\n")
            n_done_this_run += 1
        out.flush()

        if n_done_this_run % args.log_every == 0:
            elapsed = time.time() - t0
            rate = n_done_this_run / max(elapsed, 1)
            remaining = (len(pending_rows) - n_done_this_run)
            eta_min = remaining / max(rate * 60, 0.001)
            print(f"  [{n_done_this_run}/{len(pending_rows)}] "
                  f"rate={rate:.2f}/s  ETA={eta_min:.0f} min")

    out.close()
    elapsed = time.time() - t0
    print(f"\n✓ done in {elapsed/60:.1f} min, {n_done_this_run} rows rewritten")
    print(f"✓ output: {args.output}")
    print(f"  next: validate_friendly_rewrite.py")


if __name__ == "__main__":
    main()
