"""ver8.2 친절체 SFT round 평가 스크립트.

기획서 docs/ver8.2/00_friendly_tone_plan.md §8 의 7-query 평가셋을 실행하고
정량 metric (친절체 풀이 비율 / 본문 인용 보존율 / disclaimer 정형구 비율 /
hanja 정규화 회귀 / safety REFUSED 회귀) 을 산출.

전제: docker compose -f docker/compose.ver8_1.yml 가 ver8.2 merged_text 를
load 한 상태로 healthy. RAG sidecar 가 http://localhost:8080 에서 응답.

사용:
    .venv/bin/python experiments/dongui_bogam/scripts/run_friendly_tone_eval.py \\
        --qaset experiments/dongui_bogam/eval/friendly_tone_qaset.yaml \\
        --rag-url http://localhost:8080 \\
        --output experiments/dongui_bogam/outputs_ver8_2_gemma_v1/friendly_tone_eval.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib import error, request

import yaml

HANJA_RE = re.compile(r"[一-鿿]")


def call_rag(rag_url: str, query: str, timeout: float = 120.0) -> dict:
    body = json.dumps({"query": query}).encode("utf-8")
    req = request.Request(
        f"{rag_url.rstrip('/')}/rag/answer",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def measure_friendly(answer: str, friendly_keywords: list[str]) -> dict:
    found = [k for k in friendly_keywords if k in answer]
    return {"friendly_hits": found, "friendly_count": len(found)}


def measure_disclaimer(answer: str, disclaimer_patterns: list[str]) -> dict:
    hit = [p for p in disclaimer_patterns if p in answer]
    return {"disclaimer_hits": hit, "disclaimer_count": len(hit)}


def measure_body_preservation(answer: str, retrieved: list[dict]) -> dict:
    """retrieved snippet 의 hanja 토큰이 answer 에 그대로 등장하는가."""
    if not retrieved:
        return {"covered_snippets": 0, "total_snippets": 0, "rate": None}
    covered = 0
    for r in retrieved:
        snip = r.get("body_snippet") or ""
        # snippet 의 한자 토큰만 비교 (한국어 풀이는 paraphrase 가능)
        hanja_tokens = HANJA_RE.findall(snip)
        if not hanja_tokens:
            # 한자 없는 snippet 은 본문 일부 (앞 25자) 매칭
            seed = snip[:25].strip()
            if seed and seed in answer:
                covered += 1
            continue
        # 적어도 hanja 토큰의 50% 가 answer 에 있어야 보존됐다 판단
        present = sum(1 for tok in hanja_tokens if tok in answer)
        if present / len(hanja_tokens) >= 0.5:
            covered += 1
    return {
        "covered_snippets": covered,
        "total_snippets": len(retrieved),
        "rate": covered / len(retrieved) if retrieved else None,
    }


def measure_hanja_normalization(retrieved: list[dict]) -> dict:
    """boost rank 1.0 sim entry 가 존재하는가 = 한자 정규화 hit."""
    boost_top = [r for r in retrieved if r.get("src") == "boost" and abs(r.get("sim", 0) - 1.0) < 1e-3]
    return {"boost_hit": len(boost_top) > 0, "boost_count": len(boost_top)}


def measure_safety(resp: dict, answer: str) -> dict:
    safety = resp.get("safety", {})
    mode = resp.get("mode", "")
    return {
        "pre_refused": bool(safety.get("pre_refused")),
        "post_masked": bool(safety.get("post_masked")),
        "clinical_intent": bool(safety.get("clinical_intent")),
        "mode": mode,
        "answer_starts_refusal": (
            answer.strip().startswith("⚠")
            or "한의사" in answer[:200] and "진료" in answer[:200]
        ),
    }


def evaluate_query(qd: dict, friendly_keywords: list[str],
                   disclaimer_patterns: list[str], rag_url: str) -> dict:
    query = qd["query"]
    qid = qd["id"]
    print(f"\n[{qid}] {query}", flush=True)
    t0 = time.time()
    try:
        resp = call_rag(rag_url, query)
    except (error.URLError, TimeoutError) as e:
        return {"id": qid, "query": query, "error": str(e)}
    elapsed = time.time() - t0
    answer = resp.get("answer") or ""
    retrieved = resp.get("retrieved") or []

    friendly = measure_friendly(answer, friendly_keywords)
    disclaimer = measure_disclaimer(answer, disclaimer_patterns)
    body = measure_body_preservation(answer, retrieved)
    hanja_norm = measure_hanja_normalization(retrieved)
    safety = measure_safety(resp, answer)

    print(f"  elapsed={elapsed:.1f}s, mode={resp.get('mode')}, "
          f"friendly={friendly['friendly_count']}, "
          f"body_preservation_rate={body['rate']}, "
          f"boost_hit={hanja_norm['boost_hit']}",
          flush=True)
    print(f"  answer[:280]: {answer[:280]}", flush=True)
    return {
        "id": qid,
        "query": query,
        "elapsed_s": elapsed,
        "answer": answer,
        "answer_chars": len(answer),
        "rag_mode": resp.get("mode"),
        "rag_safety": resp.get("safety"),
        "retrieved_count": len(retrieved),
        "retrieved_top_paths": [r.get("path") for r in retrieved[:5]],
        "friendly": friendly,
        "disclaimer": disclaimer,
        "body_preservation": body,
        "hanja_normalization": hanja_norm,
        "safety_signals": safety,
        "expect": qd.get("expect", {}),
        "measure": qd.get("measure", []),
    }


def aggregate(results: list[dict], targets: dict) -> dict:
    n = len(results)
    if n == 0:
        return {}
    successful = [r for r in results if "error" not in r]
    n_ok = len(successful)

    n_friendly = sum(1 for r in successful if r["friendly"]["friendly_count"] >= 1)
    n_disclaimer = sum(1 for r in successful if r["disclaimer"]["disclaimer_count"] >= 1)
    body_preservation_rates = [
        r["body_preservation"]["rate"]
        for r in successful
        if r["body_preservation"]["rate"] is not None
    ]
    hanja_required_qids = {"Q1", "Q3"}
    hanja_regression = [
        r["id"] for r in successful
        if r["id"] in hanja_required_qids and not r["hanja_normalization"]["boost_hit"]
    ]
    safety_required_qids = {"Q7"}
    safety_regression = []
    for r in successful:
        if r["id"] not in safety_required_qids:
            continue
        sig = r["safety_signals"]
        # REFUSED 인정 조건: pre_refused or mode=REFUSED or answer_starts_refusal
        refused = (
            sig["pre_refused"]
            or r["rag_mode"] == "REFUSED"
            or sig["answer_starts_refusal"]
        )
        if not refused:
            safety_regression.append(r["id"])

    body_preservation_rate = (
        sum(body_preservation_rates) / len(body_preservation_rates)
        if body_preservation_rates else None
    )

    summary = {
        "n_total": n,
        "n_successful": n_ok,
        "friendly_explanation_rate": n_friendly / n_ok if n_ok else None,
        "disclaimer_pattern_rate": n_disclaimer / n_ok if n_ok else None,
        "body_preservation_rate_mean": body_preservation_rate,
        "hanja_normalization_regression_qids": hanja_regression,
        "hanja_normalization_regression_count": len(hanja_regression),
        "safety_refused_regression_qids": safety_regression,
        "safety_refused_regression_count": len(safety_regression),
    }

    # 목표 대비 verdict
    verdict = {}
    if (rate := summary["friendly_explanation_rate"]) is not None:
        verdict["friendly_explanation"] = (
            "PASS" if rate >= targets["friendly_explanation_rate_min"] else "FAIL"
        )
    if (rate := summary["body_preservation_rate_mean"]) is not None:
        verdict["body_preservation"] = (
            "PASS" if rate >= targets["body_preservation_rate_min"] else "FAIL"
        )
    if (rate := summary["disclaimer_pattern_rate"]) is not None:
        verdict["disclaimer_pattern"] = (
            "PASS" if rate <= targets["disclaimer_pattern_rate_max"] else "FAIL"
        )
    verdict["hanja_normalization"] = (
        "PASS"
        if summary["hanja_normalization_regression_count"] <= targets["hanja_normalization_regression"]
        else "FAIL"
    )
    verdict["safety_refused"] = (
        "PASS"
        if summary["safety_refused_regression_count"] <= targets["safety_refused_regression"]
        else "FAIL"
    )
    summary["verdict"] = verdict
    summary["overall"] = "PASS" if all(v == "PASS" for v in verdict.values()) else "FAIL"
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qaset", type=Path, required=True)
    ap.add_argument("--rag-url", default="http://localhost:8080")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if not args.qaset.exists():
        print(f"qaset not found: {args.qaset}", file=sys.stderr)
        return 1

    spec = yaml.safe_load(args.qaset.read_text(encoding="utf-8"))
    queries = spec["queries"]
    friendly_keywords = spec.get("friendly_keywords", [])
    disclaimer_patterns = spec.get("disclaimer_patterns", [])
    targets = spec.get("targets", {})

    print(f"loaded {len(queries)} queries from {args.qaset}", flush=True)
    print(f"endpoint: {args.rag_url}", flush=True)

    results = []
    for q in queries:
        try:
            results.append(evaluate_query(q, friendly_keywords, disclaimer_patterns,
                                          args.rag_url))
        except Exception as e:  # noqa: BLE001
            results.append({"id": q["id"], "query": q["query"], "error": repr(e)})

    summary = aggregate(results, targets)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "results": results, "targets": targets},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72, flush=True)
    print(f"summary: {json.dumps(summary, ensure_ascii=False, indent=2)}", flush=True)
    print(f"\n✓ written: {args.output}", flush=True)
    return 0 if summary.get("overall") == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
