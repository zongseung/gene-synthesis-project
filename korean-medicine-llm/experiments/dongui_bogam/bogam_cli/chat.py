#!/usr/bin/env python
"""HanMed ver8.1 RAG REPL — 8080 sidecar 통합 chat 인터페이스.

기존 `hanmed chat` 은 vLLM 만 직접 호출 (RAG 미통합). 이 스크립트는 RAG sidecar
의 `/rag/answer` 를 호출 → retrieval(한자 정규화·boost·dense) + safety(pre/post)
+ LLM 을 한 번에 거치는 통합 REPL.

사용:
  # one-shot
  .venv/bin/python experiments/dongui_bogam/cli/chat_rag.py \\
      -q "인삼(人蔘)의 성미와 귀경에 대해 간단히 설명해줘."

  # REPL (인터랙티브)
  .venv/bin/python experiments/dongui_bogam/cli/chat_rag.py

  # alias 등록 (~/.bashrc 등)
  alias hanmed-rag='.venv/bin/python experiments/dongui_bogam/cli/chat_rag.py'
  hanmed-rag -q "사물탕의 구성 약재"
  hanmed-rag --show-retrieved -q "..."

옵션:
  --endpoint URL       RAG sidecar URL (default: http://localhost:8080)
  -q / --query TEXT    one-shot mode (없으면 REPL)
  -k INT               retrieval top-k (default: 5)
  --show-retrieved     발췌 path 출력
  --json               raw JSON 출력 (jq 파이프 용)
"""
from __future__ import annotations

import argparse
import json
import sys
from importlib.resources import files
from typing import Any

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

console = Console()
err_console = Console(stderr=True, style="bold red")


# ── splash 자산 (root hanmed 와 동일한 24-bit ANSI 거북 마스코트) ─────────
CLI_TITLE = "HanMed · 동의보감 ver8.1 (RAG)"
CLI_SUBTITLE = "Dongui Bogam assistant · Gemma3-12B + ver8.1 LoRA · sidecar"
CLI_WELCOME = "동의보감 전용 · 진단·처방 불가  ·  /retrieved /quit /exit"
_DIVIDER = "─" * 70


def _load_mascot() -> tuple[str, ...]:
    """패키지 안 turtle_24col.ansi 를 줄 단위로 로드. 없으면 빈 tuple."""
    try:
        raw = files("bogam_cli").joinpath("turtle_24col.ansi").read_text(
            encoding="utf-8"
        ).rstrip("\n")
    except (FileNotFoundError, ModuleNotFoundError):
        return tuple()
    return tuple(raw.split("\n")) if raw else tuple()


MASCOT_LINES = _load_mascot()


def _print_splash(endpoint: str, k: int, show_retrieved: bool,
                  health: dict[str, Any]) -> None:
    """거북이 + 헤더 3줄 + divider + status — root hanmed 톤과 일관."""
    # 거북이 마스코트 (24-bit ANSI 그대로 — rich 거치지 않음)
    if MASCOT_LINES:
        sys.stdout.write("\n")
        for line in MASCOT_LINES:
            sys.stdout.write("  " + line + "\n")
        sys.stdout.write("\n")
        sys.stdout.flush()

    # 3-line header
    console.print(f"  [bold white]{CLI_TITLE}[/]")
    console.print(f"  [grey70]{CLI_SUBTITLE}[/]")
    console.print(f"  [dark_orange3]{CLI_WELCOME}[/]")
    console.print(f"  [grey58]{_DIVIDER}[/]")

    # status 라인 (endpoint · upstream)
    rag_color = "green" if health.get("rag") == "ok" else "red"
    vllm_color = "green" if health.get("vllm") == "ok" else "red"
    console.print(
        f"  [grey58]┌─ hanmed-bogam ─────[/] "
        f"endpoint=[cyan]{endpoint}[/] · k=[cyan]{k}[/] · "
        f"rag=[{rag_color}]{health.get('rag')}[/] · "
        f"vllm=[{vllm_color}]{health.get('vllm')}[/]"
    )
    if show_retrieved:
        console.print("  [grey58]│[/] [dim]show_retrieved=on[/]")


def _post_answer(endpoint: str, query: str, k: int) -> dict[str, Any]:
    try:
        r = httpx.post(
            f"{endpoint}/rag/answer",
            json={"query": query, "k": k},
            timeout=120,
        )
    except httpx.HTTPError as e:
        err_console.print(f"[err] connect {endpoint}: {e}")
        sys.exit(2)
    if r.status_code != 200:
        err_console.print(f"[err] {r.status_code}: {r.text[:300]}")
        sys.exit(2)
    return r.json()


def _print_response(d: dict[str, Any], show_retrieved: bool) -> None:
    if d["mode"] == "REFUSED":
        console.print(Panel(
            d["answer"],
            title="[bold red]⚠ STRICT 거부[/]",
            border_style="red",
        ))
        return

    if show_retrieved:
        t = Table(title=f"retrieved (extracted_names: {d['extracted_names']})",
                  show_lines=False)
        t.add_column("rank", justify="right", style="dim", width=4)
        t.add_column("src", width=6)
        t.add_column("sim", justify="right", width=6)
        t.add_column("path")
        t.add_column("body", overflow="ellipsis", max_width=60)
        for r in d["retrieved"]:
            t.add_row(
                str(r["rank"]),
                r["src"],
                f"{r['sim']:.3f}",
                r.get("path") or "(no path)",
                r.get("body_snippet", ""),
            )
        console.print(t)
        console.print(Rule(style="dim"))

    console.print(d["answer"])

    if d["safety"]["post_masked"]:
        console.print("\n[yellow]※ post_check 마스킹 적용[/]")

    foot = (f"\n[dim]mode={d['mode']} · elapsed={d['elapsed_ms']}ms · "
            f"prompt_tokens={d.get('prompt_tokens')}[/]")
    console.print(foot)


def _ask_one(endpoint: str, query: str, k: int,
             show_retrieved: bool, raw_json: bool) -> None:
    d = _post_answer(endpoint, query, k)
    if raw_json:
        print(json.dumps(d, ensure_ascii=False, indent=2))
    else:
        _print_response(d, show_retrieved)


def _repl(endpoint: str, k: int, show_retrieved: bool) -> None:
    # 시작 헬스체크
    try:
        h = httpx.get(f"{endpoint}/health", timeout=5).json()
    except Exception as e:
        err_console.print(f"[err] /health unreachable: {e}")
        sys.exit(2)

    _print_splash(endpoint, k, show_retrieved, h)

    while True:
        try:
            q = console.input("\n[bold cyan]>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not q:
            continue
        if q in ("/quit", "/q", "exit"):
            break
        if q == "/retrieved":
            show_retrieved = not show_retrieved
            console.print(f"[dim]show_retrieved={show_retrieved}[/]")
            continue
        d = _post_answer(endpoint, q, k)
        _print_response(d, show_retrieved)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="HanMed ver8.1 RAG REPL (sidecar 통합)"
    )
    ap.add_argument("--endpoint", default="http://localhost:8080",
                    help="RAG sidecar URL")
    ap.add_argument("-q", "--query", default=None,
                    help="one-shot 질의 (없으면 REPL)")
    ap.add_argument("-k", type=int, default=5, help="retrieval top-k")
    ap.add_argument("--show-retrieved", action="store_true",
                    help="발췌 path 출력")
    ap.add_argument("--json", dest="raw_json", action="store_true",
                    help="raw JSON 출력 (jq 파이프 용)")
    args = ap.parse_args()

    if args.query:
        _ask_one(args.endpoint, args.query, args.k,
                 args.show_retrieved, args.raw_json)
    else:
        _repl(args.endpoint, args.k, args.show_retrieved)


if __name__ == "__main__":
    main()
