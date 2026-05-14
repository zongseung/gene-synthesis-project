#!/usr/bin/env python
"""HanMed 동의보감 음성 자비스 — 마이크로 묻고 음성으로 듣는 RAG REPL.

`bogam_cli.chat` (텍스트 REPL) 의 음성 버전. 두뇌(RAG sidecar)는 그대로 두고
양 끝에 STT·TTS 만 붙인다:

  🎤 record → faster-whisper STT → POST :8080/rag/answer → edge-tts → 🔊 play

실행 모드:
  # 라이브 마이크 REPL  (맥 — sounddevice/PortAudio 필요)
  hanmed-bogam-voice --device cpu

  # 오디오 파일 1회     (서버 검증용 — 마이크 불필요)
  hanmed-bogam-voice --audio sample.wav --no-speak

  # 텍스트 1회          (STT 건너뜀 — RAG·TTS 정제만 검증)
  hanmed-bogam-voice --text "사물탕의 구성 약재" --no-speak

옵션:
  --endpoint URL      RAG sidecar (default: http://localhost:8080)
  --model NAME        faster-whisper 모델 (default: large-v3-turbo)
  --device cuda|cpu   STT 디바이스 (서버: cuda, 맥: cpu)
  --device-index N    cuda 디바이스 번호 (서버: 1 — vLLM 이 GPU0 점유)
  -k INT              retrieval top-k (default: 5)
  --no-speak          TTS 생략 (텍스트 답변만)
  --show-retrieved    발췌 path 표 표시
"""
from __future__ import annotations

import argparse
import math
import sys

import httpx
from rich.text import Text

from bogam_cli.chat import (
    MASCOT_LINES,
    _post_answer,
    _print_response,
    console,
    err_console,
)
from bogam_cli import tts
from bogam_cli.stt import STT, record_until_enter

CLI_TITLE = "HanMed · 동의보감 음성 자비스"
CLI_SUBTITLE = "Voice RAG assistant · faster-whisper + ver8.1 sidecar + edge-tts"
CLI_WELCOME = "동의보감 전용 · 진단·처방 불가  ·  Enter 녹음종료 · Ctrl+C 종료"
_DIVIDER = "─" * 70

_WAVE_CHARS = "▁▂▃▄▅▆▇█"


def _soundwave(width: int, phase: int) -> str:
    """말하는 중 사운드웨이브 — phase 가 커질수록 파형이 흐른다."""
    chars = []
    for i in range(width):
        v = (math.sin(i * 0.6 + phase * 0.7) + 1) / 2  # 0..1
        chars.append(_WAVE_CHARS[round(v * (len(_WAVE_CHARS) - 1))])
    return "".join(chars)


def _print_splash(endpoint: str, stt: STT, health: dict, speak: bool) -> None:
    """거북이 마스코트 + 헤더 — chat.py 톤과 일관."""
    if MASCOT_LINES:
        sys.stdout.write("\n")
        for line in MASCOT_LINES:
            sys.stdout.write("  " + line + "\n")
        sys.stdout.write("\n")
        sys.stdout.flush()

    console.print(f"  [bold white]{CLI_TITLE}[/]")
    console.print(f"  [grey70]{CLI_SUBTITLE}[/]")
    console.print(f"  [dark_orange3]{CLI_WELCOME}[/]")
    console.print(f"  [grey58]{_DIVIDER}[/]")

    rag_color = "green" if health.get("rag") == "ok" else "red"
    vllm_color = "green" if health.get("vllm") == "ok" else "red"
    console.print(
        f"  [grey58]┌─ hanmed-voice ─────[/] "
        f"endpoint=[cyan]{endpoint}[/] · "
        f"stt=[cyan]{stt.model_size}[/] ([cyan]{stt.load_seconds:.1f}s[/]) · "
        f"tts=[cyan]{'on' if speak else 'off'}[/] · "
        f"rag=[{rag_color}]{health.get('rag')}[/] · "
        f"vllm=[{vllm_color}]{health.get('vllm')}[/]"
    )


def _speak(text: str) -> None:
    """answer 정제 → edge-tts 합성 → 사운드웨이브 애니메이션과 함께 재생."""
    spoken = tts.clean_for_speech(text)
    if not spoken:
        return
    mp3 = tts.synthesize(spoken)
    phase = {"n": 0}
    try:
        from rich.live import Live

        with Live(console=console, refresh_per_second=12, transient=True) as live:

            def tick() -> None:
                phase["n"] += 1
                wave = _soundwave(46, phase["n"])
                live.update(Text(f"  🔊  {wave}", style="cyan"))

            tts.play(mp3, on_tick=tick, tick_hz=12)
    except RuntimeError as exc:
        # 서버 등 PortAudio 없는 환경 — 합성까지만 하고 경로를 안내.
        console.print(f"  [yellow]※ 재생 건너뜀[/] ({exc})")
        console.print(f"  [dim]합성된 음성: {mp3}[/]")


def _handle_turn(endpoint: str, query: str, k: int,
                 show_retrieved: bool, speak: bool) -> None:
    """질문 1건 처리 — 화면 출력 + (옵션) 음성 출력."""
    query = query.strip()
    if not query:
        console.print("  [yellow]※ 질문이 비어 있습니다 — 다시 말씀해 주세요.[/]")
        return

    console.print(f"\n  [bold cyan]질문[/] {query}")
    with console.status("[grey70]동의보감을 찾는 중…[/]", spinner="dots"):
        d = _post_answer(endpoint, query, k)

    _print_response(d, show_retrieved)
    if speak:
        _speak(d["answer"])


def _health(endpoint: str) -> dict:
    try:
        return httpx.get(f"{endpoint}/health", timeout=5).json()
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[err] /health unreachable: {exc}")
        sys.exit(2)


def main() -> None:
    ap = argparse.ArgumentParser(description="HanMed 동의보감 음성 자비스")
    ap.add_argument("--endpoint", default="http://localhost:8080",
                    help="RAG sidecar URL")
    ap.add_argument("--model", default="large-v3-turbo",
                    help="faster-whisper 모델")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="STT 디바이스 (서버: cuda, 맥: cpu)")
    ap.add_argument("--device-index", type=int, default=1,
                    help="cuda 디바이스 번호 (서버: 1)")
    ap.add_argument("-k", type=int, default=5, help="retrieval top-k")
    ap.add_argument("--text", default=None,
                    help="텍스트 1회 질의 (STT 건너뜀)")
    ap.add_argument("--audio", default=None,
                    help="오디오 파일 1회 질의 (마이크 불필요)")
    ap.add_argument("--no-speak", dest="speak", action="store_false",
                    help="TTS 생략")
    ap.add_argument("--show-retrieved", action="store_true",
                    help="발췌 path 표 표시")
    args = ap.parse_args()

    health = _health(args.endpoint)

    # --text 모드는 STT 불필요 — 모델 로드를 건너뛴다.
    if args.text:
        console.print(f"  [grey58]{CLI_TITLE}[/]  [dim](text mode)[/]")
        _handle_turn(args.endpoint, args.text, args.k,
                     args.show_retrieved, args.speak)
        return

    compute_type = "float16" if args.device == "cuda" else "int8"
    device_index = args.device_index if args.device == "cuda" else 0
    with console.status(f"[grey70]faster-whisper '{args.model}' 로드 중…[/]"):
        stt = STT(model_size=args.model, device=args.device,
                  device_index=device_index, compute_type=compute_type)

    _print_splash(args.endpoint, stt, health, args.speak)

    # --audio 모드: 파일 1회 (서버 검증용)
    if args.audio:
        query = stt.transcribe(args.audio)
        _handle_turn(args.endpoint, query, args.k,
                     args.show_retrieved, args.speak)
        return

    # 라이브 마이크 REPL
    while True:
        try:
            console.print("\n  [bold green]🎤[/] [dim]말씀하세요 — 끝나면 Enter[/] ",
                          end="")
            audio = record_until_enter()
        except (EOFError, KeyboardInterrupt):
            console.print("\n  [dim]종료합니다.[/]")
            break
        except RuntimeError as exc:
            err_console.print(f"[err] {exc}")
            sys.exit(2)

        with console.status("[grey70]받아쓰는 중…[/]", spinner="dots"):
            query = stt.transcribe(audio)
        _handle_turn(args.endpoint, query, args.k,
                     args.show_retrieved, args.speak)


if __name__ == "__main__":
    main()
