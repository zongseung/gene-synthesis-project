#!/usr/bin/env python
"""HanMed 동의보감 음성 자비스 — 홀로그램 GUI (pywebview).

`voice.py`(터미널 버전)와 **동일한 STT/RAG/TTS 파이프라인**을 쓰되, 출력을
터미널 대신 홀로그램 창(`bogam_cli/hologram/index.html`)으로 한다.
`chat.py`·`voice.py` 는 일절 건드리지 않는다.

구조:
  - 메인 스레드   : pywebview 창 (GUI 는 메인 스레드 필수, 특히 macOS)
  - 백그라운드    : voice_loop — record → STT → RAG → TTS 상태머신
  - Python → JS   : window.evaluate_js(...) 로 상태·오디오 레벨 push
  - JS → Python   : js_api(Bridge) 로 창 클릭 이벤트 수신

상호작용: 홀로그램을 **클릭 → 녹음 시작, 다시 클릭 → 녹음 종료** → 처리 → 답변.

실행 (맥):
  hanmed-bogam-hologram --device cpu
"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import webview

from bogam_cli import tts
from bogam_cli.stt import SAMPLE_RATE, STT

HTML_PATH = Path(__file__).resolve().parent / "hologram" / "index.html"


class Bridge:
    """JS ↔ Python 브리지. JS 가 window.pywebview.api.click() 으로 호출."""

    def __init__(self) -> None:
        self._click = threading.Event()

    # ── JS 에서 호출되는 메서드 ──────────────────────────────────
    def click(self) -> None:
        """홀로그램 창 클릭 — 녹음 시작/종료 토글 신호."""
        self._click.set()

    # ── Python 쪽에서 사용 ──────────────────────────────────────
    def wait_click(self) -> None:
        self._click.wait()
        self._click.clear()


def _js(window, code: str) -> None:
    """evaluate_js 안전 래퍼 — 창이 닫히는 중이면 조용히 무시한다."""
    try:
        window.evaluate_js(code)
    except Exception:  # noqa: BLE001
        pass


def _set_state(window, state: str) -> None:
    _js(window, f"window.hologram.setState('{state}')")


def _set_status(window, text: str) -> None:
    """상태 라벨에 임의 텍스트 표시 (로딩·오류 메시지용)."""
    safe = text.replace("'", " ").replace("\n", " ").replace("\\", " ")[:64]
    _js(window, f"document.getElementById('status').textContent='{safe}'")


def _set_transcript(window, text: str) -> None:
    """인식 텍스트 영역에 표시 (STT 결과·진행 메시지용)."""
    safe = (text or "").replace("'", " ").replace("\n", " ").replace("\\", " ")[:120]
    _js(window, f"window.hologram.setTranscript('{safe}')")


def _show_error(window, msg: str) -> None:
    """오류를 표시하고 idle 로 복귀한다.

    오류 메시지는 transcript 영역에 남긴다 — status 라벨은 setState 가 곧바로
    LABEL['idle'] 로 덮어쓰므로 오류 표기에 부적합하다.
    """
    _set_state(window, "idle")
    _set_transcript(window, f"⚠ {msg}")


def _record_until_click(bridge: Bridge) -> np.ndarray:
    """마이크 녹음 → 16kHz mono float32. 다음 클릭이 올 때까지 캡처한다."""
    import sounddevice as sd

    frames: list = []
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=lambda indata, *_: frames.append(indata.copy()),
    ):
        bridge.wait_click()  # 두 번째 클릭까지 블로킹
    if not frames:
        return np.zeros(0, dtype="float32")
    return np.concatenate(frames, axis=0).flatten()


def _rag_answer(endpoint: str, query: str, k: int) -> dict:
    r = httpx.post(
        f"{endpoint}/rag/answer", json={"query": query, "k": k}, timeout=120
    )
    r.raise_for_status()
    return r.json()


def _speak_with_levels(window, text: str, backend: str) -> None:
    """answer 정제 → TTS 합성 → 재생하며 오디오 진폭을 JS 로 push한다."""
    import sounddevice as sd

    spoken = tts.clean_for_speech(text)
    if not spoken:
        return
    mp3 = tts.synthesize(spoken, backend=backend)
    data, sr = tts.load_audio(mp3)
    if data.ndim > 1:
        data = data.mean(axis=1)

    sd.play(data, sr)
    win = int(sr * 0.05)
    start = time.time()
    stream = sd.get_stream()
    while stream is not None and stream.active:
        pos = int((time.time() - start) * sr)
        chunk = data[pos:pos + win]
        level = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0
        _js(window, f"window.hologram.pushLevel({min(1.0, level * 4.0):.3f})")
        time.sleep(0.05)
    sd.stop()
    _js(window, "window.hologram.pushLevel(0)")


def voice_loop(window, bridge: Bridge, cfg: argparse.Namespace) -> None:
    """백그라운드 음성 루프 — idle→listening→thinking→speaking 상태머신."""
    # STT 모델 로드 (창은 이미 떠 있음 — '준비 중' 표시)
    _set_status(window, "엔진 준비 중…")
    compute_type = "float16" if cfg.device == "cuda" else "int8"
    device_index = cfg.device_index if cfg.device == "cuda" else 0
    try:
        stt = STT(
            model_size=cfg.model,
            device=cfg.device,
            device_index=device_index,
            compute_type=compute_type,
        )
    except Exception as exc:  # noqa: BLE001
        _set_state(window, "idle")
        _set_transcript(window, f"⚠ STT 로드 실패: {exc}")
        return

    # RAG 서버 연결 확인 — 안 되면 경고 (SSH 터널 미연결이 흔한 원인)
    _set_state(window, "idle")
    try:
        httpx.get(f"{cfg.endpoint}/health", timeout=5).raise_for_status()
    except Exception as exc:  # noqa: BLE001
        _set_transcript(
            window, f"⚠ RAG 서버 연결 안 됨 — SSH 터널(8080) 확인 ({exc})"
        )

    while True:
        # 1) 대기 → 클릭 → 녹음 → 클릭
        bridge.wait_click()
        _set_transcript(window, "")
        _set_state(window, "listening")
        audio = _record_until_click(bridge)

        # 1-b) 마이크 입력 점검 — 무음/너무 짧으면 권한·장치 문제
        dur = len(audio) / SAMPLE_RATE
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if dur < 0.3 or peak < 0.01:
            _show_error(
                window,
                f"마이크 입력 없음 (길이 {dur:.1f}s · 피크 {peak:.3f}) — "
                f"터미널 앱의 마이크 권한 확인",
            )
            continue

        # 2) 받아쓰기
        _set_state(window, "thinking")
        _set_transcript(window, f"({dur:.1f}초 녹음 · 받아쓰는 중…)")
        try:
            query = stt.transcribe(audio)
        except Exception as exc:  # noqa: BLE001
            _show_error(window, f"STT 오류: {exc}")
            continue
        if not query.strip():
            _show_error(window, "음성을 알아듣지 못했어요 — 다시 시도해 주세요")
            continue
        _set_transcript(window, f"❝ {query} ❞")

        # 3) RAG
        try:
            answer = _rag_answer(cfg.endpoint, query, cfg.k).get("answer", "")
        except Exception as exc:  # noqa: BLE001
            _show_error(window, f"RAG 오류: {exc} — SSH 터널(8080) 확인")
            continue

        # 4) 답변 음성 + 파형
        _set_state(window, "speaking")
        try:
            _speak_with_levels(window, answer, cfg.tts)
        except Exception as exc:  # noqa: BLE001
            _show_error(window, f"재생 오류: {exc}")
            continue
        _set_state(window, "idle")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="HanMed 동의보감 음성 자비스 — 홀로그램 GUI"
    )
    ap.add_argument("--endpoint", default="http://localhost:8080",
                    help="RAG sidecar URL")
    ap.add_argument("--model", default="large-v3-turbo",
                    help="faster-whisper 모델")
    ap.add_argument("--device", default="cpu", choices=["cuda", "cpu"],
                    help="STT 디바이스 (맥: cpu, 서버 GPU: cuda)")
    ap.add_argument("--device-index", type=int, default=0,
                    help="cuda 디바이스 번호")
    ap.add_argument("-k", type=int, default=5, help="retrieval top-k")
    ap.add_argument("--tts", default="openai", choices=["openai", "edge"],
                    help="TTS 백엔드")
    args = ap.parse_args()

    if not HTML_PATH.is_file():
        raise SystemExit(f"홀로그램 HTML 을 찾을 수 없습니다: {HTML_PATH}")

    bridge = Bridge()
    window = webview.create_window(
        "HanMed · 동의보감 음성 자비스",
        url=str(HTML_PATH),
        js_api=bridge,
        width=440,
        height=640,
        frameless=True,
        easy_drag=False,  # 클릭=녹음 과 충돌하지 않도록 드래그 비활성
        on_top=True,
        background_color="#03070A",
    )
    webview.start(lambda: voice_loop(window, bridge, args))


if __name__ == "__main__":
    main()
