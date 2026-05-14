"""음성 자비스 TTS — openai / edge-tts 백엔드 + RAG answer 텍스트 정제.

백엔드:
  - openai : gpt-4o-mini-tts. `instructions` 로 톤·생동감을 글로 지시할 수 있어
             "인공적" 느낌을 잡기 좋다. API 키 필요.
  - edge   : edge-tts. 무료·키 불필요. 한국어 음성 3종 (SunHi/InJoon/Hyunsu).

기본 백엔드는 openai. 키를 못 찾거나 API 오류가 나면 edge 로 자동 폴백한다.
키 탐색 순서: 환경변수 OPENAI_API_KEY → API_key → 상위 디렉토리의 .env 파일.

재생(play)은 sounddevice/PortAudio 필요 — 맥 라이브 전용.
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
import tempfile
from pathlib import Path

import edge_tts
import soundfile as sf

# ── 백엔드 설정 ───────────────────────────────────────────────────────────
DEFAULT_BACKEND = os.environ.get("TTS_BACKEND", "openai")

EDGE_VOICE = "ko-KR-SunHiNeural"

OPENAI_MODEL = "gpt-4o-mini-tts"
OPENAI_VOICE = "nova"
# gpt-4o-mini-tts 는 instructions 로 말투를 지시받는다 — 생동감의 핵심 레버.
OPENAI_INSTRUCTIONS = (
    "한의학 고전을 풀어 설명해 주는 따뜻하고 친근한 선생님처럼 읽어 주세요. "
    "또박또박하되 너무 빠르지 않게, 생동감 있고 자연스러운 한국어 억양으로요."
)

# ── 텍스트 정제 ───────────────────────────────────────────────────────────
_CITATION = re.compile(r"\[\d+\]")           # [1] 같은 인용 마커
_CJK = re.compile(r"[㐀-鿿]")        # 한자 — TTS 가 못 읽거나 깨뜨림
_EMPTY_PAREN = re.compile(r"[（(]\s*[)）]")   # 한자 제거 후 남는 빈 괄호
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{2,}")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,!?])")  # 마커 제거 후 남는 " ." 정리


def clean_for_speech(answer: str) -> str:
    """RAG answer 를 TTS 가 자연스럽게 읽도록 정제한다.

    화면 표시용 텍스트에는 음성에 부적합한 요소가 있다:
      - [N] 인용 마커        → 제거
      - '풀이:' 라벨         → 자연스러운 구어 전환구로 치환
      - 한자 표기(人蔘 등)   → 제거 (한글 표기는 보통 함께 있음)
      - 한자 제거 후 빈 괄호 → 제거
    """
    text = _CITATION.sub("", answer)
    text = text.replace("풀이:", " 쉽게 설명드리면, ")
    text = _CJK.sub("", text)
    text = _EMPTY_PAREN.sub("", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _MULTI_NL.sub("\n", text)
    return text.strip()


# ── OpenAI 키 탐색 ────────────────────────────────────────────────────────
# 일반 + 스마트 따옴표 — .env 를 에디터에서 편집하면 따옴표가 곡선으로 바뀌곤 한다.
_QUOTE_CHARS = "\"'“”‘’"


def _unquote(value: str) -> str:
    """양끝의 공백·(스마트 포함)따옴표를 벗긴다."""
    return value.strip().strip(_QUOTE_CHARS).strip()


def _find_openai_key() -> str | None:
    """환경변수 → 상위 디렉토리 .env 순으로 OpenAI 키를 찾는다."""
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_key")
    if key:
        return _unquote(key)
    # tts.py 위치에서 위로 올라가며 .env 탐색 (서버·맥 동일하게 동작)
    for parent in Path(__file__).resolve().parents:
        env_file = parent / ".env"
        if not env_file.is_file():
            continue
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            for name in ("OPENAI_API_KEY=", "API_key="):
                if line.startswith(name):
                    return _unquote(line[len(name):])
    return None


# ── 백엔드별 합성 ─────────────────────────────────────────────────────────
async def _synth_edge(text: str, out_path: str, voice: str) -> None:
    await edge_tts.Communicate(text, voice).save(out_path)


def _synthesize_edge(text: str, out_path: str, voice: str = EDGE_VOICE) -> str:
    asyncio.run(_synth_edge(text, out_path, voice))
    return out_path


def _synthesize_openai(text: str, out_path: str, voice: str = OPENAI_VOICE) -> str:
    from openai import OpenAI

    key = _find_openai_key()
    if not key:
        raise RuntimeError(
            "OpenAI API 키를 찾을 수 없습니다 — .env 의 API_key 또는 "
            "환경변수 OPENAI_API_KEY 를 설정하세요."
        )
    client = OpenAI(api_key=key)
    with client.audio.speech.with_streaming_response.create(
        model=OPENAI_MODEL,
        voice=voice,
        input=text,
        instructions=OPENAI_INSTRUCTIONS,
        response_format="mp3",
    ) as response:
        response.stream_to_file(out_path)
    return out_path


# ── 공개 API ──────────────────────────────────────────────────────────────
def synthesize(text: str, out_path: str | None = None,
               backend: str | None = None) -> str:
    """text → 음성 파일 합성. 파일 경로를 반환한다.

    backend: 'openai' | 'edge'. 기본은 DEFAULT_BACKEND(openai).
    openai 백엔드가 키 없음·API 오류로 실패하면 edge 로 폴백한다.
    """
    if out_path is None:
        fd = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        fd.close()
        out_path = fd.name
    backend = backend or DEFAULT_BACKEND
    if backend == "openai":
        try:
            return _synthesize_openai(text, out_path)
        except Exception as exc:  # noqa: BLE001 — 키 없음·네트워크·API 오류 전부 폴백
            sys.stderr.write(
                f"\n  ※ OpenAI TTS 실패 → edge-tts 로 폴백 ({exc})\n"
            )
            return _synthesize_edge(text, out_path)
    return _synthesize_edge(text, out_path)


def load_audio(audio_path: str):
    """mp3/wav → (float32 numpy 배열, samplerate). 재생/길이 계산용."""
    data, samplerate = sf.read(audio_path, dtype="float32")
    return data, samplerate


def play(audio_path: str, on_tick=None, tick_hz: int = 8) -> None:
    """오디오 재생. sounddevice/PortAudio 필요 — 맥 라이브 전용.

    on_tick 콜백을 주면 재생 중 1/tick_hz 초마다 호출 — 말하는 중 애니메이션용.
    """
    try:
        import sounddevice as sd
    except OSError as exc:
        raise RuntimeError(
            f"오디오 재생 불가 — sounddevice/PortAudio 미설치 ({exc})"
        ) from exc

    data, samplerate = load_audio(audio_path)
    sd.play(data, samplerate)
    if on_tick is None:
        sd.wait()
        return

    import time

    stream = sd.get_stream()
    while stream is not None and stream.active:
        on_tick()
        time.sleep(1.0 / tick_hz)
    sd.stop()
