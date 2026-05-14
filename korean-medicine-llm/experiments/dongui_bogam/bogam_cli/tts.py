"""음성 자비스 TTS — edge-tts 래퍼 + RAG answer 텍스트 정제.

edge-tts: MS 뉴럴 음성(ko-KR), 무료·API 키 불필요·인터넷 필요. mp3 출력.
재생(play)은 sounddevice/PortAudio 필요 — 맥 라이브 전용. 서버에서는 synthesize
로 mp3 파일까지만 만들어 검증한다.
"""
from __future__ import annotations

import asyncio
import re
import tempfile

import edge_tts
import soundfile as sf

VOICE = "ko-KR-SunHiNeural"

_CITATION = re.compile(r"\[\d+\]")           # [1] 같은 인용 마커
_CJK = re.compile(r"[㐀-鿿]")        # 한자 — edge-tts 가 못 읽거나 깨뜨림
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


async def _synth(text: str, out_path: str, voice: str) -> None:
    await edge_tts.Communicate(text, voice).save(out_path)


def synthesize(text: str, out_path: str | None = None, voice: str = VOICE) -> str:
    """text → mp3 파일 합성. 파일 경로를 반환한다."""
    if out_path is None:
        fd = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        fd.close()
        out_path = fd.name
    asyncio.run(_synth(text, out_path, voice))
    return out_path


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
