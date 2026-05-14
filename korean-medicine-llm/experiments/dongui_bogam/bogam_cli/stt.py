"""음성 자비스 STT — faster-whisper 래퍼.

서버: GPU1(cuda:1)에서 동작 — GPU0 의 vLLM 과 간섭 없음.
맥:   device="cpu" (Apple Silicon) — 같은 코드, 생성자 인자만 다름.

녹음(record_until_enter)은 sounddevice/PortAudio 가 필요하다. 서버에는 PortAudio
가 없으므로 라이브 녹음은 맥에서만 가능하고, 서버에서는 transcribe(파일/배열)만
쓴다.
"""
from __future__ import annotations

import time
from pathlib import Path

from faster_whisper import WhisperModel

# large-v3-turbo: large-v3 대비 약 6배 빠르고 정확도는 거의 동일 — 대화형에 적합.
DEFAULT_MODEL = "large-v3-turbo"
SAMPLE_RATE = 16000


class STT:
    """faster-whisper 모델 래퍼. 생성 시 모델을 1회 로드한다."""

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL,
        device: str = "cuda",
        device_index: int = 1,
        compute_type: str = "float16",
        language: str = "ko",
    ) -> None:
        self.language = language
        self.model_size = model_size
        t0 = time.time()
        self.model = WhisperModel(
            model_size,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
        )
        self.load_seconds = time.time() - t0

    def transcribe(self, audio) -> str:
        """audio: 파일 경로(str/Path) 또는 16kHz mono float32 numpy 배열.

        vad_filter 로 앞뒤 무음을 잘라 환각성 토큰을 줄인다.
        """
        if isinstance(audio, (str, Path)):
            audio = str(audio)
        segments, _ = self.model.transcribe(
            audio, language=self.language, vad_filter=True
        )
        return " ".join(seg.text.strip() for seg in segments).strip()


def record_until_enter():
    """마이크 녹음 → 16kHz mono float32 numpy 배열 (push-to-talk).

    Enter 를 누르면 녹음이 끝난다. sounddevice/PortAudio 필요 — 맥 라이브 전용.
    """
    try:
        import numpy as np
        import sounddevice as sd
    except OSError as exc:  # PortAudio 미설치 시 sounddevice import 가 OSError
        raise RuntimeError(
            "sounddevice/PortAudio 를 사용할 수 없습니다 — 라이브 녹음은 맥에서 "
            f"실행하세요 (서버는 PortAudio 미설치). 원인: {exc}"
        ) from exc

    frames: list = []

    def _callback(indata, _frames, _time, _status):
        frames.append(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=_callback
    ):
        input()  # Enter 까지 블로킹

    if not frames:
        return np.zeros(0, dtype="float32")
    return np.concatenate(frames, axis=0).flatten()
