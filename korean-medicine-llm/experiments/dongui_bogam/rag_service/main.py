"""HanMed RAG sidecar — FastAPI app.

Pipeline:
  query → safety.pre_check → hybrid_search → prompt → vLLM /v1/chat/completions
        → safety.post_check → response

Endpoints:
  GET  /health              — self + vLLM upstream health
  POST /rag/answer          — main RAG pipeline
  GET  /rag/retrieve        — retrieval only (debug, no LLM call)
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# hanmed_cli 는 Dockerfile 이 src/hanmed_cli 를 /app 에 복사 → PYTHONPATH=/app
from hanmed_cli.safety import is_clinical_intent, post_check, pre_check  # type: ignore

from rag_core import RagCore, RetrievedItem
from settings import settings

logger = logging.getLogger("hanmed_rag")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s — %(message)s")


SYSTEM_RAG = (
    "당신은 한의학 고전 문헌 연구 보조 AI 입니다. 사용자 질문에 답할 때 다음 규칙을 "
    "엄격히 지키세요.\n\n"
    "1. [동의보감 발췌] 안의 글자만 사용해서 답하세요. 발췌에 없는 약재명·처방명·"
    "편명·인용·약재 개수는 절대 만들지 마세요.\n"
    "2. 약재 이름·처방 이름은 발췌의 표기를 한 글자도 바꾸지 말고 그대로 옮기세요.\n"
    "3. 약재 목록을 답할 때는 발췌의 'ㆍ' 또는 ',' 로 구분된 토큰만 그대로 나열하세요.\n"
    "4. 발췌만으로 답할 수 없으면 '발췌 자료에서는 확인할 수 없습니다' 라고 답하세요.\n"
    "5. 답변 끝에 [N] 으로 어느 발췌를 근거로 했는지 표시하세요.\n"
    "6. 본문 인용 뒤에 **자연스러운 현대 한국어 어투**로 2-4 문장의 풀이를 한국어로 "
    "덧붙이세요. 본문의 한문 직역체 표현 (예: '오장의 양을 보한다', '담을 삭인다') 을 "
    "일상적 한국어 ('오장의 양기를 보충합니다', '담(가래)을 삭여 줍니다') 로 풀어쓰세요. "
    "단 풀이에서도 발췌에 없는 약재명·처방명·수치·인용은 만들지 마세요. "
    "(풀이는 '풀이:' 로 시작해 본문과 시각적으로 구분하세요.)\n"
    "7. 처방·약재 관련 질문에는 발췌에 있는 정보를 다음 항목 순서로 묶어 정리하세요. "
    "발췌에 없는 항목은 생략하세요 (지어내지 말 것):\n"
    "   • 적응증 (어떤 증상·병에 쓰는지)\n"
    "   • 약재 구성 (발췌의 'ㆍ' 토큰 그대로)\n"
    "   • 용법·복용법 (썰어·달여·가루내어 등)\n"
    "   • 출전 (《국방》《단심》《입문》 같은 인용서)\n"
    "   **항목 정리 후에는 룰 6 에 따라 '풀이:' 섹션을 반드시 추가**해, 일반 사용자가 "
    "이해하기 쉽도록 자연스러운 현대 한국어로 2-4 문장의 부연 설명을 하세요. "
    "풀이는 어떤 답변에도 빠뜨리지 마세요."
)


# ── App lifecycle ─────────────────────────────────────────────────────────
state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("loading FAISS + meta + encoder...")
    t0 = time.time()
    state["rag"] = RagCore(
        index_path=settings.faiss_index_path,
        meta_path=settings.faiss_meta_path,
        encoder_name=settings.encoder_name,
        encoder_device=settings.encoder_device,
        min_body_len=settings.min_body_len,
    )
    logger.info(f"  loaded in {time.time()-t0:.1f}s "
                f"({len(state['rag'].meta)} meta records)")

    state["http"] = httpx.AsyncClient(
        base_url=settings.vllm_url, timeout=settings.vllm_timeout_s
    )
    yield
    await state["http"].aclose()


app = FastAPI(title="HanMed RAG sidecar", version="ver8.1", lifespan=lifespan)


# ── Schemas ───────────────────────────────────────────────────────────────
class AnswerRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(default=settings.top_k, ge=1, le=20)
    boost_k: int = Field(default=settings.boost_k, ge=0, le=10)


class RetrievedOut(BaseModel):
    rank: int
    src: str
    sim: float
    path: str | None
    content_level: str | None
    body_snippet: str


class SafetyOut(BaseModel):
    pre_refused: bool
    refusal_text: str | None
    post_masked: bool
    clinical_intent: bool  # pre_check 통과했지만 약한 진료성 — dosage 마스킹 적용 여부


class AnswerResponse(BaseModel):
    answer: str
    extracted_names: list[str]
    retrieved: list[RetrievedOut]
    safety: SafetyOut
    mode: str  # "STRICT" | "REFUSED"
    elapsed_ms: int
    prompt_tokens: int | None = None


# ── Helpers ───────────────────────────────────────────────────────────────
def _retrieved_to_out(items: list[RetrievedItem]) -> list[RetrievedOut]:
    out = []
    for i, it in enumerate(items, 1):
        body = (it.rec.get("trans_ko") or it.rec.get("original") or "").replace("\n", " ").strip()
        out.append(RetrievedOut(
            rank=i, src=it.src, sim=round(it.sim, 4),
            path=it.rec.get("up_path_nm"),
            content_level=it.rec.get("content_level"),
            body_snippet=body[:120],
        ))
    return out


async def _call_vllm(messages: list[dict[str, str]]) -> tuple[str, int | None]:
    payload = {
        "model": settings.vllm_model,
        "messages": messages,
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "repetition_penalty": settings.repetition_penalty,
    }
    try:
        resp = await state["http"].post("/v1/chat/completions", json=payload)
    except httpx.HTTPError as e:
        logger.error(f"vLLM upstream error: {e}")
        raise HTTPException(status_code=502, detail=f"vllm upstream: {e}") from e

    if resp.status_code != 200:
        raise HTTPException(status_code=502,
                            detail=f"vllm {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    text = data["choices"][0]["message"]["content"].strip()
    prompt_tokens = data.get("usage", {}).get("prompt_tokens")
    return text, prompt_tokens


# ── Routes ────────────────────────────────────────────────────────────────
@app.get("/health")
async def health() -> dict[str, Any]:
    out: dict[str, Any] = {"rag": "ok", "vllm": "unknown"}
    try:
        r = await state["http"].get("/health", timeout=3)
        out["vllm"] = "ok" if r.status_code == 200 else f"status={r.status_code}"
    except Exception as e:
        out["vllm"] = f"error: {type(e).__name__}"
    return out


@app.get("/rag/retrieve")
async def retrieve(query: str, k: int = 5, boost_k: int = 2) -> dict[str, Any]:
    items, names = state["rag"].hybrid_search(query, k=k, boost_k=boost_k)
    return {
        "query": query,
        "extracted_names": names,
        "retrieved": [r.model_dump() for r in _retrieved_to_out(items)],
    }


@app.post("/rag/answer", response_model=AnswerResponse)
async def answer(req: AnswerRequest) -> AnswerResponse:
    t0 = time.time()
    pre = pre_check(req.query)
    if pre.refused:
        return AnswerResponse(
            answer=pre.refusal_text or "",
            extracted_names=[],
            retrieved=[],
            safety=SafetyOut(pre_refused=True,
                             refusal_text=pre.refusal_text,
                             post_masked=False,
                             clinical_intent=False),
            mode="REFUSED",
            elapsed_ms=int((time.time() - t0) * 1000),
        )

    # context-aware masking: 학술 query 는 본문 그대로, 약한 진료성 query 는 보호.
    clinical = is_clinical_intent(req.query)

    items, names = state["rag"].hybrid_search(
        req.query, k=req.k, boost_k=req.boost_k
    )
    excerpt = state["rag"].build_excerpt_block(items)
    user_msg = f"[동의보감 발췌]\n\n{excerpt}\n\n[질문] {req.query}"
    messages = [
        {"role": "system", "content": SYSTEM_RAG},
        {"role": "user", "content": user_msg},
    ]

    raw, prompt_tokens = await _call_vllm(messages)
    final = post_check(raw, mask_dosage=clinical)
    masked = final != raw

    return AnswerResponse(
        answer=final,
        extracted_names=names,
        retrieved=_retrieved_to_out(items),
        safety=SafetyOut(pre_refused=False, refusal_text=None,
                         post_masked=masked, clinical_intent=clinical),
        mode="STRICT",
        elapsed_ms=int((time.time() - t0) * 1000),
        prompt_tokens=prompt_tokens,
    )
