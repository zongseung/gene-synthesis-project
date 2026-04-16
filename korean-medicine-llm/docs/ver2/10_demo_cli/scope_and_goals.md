# 10.1 Scope and Goals

## In-scope (v0)

- `hanmed chat` — 단일 process 로컬 REPL (한의학 질의응답)
- 기관 서버 사용은 **S1: SSH 로그인 후 REPL 실행** 까지만 포함 (`hanmed serve` / `--remote` 는 v1)
- **Bllossom-8B + HanMed-CPT LoRA (Stage 1)** 로드 — 기본 경로
- (옵션) Stage 2 SFT adapter 로드 — 10.3 참조
- ChatML 프롬프트 템플릿 (Llama-3 chat)
- 멀티턴 대화 (컨텍스트 최대 8K tokens, sliding window)
- Streaming output
- 세션 저장/불러오기
- 안전성 refusal layer (§05 T4 패턴)
- 출처 고정 footer ("KIOM mediclassics.kr 기반 학습")
- 배포: 로컬 + 기관 서버 (10.7 참조)

## Out-of-scope (v0 에서 제외)

- ❌ **RAG** — §README.R3.3 변경, 기여 축 분산 방지
- ❌ Agentic tool use (파일 편집, bash 실행, 웹 검색 등 Claude Code 류)
- ❌ 웹/GUI UI
- ❌ 서버-클라이언트 분리 (v1 에서 `hanmed serve` 검토)
- ❌ 음성 / 이미지 입출력
- ❌ 클라우드 배포 (KIOM 승인 전)
- ❌ 모델 학습 (별도 §04a)
- ❌ 4-bit quantization (v1)

## 왜 RAG 를 쓰지 않는가

### 논문 기여 축 분산 방지

ver2 §01.4 primary 기여:
1. 병렬 한문-한국어 CPT 레시피
2. 한의학 평가 벤치 v0
3. bilingual block concatenation 효과 분석

RAG 를 추가하면 기여가 분산되어 논문이 "CPT vs RAG vs CPT+RAG 비교" 가 되고, 이는 사용자 명시 피드백 ("comparison paper 아님") 에 위배.

### 측정 왜곡

CPT 의 knowledge injection 성공 여부가 ver2 논문의 핵심 주장. RAG 로 감싸면 **retrieval 이 좋아서 답변이 정확한지 CPT 가 knowledge 를 학습했는지 구분 불가**. §05 평가 지표 해석이 노이즈로 오염됨.

### 구현 복잡도

RAG 시스템은 (a) chunking, (b) embedding 모델, (c) vector DB, (d) retrieval, (e) rerank, (f) prompt stitching 6 단계가 추가. 각각이 ablation 대상 → M2 실행 부담이 2배 이상 증가. v0 scope 초과.

### Null-result fallback (R3.4 — 판별자 지적 반영, 순서 역전)

만약 `§E ablation` 에서 CPT-only 가 null-result 로 기각되면, **실패 축에 따라 분기**:

| 실패 증거 | 1차 fallback | 근거 |
|---|---|---|
| **T2 QA factual accuracy** 불충분 (knowledge 부족) | **RAG** (retrieval + mediclassics 원문 inject) | Meditron / HuatuoGPT-II 선행사례. SFT 는 knowledge injection 해결하지 못함 |
| **T4 format / instruction following** 불충분 (format 부족) | **SFT** (instruction-tuning) | format 학습은 supervised 쌍이 효율적 |
| 둘 다 | RAG + SFT 순차 검토 | |

즉 R3.3 의 "SFT 1차 fallback, RAG v2 이연" 은 **R3.4 에서 폐기**. **RAG 는 knowledge null-result 의 1차 fallback** 으로 정정. 단 ver2 primary scope (CPT 레시피 검증) 에서는 여전히 RAG 제외. RAG 활성화는 null-result 가 realized 된 이후에만.

## 유사 프로젝트 비교

| 프로젝트 | 추론 | 인터페이스 | 라이선스 | 본 스펙과의 차이 |
|---|---|---|---|---|
| **Claude Code** | Anthropic API | REPL + tool use | 상용 | 원격 API, agentic, 파일 편집 가능 |
| **Gemini CLI** | Google API | REPL | 상용 | 원격 API |
| **Ollama** (`ollama run`) | llama.cpp GGUF | REPL | MIT | 로컬 GGUF, 모델 레지스트리 |
| **LM Studio CLI** | llama.cpp | REPL | freemium | GUI 동반, 범용 |
| **Simon Willison `llm`** | plugin (OpenAI/local) | CLI + REPL | Apache-2.0 | 다양한 backend, SQLite 로그 |
| **llama.cpp `-i`** | C++ | REPL | MIT | 경량, GGUF 필요 |
| **vLLM + `curl`** | vLLM | HTTP | Apache-2.0 | 고성능, 별도 클라이언트 |
| **HanMed-CLI (본 스펙)** | vLLM / transformers | REPL | TBD | **한의학 domain refusal + 한문/국역 bilingual + KIOM 출처** |

v0 는 Ollama + Simon Willison `llm` 의 중간 — 단일 도메인 특화, 프롬프트 template hardcoded, safety layer 내장.

## 목표 지표 (E1~E5)

| # | 조건 | 측정 |
|---|---|---|
| E1 | REPL prompt 표시 < 5 s, **cold** first-token < 30 s, **warm** first-token < 5 s | startup + latency timer |
| E2 | §05 safety gate 통과: core 20개 ≥ 99%, paraphrase 30개 ≥ 95%, 한문 jailbreak 10개 ≥ 90% | `hanmed eval` |
| E3 | 멀티턴 8K context sliding window 후에도 system prompt 보존 | conversation 단위 테스트 |
| E4 | 세션 save/load round-trip 무결성 | JSON schema 검증 |
| E5 | Peak GPU mem < 30 GB (8K context) | `nvidia-smi --loop` |

`E1` 은 R3.5 에서 분리 정의. `chat.py` 가 backend lazy-load 를 택하면 REPL 진입은 빠르지만 첫 질의는 model load 를 포함하므로, cold/warm latency 를 같은 숫자로 묶으면 운영 지표가 왜곡된다.

세부 마일스톤은 `milestones_and_exit.md`.
