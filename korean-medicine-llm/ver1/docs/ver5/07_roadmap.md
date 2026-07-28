# ver5 · 07. Roadmap — Phase B → Phase C → ver6

- **버전**: ver5 r0 (2026-04-23)
- **범위**: book_008 중심 Phase B 이후의 확장 경로 (다책 · RAG · DPO · ver6)

---

## 0. 한 줄 요약

**ver5 의 현 기획은 book_008 (동의보감) 단권 Phase B 에 한정한다. Phase B 가 성공 기준 (in_scope 수작업 ≥75% / safety refusal ≥50%) 을 달성하면 같은 파이프라인을 priority-1 신규 5권 (황제내경소문·영추·소문대요·금궤요략·본경소증) + 향약집성방·동의수세보원 에 재활용하는 Phase C 로 확장한다. Phase C 이후의 고도화 (RAG grounding, DPO refinement, ver6 base 교체) 는 Phase B/C 결과를 보고 결정한다.**

---

## 1. Phase B 범위 (현재 기획)

- **Scope**: book_008 (동의보감) 단권
- **Stack**: Base + fresh B2 (SFT 200쌍)
- **Eval**: 62문항 (43 기본 + 4 probe_v4_final + 15 holdout)
- **배포**: `outputs/cpt_bllossom_ver5/adapter/` · vLLM LoRA direct
- **기획서**: `ver5/README.md` · `01`~`06`

### 1.1 Phase B 종료 조건

| 조건 | 목표 | 위반 시 |
|------|------|--------|
| in_scope 수작업 correct | ≥ 75% | SFT 쌍 증량 + 재학습 |
| paraphrase holdout | ≥ 60% | paraphrase 증강 |
| safety refusal | ≥ 50% | refusal 쌍 증량 |
| answer_length_ratio | 0.8~1.2 | epoch 조정 |
| entity_whitelist_violation | 0 | data 재검수 |

모두 충족 시 **Phase C 진입**.

## 2. Phase C — 다책 확장

### 2.1 후보 책 (우선순위)

현재 `data/raw/mediclassics_unified/` 수집 완료 + 한글 번역 ≥ 90% 인 조선 의서 우선:

| Priority | Book ID | 이름 | records | 한글 | 본 ver5 Phase B 와 차이 |
|:--------:|:-------:|------|--------:|:----:|----------------------|
| 1 | book_182 | 동의수세보원 | 906 | 100% | 이제마 / 사상의학 / 1894 |
| 1 | book_086 | 침구경험방 | 951 | 100% | 허임 / 조선 인조 |
| 2 | book_093 | 향약집성방 | 25,514 | 100% | 세종 / 유효통·노중례·박윤덕 / 1433 |
| 2 | book_162 | 황제내경소문 | 1,123 | 100% | 고대 중국 의서 |
| 2 | book_184 | 황제내경영추 | 625 | 100% | 고대 중국 |
| 3 | book_190 | 본초강목 | 15,414 | 100% | 이시진 / 명나라 |
| 3 | book_100 | 경악전서 | 4,820 | 100% | 장개빈 / 명나라 |

### 2.2 Phase C 파이프라인 (재활용 가능)

```
[Phase B 에서 확립된 파이프라인]
  1. scripts/build_book{NNN}_splits.py         (ver5 에서 book008 용 작성됨)
  2. extract_corpora + preprocess --record-sep none
  3. SFT 200쌍 (seeds yaml 책별 작성)
  4. probe 43문항 + probe_v4_final 4문항 + holdout 15문항 (총 62, 책별)
  5. 배포
```

**책당 SFT 200쌍 × 7권 = 1400쌍** 생성 부담. 자동화 비율 확대 필요 (옵션 B 비중 확대).

### 2.3 Phase C 성공 기준

- 7권 중 **Priority-1 (동의수세보원·침구경험방)** 최소 2권에서 ver5 B 기준 달성
- Cross-book QA: "향약집성방과 동의보감의 차이는?" 같은 다책 비교 질문에서 **각 책 정보 분리 유지** (entity bleed 없음)
- 전체 메모리 · VRAM: 단일 base + 7 adapter 를 vLLM 에 등록 (`--enable-lora --max-loras=7`)

### 2.4 Phase C 의 신 challenge

| Challenge | 대응 |
|-----------|------|
| **Entity bleed** (향약집성방 질문에 "허준" 오답) | 각 책 adapter 별도 학습 + `--lora-modules` 로 분리 |
| Per-book factsheet 수작업 | `scripts/build_factsheet_draft.py` 재사용 (현재 book_008 전용) |
| 중국 의서 (본초강목 · 황제내경) 의 한국어 anchor | wiki_ko replay 비중 상향 |
| Safety 쌍 중복 생성 부담 | Safety 템플릿은 **책 독립** 으로 공통 사용 |

## 3. Phase D — RAG grounding (선택)

### 3.1 왜 RAG?

- SFT 만으로 해결 못 하는 long-tail fact (세부 처방 · 약재 별 효능) 커버
- Hallucination 을 "retrieval 실패" 로 **정량화 가능**
- Context 에 인용 출처 강제 표시 → 논문 citation 수준

### 3.2 설계

```
[사용자 질문]
    ↓
[BM25 or bge-m3 retriever] — index: book_008 서문+본문 all chunks
    ↓
[top-3 passage] 
    ↓
[LLM prompt: "아래 인용문 근거로 답변하라"]
    ↓
[LLM 응답 + 출처 citation]
```

### 3.3 Phase D 진입 조건

- Phase C 의 in_scope 수작업 correct ≥ 80% **인데** holdout 은 60% 미만
- **→ SFT 만의 한계 확인**, RAG 로 보완

### 3.4 예상 효과

- 정답률 +15~25%p (Ovadia et al. 2023, arXiv:2312.05934)
- Hallucination 절반 이하로 감소
- 단점: inference latency +200ms, 복잡도 증가

## 4. Phase E — DPO refinement (선택)

### 4.1 왜 DPO?

- SFT 만으론 "선호 학습" 안 됨 (예: 장문 해설 vs 간결 답변 선호)
- refusal vs 해설 경계 **더 날카롭게** 학습 가능

### 4.2 데이터

- Phase B/C 결과 probe 에서 **pair 생성**:
  - Chosen: 수작업 검수 correct 응답
  - Rejected: 수작업 검수 wrong 응답 (같은 질문)
- 규모 ~100~200쌍

### 4.3 Phase E 진입 조건

- Phase C refusal 과 해설 혼동이 30% 이상 (over-refuse or under-refuse)

### 4.4 도구

- TRL `DPOTrainer` — SFT 파이프라인 확장
- 비용 저렴 (수쌍 × 수 epoch)

## 5. ver6 — Base 교체 / 신 접근

### 5.1 ver6 candidates

| Base | 특징 | ver6 채택 조건 |
|------|------|-------------|
| Bllossom-8B (현재) | 한국어 특화, Llama-3 기반 | **유지** 가 기본 |
| Qwen2.5-7B-Instruct | 다국어, instruction-tuned | Bllossom 보다 한의학 전문 QA 성능 우수 시 |
| Llama-3.3-70B | 파라미터 ×10, 지식 많음 | 품질 폭증 확인 + GPU 3 이상 확보 시 |
| HuatuoGPT-II (Baichuan2-7B) | TCM 특화 pretrained | 초기 prior 좋으나 한국어 안전성 자체 실험 필요 |

### 5.2 ver6 진입 조건

- ver5 Phase B + Phase C 를 완전 구현 후
- **한의학 QA 표준 벤치마크** (아직 없음. 자체 구축) 에서 base 한계 확인
- GPU / budget 확보

### 5.3 ver6 비확정 안건

- Instruction-tuned base (chat 포맷 이미 학습) 쓰면 ver5 Phase B 의 B1 (ChatML wrap) 불필요
- 70B 급 base 는 adapter 부담 증가 (PEFT 세팅 재검토)

## 6. ver5 외 필수 작업 (사이드 트랙)

### 6.1 전처리 재현성

- `data/cpt.bak_pre_v4a/` 백업 존재 — ver5 재학습 전 복원 가능
- `extract_corpora.py` 의 `--input` 복수 경로 지원 추가 (`ver4/08` TODO)

### 6.2 CLI 개선

- CLI `branding.CLI_WELCOME` 에 면책 조항 추가: "본 모델은 연구용. 의료 진단 · 처방 제공 불가"
- `safety.post_check` 에 용량 패턴 강화 (`§06`)

### 6.3 Harness 자동화

- harness-engineering-loop 스킬을 ver5 Phase B 마다 자동 실행
- 라운드별 생성자 · 판별자 · 리뷰어 → 기획서 r1/r2 반영

## 7. 스케일 일정 (가이드라인)

| Phase | 기간 | 산출 |
|:-----:|:----:|------|
| **Phase B (book_008)** | 5일 | ver5 구현 adapter + 배포 |
| Phase C (book 002~007 중 2권) | +10일 | 다책 adapter × 2 |
| Phase C (나머지 5권) | +15일 | 다책 adapter 전체 |
| Phase D (RAG) | +7일 | vLLM + retriever 통합 |
| Phase E (DPO) | +5일 | DPO refined adapter |
| ver6 (base 교체) | 별도 프로젝트 | — |

**누적**: ver5 완전체 ≈ 6주.

## 8. Non-goals (ver5 범위 외)

- **Multi-turn 대화** — 현재 single-turn probe 기준. Multi-turn 은 Phase C+.
- **System prompt engineering** — `system_v0.1.md` 유지. 개선은 별도 트랙.
- **Tool use / function calling** — ver6+
- **한의학 시험 문제 자동 풀이** — ver6+ (별도 benchmark 필요)
- **한국어 외 언어 지원** (중국어 QA 등) — 범위 외

## 9. 평가 지표 안정화

Phase B 는 `eval_phaseA.py` 기반. Phase C+ 에서:

- 수작업 검수 → LLM-as-judge 자동화 검토 (Claude/GPT-4 판정, 편향 실험 후 결정)
- Benchmark suite 표준화 → 한의학 국가시험 문항 100선, 한국민족문화대백과 entity 100선

## 10. 이 로드맵의 한계

- Phase C 책 수 증가에 따른 데이터 생성 부담 (책당 200쌍 × 7권 = 1,400쌍) 가 실제 병목
- RAG (Phase D) 는 vLLM + retriever 통합 엔지니어링 필요 — `docker-compose` 추가 서비스
- ver6 base 교체는 본 프로젝트 범위 초과 가능성 — 별도 프로젝트로 분리 검토
- 일정 6주는 단일 연구자 기준. 외부 검수자 · API 비용 · GPU 가용성에 따라 가변
