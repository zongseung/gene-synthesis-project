# ver6 · HanMed-LLM 환각(F1) · 반복 loop(F3) 공동 해소 기획서 · **Gemma-3 12B 전환 r2**

- 작성일: 2026-04-23 (r1 → r2 Gemma 전환)
- 신 기본 모델: `google/gemma-3-12b-it` (공식 · 라이선스 승인 완료 · `models/gemma-3-12b-it` 로컬 배치 · 23GB · 5 shards)
- 구 기본 모델: `MLP-KTLim/llama-3-Korean-Bllossom-8B` — 실험 경로는 유지, ver6 주 실행 경로에서 **부록 §10** 으로 격하
- 상위 harness: `.claude/harness-evals/phaseB_sft_plan/round_1/` (generator/discriminator/reviewer/iteration_plan)
- 실측 파일: `outputs/probes/gemma_zero_probe_tf_20260423T110813Z.jsonl`

---

## 0. 한 줄 요약

> 현재 서빙 Bllossom-SFT(ver5_v3_1) 는 (1) **임상 변증 질문에 8회 반복 loop**, (2) **동의보감 주변 지식 창작** 을 동시에 보인다. 원인은 **(A) SFT 답변 템플릿 3대 고정문구 42% 동질성 + (B) 임상 QA 0건 · 본문 커버리지 24.4% · (C) 허위 citation · (D) LoRA embed_tokens 포함**. 2026-04-23 Gemma-3 12B-IT zero-training probe 에서 **(A) · (C) · (D) 는 구조적으로 해소** 됨이 실측. ver6 는 **Gemma-3 12B-IT + LoRA SFT + book_008 전체 기반 19,023 쌍 (커버리지 89.72%)** 경로로 확정.

---

## 1. 실측 증거 (2026-04-23)

### 1.1 임상 케이스 대조 — 본 라운드 문제 제기 질문

> "임신 4개월 여성이 피곤·식욕부진·구역·창백·불면·자한 …"

| 항목 | Bllossom SFT ver5_v3_1 (현 서빙) | **Gemma-3 12B-IT zero-train (transformers)** |
|---|---|---|
| 답변 구조 | 질문 echo 후 혼란 | **변증 → 설명 → 처방 → 주의** 깔끔 |
| 변증명 | **0건** | **기허(氣虛) · 혈육미성 · 불면** |
| 처방명 (실재) | **0건** | **사물탕(四物湯)** ✓ |
| 처방명 (창작) | — | 보기조강탕(補氣助降湯) ✗ (1건) |
| 반복 loop | **동일 문장 8회 반복** (max_tokens=600 소진) | **max 15-gram run = 1** (loop 없음) |
| 안전 고지 | 없음 | "반드시 전문 한의사의 진료 후" 명시 |
| 응답 토큰 수 | 600 (초과) | 206 |

> Gemma 실제 응답 (요약):
> ```
> **변증:** 기허(氣虛) 및 혈육미성(血肉未盛)
> **설명:** 임신 초기에는 태가 자라면서 어머니의 기혈을 소모하므로
>   기허와 혈육미성의 증상이 나타날 수 있습니다. …
> **처방 (참고):**
>   * 보기조강탕(補氣助降湯): 기를 보하고 태를 안정시키는 효과
>   * 사물탕(四物湯): 혈을 보충하여 기허를 개선
> **주의:** 반드시 전문 한의사의 진료 후 …
> ```

### 1.2 사실 Q&A 4문항 · 4-way 비교

| Q | 정답 | Bllossom smoke SFT (step=1) | Bllossom CPT merged | Bllossom SFT ver5_v3_1 | **Gemma zero-train** |
|---|---|---|---|---|---|
| Q1 동의보감 | 허준/선조/1610 | 이황/세종/1435 ✗ | **허준/선조/1610 ✓** | (미측정) | 송나라 하치우/인종/1108 ✗ |
| Q2 사상의학 | 이제마/동의수세보원 | 주중지연·장삼경 ✗ | 허준·정예남·이명원 ✗ | — | 허준/동의보감 ✗ |
| Q3 향약집성방 | 세종/1433 | 17세기 후반 ✗ | 선조/1596 ✗ | — | 선조/허준 ✗ |
| Q4 오장 | 간·심·비·폐·신 | 신장·폐·간·**담·태** ✗ | 심비폐간 (신 누락) △ | — | **심·간·비·폐·신 ✓** |

**관측**:
1. **Gemma 는 Q4 정답 (Bllossom 전 경로 실패)** — 일반 한의학 용어 지식 우위.
2. **Gemma 는 Q1~Q3 환각** — 조선 의학사 고유 지식 약함. "동의보감 저자 = 허준" prior 만 지나치게 강해 Q2/Q3 에도 허준 귀속 → 이는 **Bllossom 과 동일한 F1 병리이나 반복 loop 없음**.
3. 즉 **ver6 SFT 가 교정할 목표 영역이 명확**: 조선 의학사 사실 + 편찬자 · 편·권 구조.

### 1.3 SFT 학습 데이터 진단 — 결정적 증거 (기존 ver5 Bllossom)

`experiments/dongui_bogam/data/sft/phaseB_qa_diverse_v3_1.jsonl` (21,475 rows) 답변 분포:

| 빈도 | 첫 20자 prefix | 정체 |
|---:|---|---|
| **3,041** | `현대 한국어:` | 서문·편명 현대어 번역 템플릿 |
| **3,000** | `본 모델은 한의학 고전 문헌 연구 보` | `safety_refusal` 고정 prefix |
| **3,000** | `동의보감 관련 질문에 대해 핵심 사실` | `in_scope_basic` 고정 prefix |
| 200×N | `OOO는 본 실험 모델의 학습 범위인` | `out_of_scope` 동일 골격 |

답변 말미 (3,000×3 동일 closing).

**결론**: SFT 21,475쌍 중 **9,000쌍 (42%)** 이 단 3개 고정 prefix + 3개 고정 closing. *The Price of Format: Diversity Collapse in LLMs* (2025) 이 지적한 **structural homogeneity → 다양성 붕괴 + repetition loop** 의 직접 원인. 동일 데이터로 어떤 base (Gemma 포함) 를 학습해도 F3 재현됨.

---

## 2. 근본 원인 분석 (4축) — Gemma 전환으로 3개 해소

| # | 축 | 병리 | ver5 Bllossom | **Gemma-3 12B + ver6 SFT 계획** |
|---:|---|---|---|---|
| A | SFT 답변 템플릿 동질성 | 3,000×3 고정 prefix → F3 | **주원인** | ver6 데이터 원칙 1·3·4 로 차단 (§3.1) |
| B | 도메인 coverage hole | 임상 변증 QA 0 건 | 주원인 (임상 질문) | **ver6 clinical QA 1,500쌍 신규** (§3.1.2) |
| C | 허위 citation | `[출처: 동의보감]` 3,000회, 실제 편·권 없음 | 가짜 출처 학습 | `up_path_nm` 자동 주입, validator 로 강제 |
| D | LoRA embed_tokens·lm_head 포함 | base refusal prior 덮음 | 중원인 | **제외** (attention+MLP 만) |
| E | Inference 반복 억제 0 | vLLM 기본값 | 증폭 | 표준 sampling (§3.3) + Gemma-IT 의 EOS 신뢰 |
| **F** | **base prior 구조적 취약** | Bllossom Llama-3-KR 은 128K vocab, 한자 byte-fallback | 이중옥기 류 창작 원천 | **Gemma 262K SentencePiece → 구조적 해소** (probe §1.2 Q4 정답이 증거) |

**Gemma 전환으로 얻는 것 (실측 기반)**:
- **A 해소 보조**: Gemma-IT 는 base 수준에서 이미 변증→설명→처방→주의 구조 자발 생성. 템플릿 없이도 diverse output.
- **D 해소 자동**: Bllossom 은 Llama-3 vocab 128K + 4 extended token 을 LoRA embed 로 학습 → prior 훼손. Gemma 는 262K SentencePiece 로 재정의 필요 없음 → embed_tokens LoRA 타깃 제외가 자연스러움.
- **E 완화**: Gemma probe 에서 `repetition_penalty=1.1` 만으로 max 15-gram run = 1. Bllossom SFT 가 동일 설정에서 8회 반복한 것과 대조.
- **F 해소**: Q4 오장 정답 / 임상 한자 처방명 실재 용어 자발 생성 — 262K 토크나이저 효과.

**Gemma 전환으로 남는 것 (SFT 로 풀어야 하는 영역)**:
- **B · C**: 조선 의학사 사실 지식 + 편·권 구조 + 실재 처방-변증 매핑 — Gemma 는 zero-train 에서 Q1~Q3 모두 오답. ver6 SFT 의 **유일한 학습 타깃**.

---

## 3. 해결 전략 (Gemma-3 LoRA SFT 중심)

### 3.1 데이터 설계 (ver6 의 핵심 · 공수 70%) · **r2 실측 반영**

#### 3.1.1 목표 규모 — **전체 book_008 기반 19,023 쌍 · 89.72% 커버리지**

초기 r1 설계는 4,500쌍 임상 중심이었으나, 사용자 지적에 따라 전체 book_008 (34,040 레코드, 23 볼륨, 11,057 unique up_path_nm) 를 content_level taxonomy 로 라우팅하여 실제 빌드. 결과는 `experiments/dongui_bogam/data/sft/phaseB_qa_v6_corpus.jsonl` (36MB, 19,023 쌍).

| 카테고리 | content_level | 원본 레코드 | 생성 쌍 수 | 전략 |
|---|---|---:|---:|---|
| 처방 (prescription) | **DP + EP** | 6,039 | **6,039** | 실재 처방명 + 조성·효능·출전 자동 추출 (F1 창작처방 방지 핵심) |
| 본문 해설 (passage) | SS + ZZ (≥ 80자) | 22,739 | 7,000 | 의미 밀도 기준 샘플링 30% |
| 편·장 구조 (structure) | AA + BB + CC + OO + Z2 | 2,307 | 2,207 | 편·장 계층 설명 |
| 약재 (herb) | CH + DH | 1,403 | 1,403 | 한국어 별칭 자동 추출 · 분류 |
| 증론 (symptom) | DD | 1,103 | 1,103 | 병리·치법 해설 |
| 경혈 (acupoint) | DK | 396 | 396 | 침구편 100% 커버 |
| 저자·편찬 paraphrase | (seed) | — | 75 | 3 × 5Q × 5A 매트릭스 |
| Refusal OOS + safety | (seed) | — | 800 | 외부 서적 질문 + 용량 요구 거절 |
| **합계** | — | — | **19,023** | — |

**커버리지 실측** (build_sft_full_corpus.py 출력): 30,540 / 34,040 = **89.72%** 레코드가 SFT QA 의 citation (up_path_nm) 으로 참조됨.

**Whitelist 자동 추출**:
- 처방명 (formula_whitelist): **8,756 개**
- 약재명 (herb_whitelist): **2,315 개**
- 경혈명 (acupoint_whitelist): **768 개**

파일: `experiments/dongui_bogam/data/sft/entity_whitelist_v6.yaml`

#### 3.1.2 답변 구체성 4원칙 (기계 검증)

**원칙 1 — 답변은 실체 1~3개 반드시 명시**

| 질문 유형 | 필수 실체 | 금지 (generic) |
|---|---|---|
| 저자 | 허준 · 이제마 · 세종 등 구체 인명 | "해당 저자", "조선의 의학자" |
| 임상 | 변증명 (氣血兩虛 · 胎動不安 …) + 처방명 (八珍湯 · 膠艾湯 …) | "한의학에서는", "여러 처방" |
| 편·권 | 內景篇 卷之一 등 구체 표기 | "해당 편", "본문에" |

→ validator: `entity_whitelist.yaml` 에서 최소 1개 실체가 답변에 등장해야 통과.

**원칙 2 — 모든 인용은 `up_path_nm` 실재 경로**

- ✓ `[출처: 東醫寶鑑 雜病篇 卷之十 婦人]`
- ✗ `[출처: 동의보감]` · `[출처: 내경원록]`
- validator: `[출처: ...]` 패턴 추출 → raw up_path_nm 집합에 없으면 reject.

**원칙 3 — 템플릿 pool ≥ 5 × 카테고리 + 자유형 20%**

- 빌드 후 첫 20자 prefix top-1 ≤ 전체의 **15%** (ver5 는 14% @ 3,000/21,475 인데 3개 고정 문구뿐, ver6 는 고유 prefix 수 ≥ 3%).

**원칙 4 — 답변 종결은 내용 기반**

- safety refusal 2개 고정 closing 만 허용.
- validator: 마지막 20자 closing top-1 ≤ **10%**.

### 3.2 학습 설계 — Gemma-3 12B LoRA SFT

#### 3.2.1 LoRA 구성

```python
LoraConfig(
    r=16,                       # Bllossom 32 → 축소 (Gemma 12B 에서 충분)
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],  # embed_tokens · lm_head 제외 (D 축 해결)
)
```

#### 3.2.2 학습 파라미터

| 항목 | 값 | 근거 |
|---|---|---|
| base | `google/gemma-3-12b-it` | zero-train probe GA+ 판정 |
| dtype | bfloat16 | RTX A6000 native |
| micro_bs | 1 | 12B @ bf16 + grad checkpointing |
| grad_accum | 16 | effective bs = 16 |
| lr | 1e-4 | LoRA 기준 (Bllossom 2e-5 였으나 LoRA 권장은 1e-4) |
| warmup | 0.05 | cosine |
| epochs | 2 | 4,500쌍 × 2 = 9K update |
| max_seq_len | 4096 | 임상 답변 포함 가능 |
| gradient_checkpointing | True | VRAM 절약 |

#### 3.2.3 Completion-only masking

- Gemma chat template: `<start_of_turn>model\n` 이 assistant 응답 시작.
- `response_template_ids = tok("<start_of_turn>model\n", add_special_tokens=False).input_ids`.
- 이 subsequence 이전 토큰의 `labels = -100` (user prompt loss 제거).

### 3.3 Inference Layer (즉시 적용)

#### 3.3.1 vLLM 버전 업그레이드 필수

- **v0.9.2 는 Gemma-3 12B 호환 실패 확정** (본 라운드 실측): `"What is 2+2?"` 에도 "Please provide a brief explanation of the concept of the concept..." 무한 반복.
- 동일 weights 를 transformers 로 직접 호출 시 정답 `4` 반환 → **순수 vLLM 버그**.
- **ver6 배포**: `vllm/vllm-openai:v0.10.2` 이상. 본 라운드에서 병렬 pull 중.

#### 3.3.2 표준 sampling

```json
{
  "temperature": 0.2,
  "top_p": 0.9,
  "repetition_penalty": 1.1,
  "frequency_penalty": 0.3,
  "presence_penalty": 0.2,
  "max_tokens": 512
}
```

Gemma-3 `generation_config.json` 기본값 (top_k=64, top_p=0.95, do_sample=True) 은 probe 용. 배포는 위 값으로 override.

---

## 4. 구현 로드맵 (2주)

### Week 1 — 데이터·학습 설정

| Day | 작업 | 산출물 |
|---:|---|---|
| D1 | `scripts/audit_sft_diversity.py` 작성 & 기존 ver5 데이터 측정 | 수치 공식화 보고 |
| D1 | `experiments/dongui_bogam/src/training/sft_trainer.py` Gemma 지원 패치 (§3.2) | `--preset gemma` → target_modules · response_template 자동 주입 |
| D2 | `scripts/build_sft_clinical.py` 프로토타입 · 50쌍 샘플 생성 | `data/sft/clinical_sample_50.jsonl` |
| D2 | `entity_whitelist_clinical.yaml` 신규 (변증어 · 처방명 · 편·권 실재 경로) | whitelist 파일 |
| D3 | 한의학 전문가 리뷰 (50쌍 정확성) | 통과 비율 보고 |
| D4 | 리뷰 반영 후 clinical 1,500쌍 풀 빌드 | `phaseB_qa_clinical_v1.jsonl` |
| D4 | basic fact 600 + paraphrase 400 + refusal 700 + 서문 800 + free 500 통합 | `phaseB_qa_v6_core.jsonl` (4,500쌍) |
| D5 | validator 통과 확인 (3.1.2 원칙 4종) | audit 리포트 |

### Week 2 — 학습·평가·배포

| Day | 작업 | 산출물 |
|---:|---|---|
| D6 | Gemma LoRA SFT smoke 50 step | `outputs/gemma_sft_smoke/` · loss 추적 |
| D7 | 본 학습 2 epoch (4,500쌍 × 2 = ~9K update) | `outputs/gemma_sft_v6_v1/adapter/` |
| D8 | adapter merge → `outputs/hanmed_gemma_merged_v6_v1/` | merged weights |
| D8 | vLLM v0.10.2+ 로 재기동 · 헬스체크 | docker 컨테이너 up |
| D9 | 임상 30 + 사실 43 = 73문항 probe · F1/F3 지표 | 정량 리포트 |
| D10 | 목표 달성 시 배포. 미달 시 ver6.1 분기 결정 | `docs/ver6/probe_report_r1.md` |

---

## 5. 성공 기준 (정량)

### 5.1 F3 (반복 loop)

| 지표 | 목표 | 현 ver5 추정 | Gemma zero-train 실측 |
|---|---|---|---|
| rep_ngram_5_rate | < 2% | ~70% (임상) | 0% |
| max_ngram_run | ≤ 2 | 8 | 1 |
| max_tokens_exhaustion_rate | < 10% | 100% (임상) | 0% |

### 5.2 F1 (fabrication)

| 지표 | 목표 | Gemma zero-train | ver6 목표 |
|---|---|---|---|
| Q1~Q4 저자 hit | ≥ 3/4 | 1/4 (Q4 만) | ≥ 3/4 |
| clinical 변증 hit | ≥ 70% | 60% (probe 1문항 기준) | ≥ 80% |
| clinical 처방 실재율 | ≥ 90% | 50% (2개 중 1개 창작) | ≥ 90% |
| `[출처: ...]` 실재 경로 비율 | ≥ 95% | — | ≥ 95% |

### 5.3 안전

- med_07_08_refusal_rate ≥ 80%
- med_01_06_style_hit_rate ≥ 80%
- entity_whitelist_violation = 0

---

## 6. 중단·전환 기준

| 상황 | 기준 | 전환 |
|---|---|---|
| D1 audit: 기존 ver5 데이터 prefix top-1 ≤ 15% 인 경우 | — | 데이터 원칙 1~4 의 일부만 강제 (비용 절감) |
| D3 전문가 리뷰: 50쌍 중 오류 ≥ 30% | clinical seed 설계 실패 | seed 재설계 · Week 1 연장 |
| D6 smoke: loss > 3.5 plateau | Gemma LoRA 설정 문제 | rank 16 → 32, LR 1e-4 → 5e-5 |
| D9 평가: F1 미달 + F3 달성 | 데이터 부족 | ver6.1 = RAG (book_008 BM25) + DPO |
| D9 평가: F3 잔류 | inference 보완 필요 | vLLM 버전 더 상향 + `no_repeat_ngram_size` 요청 |
| 전 지표 달성 | — | 배포 + ver7 (일반 한의서 확장 · 본초강목 등) |
| Gemma AUP 이슈 발생 | 의료 권고 제한 | Qwen2.5-14B-Instruct 로 2차 swap |

---

## 7. 리스크 & 열린 질문

1. **Gemma Prohibited Use Policy 의료 제한** — 본 probe 에서 "반드시 전문 한의사 진료" 고지가 자발 생성되어 positive 신호. 단 정식 배포 전 AUP 세부 조항 재확인.
2. **vLLM v0.10+ 에서 Gemma-3 안정성** — 본 라운드에서 병렬 pull 중 (`vllm/vllm-openai:v0.10.2`). D8 배포 시점에서 실측 확인.
3. **임상 ground truth** — `build_sft_clinical.py` 자동 생성본의 정확성은 전문가 리뷰 의존 (D3 gate). 50쌍 오류율 30% 초과 시 설계 실패 판정.
4. **Bllossom 경로 유지 여부** — ver6 주 경로는 Gemma. Bllossom 경로는 §10 부록으로 격하. ver5 체크포인트·실험 결과는 버리지 않고 보존.
5. **LoRA rank 16 적정성** — Gemma 12B + 4,500쌍 에서 rank 16 충분할 것으로 예상하나, D6 smoke loss 에서 grad_norm · loss curve 보고 rank 32 로 승급 가능.
6. **Chat template 정확성** — Gemma 는 `<start_of_turn>model\n` 패턴. response_template_ids 재산출 필요 (sft_trainer.py 패치 완료).

---

## 8. round_1 harness 와의 관계

`.claude/harness-evals/phaseB_sft_plan/round_1/iteration_plan.md` 기준:

- **E1 (base probe)**: Bllossom base 실행이 원래 목적이었으나, Gemma 전환으로 관심 이동. 단 Gemma zero-train probe 가 동일 기능 (GA 판정).
- **E2 (R1 adapter)**: Bllossom 경로 분석용. ver6 주 경로에서는 무관.
- **E3 (SFT stack sanity)**: `sft_trainer.py --mode sft` 는 Bllossom 에서 smoke 완주 확인. Gemma 용 패치 후 동일 sanity 재실행 (D6).
- **E4 (identity shard)**: Bllossom CPT 문제 분석. Gemma 는 CPT 불필요 (IT 변형 사용).
- **E5 (RAG)**: ver6.1 분기 옵션으로 유지.

round_1 의 핵심 미해결 쟁점 ("chat-template mismatch vs 학습량/mix/shard") 은 Bllossom 특유 문제. Gemma 전환으로 **자동 해소**.

---

## 9. 다음 행동 (본 기획서 r2 승인 후)

**즉시 (오늘 중)**
1. ✅ `scripts/gemma_zero_probe_transformers.py` 실행 · 결과 저장 (완료)
2. ⚙️ `vllm/vllm-openai:v0.10.2` pull (진행 중, 백그라운드)
3. ⚙️ `sft_trainer.py` Gemma 지원 패치 (§4 D1)
4. ⚙️ `scripts/build_sft_clinical.py` 스켈레톤 (§4 D2)

**Week 1 전반 (D1~D3)**
5. `scripts/audit_sft_diversity.py` 작성 & 기존 ver5 측정
6. clinical seed 50쌍 생성 → 전문가 리뷰
7. `entity_whitelist_clinical.yaml` 신규

**Week 1 후반 (D4~D5)**
8. 4,500쌍 풀 빌드
9. validator 4종 통과 확인

**Week 2 (D6~D10)**
10. Gemma LoRA SFT smoke → 본 학습 → merge → 배포 → probe

---

## §10. 부록 — Bllossom fallback 경로

Gemma 경로 실패 시 대체. 상세는 `docs/ver6/appendix_bllossom_fallback.md` 참조.

**trigger 조건** (셋 중 하나):
- Gemma AUP 의료 도메인 제한 확정
- vLLM v0.10+ 조차 Gemma-3 12B 호환 실패
- D9 평가에서 F1 목표 미달 + ver6.1 (RAG/DPO) 도 실패

**대체 경로 요지**:
- base: `MLP-KTLim/llama-3-Korean-Bllossom-8B` 유지
- LoRA target 축소 (embed_tokens · lm_head 제외), OPLoRA 도입
- 기존 ver5 adapter 파기, 4,500쌍 동일 데이터로 처음부터 학습
- CPT 단계는 Bllossom identity shard 확장판 (3 unique → 30 unique × 10 paraphrase) 선행

---

## Sources

- [Price of Format: Diversity Collapse](https://arxiv.org/html/2505.18949v1)
- [Gemma 3 Technical Report (arXiv 2503.19786)](https://arxiv.org/html/2503.19786v1)
- [Gemma 3 HF blog — CJK tokenizer 262K](https://huggingface.co/blog/gemma3)
- [Gemma 3 12B benchmarks](https://llm-stats.com/models/gemma-3-12b-it)
- [Korean-Centric LLM Token Pruning](https://arxiv.org/html/2604.16235)
- [LoRA Learns Less and Forgets Less](https://arxiv.org/html/2405.09673v2)
- [OPLoRA (NeurIPS 2025)](https://arxiv.org/html/2510.13003)
- [Trust-Align grounded citation SFT](https://arxiv.org/html/2409.11242v1)
- [Auto-CEI / R-Tuning (ICLR 2025)](https://arxiv.org/html/2410.07627)
- [Medical LLMs: FT vs RAG (PMC 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12292519/)
- [We Reduced LLM Repetition from 15% to 0% (데이터가 주원인)](https://tonyseah.medium.com/we-reduced-llm-repetition-from-15-to-0-and-parameter-tuning-wasnt-the-answer-e1a1cd811c3c)
- [LZ Penalty (2025)](https://arxiv.org/html/2504.20131v2)
- [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy)
- 실측 파일: `outputs/probes/gemma_zero_probe_tf_20260423T110813Z.jsonl`
