# 10.3 Adapter 경로 — CPT-only vs CPT+SFT

> **사용자 지적 (R3.3)**: "자기지도 학습, DAPT 를 쓰는데 왜 SFT 지?"
>
> 이 섹션은 논문 기여 축(DAPT/CPT) 과 데모 UX(instruction following) 사이의 긴장을 명시적으로 해소한다.

## 문제 정의

| Stage | 목적 | 학습 방식 | 데모 요구사항과의 관계 |
|---|---|---|---|
| Stage 1 CPT | 도메인 adapt (한의학 고전 knowledge injection) | self-supervised (next-token prediction on domain corpus) | knowledge 는 배우지만 **instruction following 은 직접 학습 안 함** |
| Stage 2 SFT | instruction following | supervised (prompt-response 쌍) | 데모 UX 직접 제공하지만 **별도 데이터 curation + 새 기여 축** |

ver2 논문 primary 기여 = **Stage 1 CPT 레시피**. 데모에서 Stage 2 SFT 필수로 두면 논문 scope 가 CPT + SFT 혼합 → 기여 축 분산.

## 해결: 2가지 경로 분리

### P-CPT (primary, ver2 데모 기본)

**구성**: `Bllossom-8B base` + `HanMed-CPT LoRA`

**Instruction following 은 어디서 오는가**: Bllossom-8B = Llama-3.1-8B-Instruct 위의 2차 한국어 튜닝. 원본 Instruct 의 chat template (`<|start_header_id|>`, `<|eot_id|>`) 생성 능력은 base logit 에 encode 되어 있고, LoRA additive delta 가 이를 완전히 덮지는 않을 가능성이 높음. **R3.4 중요 보정**: 이 "보존" 주장은 §04a §C.5 하이퍼파라미터 표 (`prompt format: plain text (Stage 1) — ChatML 은 Stage 2`) 를 근거로 하되, **Stage 1 corpus 에 Llama-3 chat special token 이 포함되지 않으므로 CPT 가 chat 분포를 직접 덮지 못할 뿐**이며, LoRA delta 가 chat header adjacent token 에 **간접적으로 영향을 미칠 가능성은 남는다**. 따라서 **M2 H1 실측 gate 로 반드시 검증**.

**§05 T5 regression gate (KLUE-YNAT drop ≤ 3%p) 는 일반 한국어 ability 측정이지 chat template 생성 능력 측정 아님** (판별자 R3.3 지적). chat template preservation 은 별도 측정 필요 (§10.9 E6 참조).

**메커니즘**:
1. User: `"인삼의 성미에 대해 설명해줘"`
2. CLI 가 Llama-3 chat template 으로 wrap:
   ```
   <|begin_of_text|><|start_header_id|>system<|end_header_id|>
   {HanMed system prompt}<|eot_id|>
   <|start_header_id|>user<|end_header_id|>
   인삼의 성미에 대해 설명해줘<|eot_id|>
   <|start_header_id|>assistant<|end_header_id|>
   ```
3. Bllossom-8B + CPT LoRA 가 decode. chat header 인식은 **base Bllossom 의 Instruct 역량**, 한의학 내용은 **CPT adapter** 에서 출력.

**장점**:
- 논문 기여 축 (CPT) 과 데모가 **직접 일치** — CPT 의 knowledge injection 성공 여부가 데모 품질로 바로 드러남
- Stage 2 SFT 데이터 curation (수천~수만 prompt-response 쌍, §04.6.1) 불필요 → M2 scope 축소
- RAG 비사용 결정과 일관 (knowledge 는 CPT adapter 가 책임)

**리스크**:
- CPT 가 chat format 을 망가뜨리면 데모 동작 불가 → §04a §C.5 하이퍼파라미터 표 (R3.4 정정: C.3 아님) 에 "Stage 1 은 plain text" 명시. 실측은 **M2 H1** 으로 이연 (§10.9 E6)
- CPT knowledge 가 약하면 답변이 "동의보감에는..." 같은 일반 수준에 머물 수 있음 — §E ablation 으로 실증

**데모 동작 예시 (P-CPT 가정)**:
```
[user] 인삼(人蔘)의 성미와 귀경 알려줘
[hanmed]
  인삼은 맛이 달고 약간 쓰며, 성질이 약간 따뜻합니다.
  비경(脾經)과 폐경(肺經)에 귀경하며,
  대보원기(大補元氣) 작용이 있어 원기 쇠약에 주로 쓰입니다.
  — 동의보감 탕액편 권1
```

### P-SFT (옵션, v1 이후)

**구성**: `Bllossom-8B base` + `HanMed-CPT LoRA (merged)` + `HanMed-SFT LoRA`

**언제 사용**:
- SFT 데이터 curation 완료 (§04.6.1, ≥ 2,000 instruction-response 쌍 + human-in-the-loop 검수 20%)
- ver2 논문 제출 이후 v1 확장
- 사용자 체감 품질 (helpful, format-following) 중요한 배포

**장점**:
- format 일관성 (출처 표기, 한문 인용 block 형식 등) 이 SFT 로 직접 학습되어 안정적
- §05 T1~T4 평가 점수 전반 향상 (특히 T2 QA 정확도)
- helpfulness 향상

**단점**:
- SFT 합성 데이터 hallucination 리스크 (§04.6.3) — `synthesis_provenance` 필드로 추적 필요
- 논문 scope 확장 — ver2 범위 밖

## 경로별 의존성

| 요소 | P-CPT | P-SFT |
|---|---|---|
| Bllossom-8B base | ✅ | ✅ |
| Stage 1 CPT adapter | ✅ | ✅ (merged) |
| Stage 2 SFT adapter | ❌ | ✅ |
| SFT dataset curation (§04.6.1) | ❌ | ✅ |
| T5 regression green (§05) | ✅ 필수 | 권장 |
| T4 refusal layer | ✅ (외부 wrapping) | ✅ (외부 wrapping) |
| Chat template | **base Bllossom** 의존 | SFT 가 학습한 format |
| KIOM 라이선스 | CPT 공개 adapter 범위 (§07) | 동일 |

## CLI 실행 인자

```bash
# P-CPT (기본)
hanmed chat --adapter outputs/cpt_bllossom/adapter

# P-SFT (옵션)
hanmed chat --adapter outputs/sft_bllossom/adapter --mode sft
```

내부적으로 `--mode sft` 는:
1. CPT adapter 를 base 에 merge (`peft_model.merge_and_unload()`)
2. SFT adapter 를 merged base 에 적용
3. Chat template 을 SFT 가 학습한 format 으로 전환 (필요 시)

## 검증 (P-CPT 가 실제로 동작하는지)

§E ablation 착수 시 다음을 함께 측정. **R3.5**: 간단 sanity test 와 exit gate 를 분리한다. 아래 1행은 `10.9 E6` 의 정식 gate 를 축약 인용한 것이다.

| 테스트 | 통과 기준 |
|---|---|
| Llama-3 chat template preservation | `10.9 E6` 준수: 200 generic multi-turn prompt 자동평가, `ΔEOT-rate < 2%p vs base`, 3 seed variance 보고. 10개 수동 prompt 는 sanity only |
| Instruction following (generic) | KLUE-YNAT T5 drop ≤ 3%p (§05) |
| Instruction following (domain) | T2 QA chrF ≥ baseline (no-CPT) + 1.0 |
| Format stability | "다음을 JSON 으로 답해라" 류 format prompt 10개에서 base 대비 format 유지율 degradation 없음 |

모두 green → P-CPT 로 데모 충분. 하나라도 red → SFT 경로 or 학습 수정.

## 결론

- **ver2 논문 데모 = P-CPT 고정**
- **P-SFT 는 v1 이후 or 기여 축이 다른 후속 논문**
- RAG 는 별도 연구 (v2 이후)
