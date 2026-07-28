# 06. 중의학 LLM 의 adapter / vocab extension 관행 조사

> 작성: 2026-04-21
> 질문: "`modules_to_save=[embed_tokens, lm_head]` + LoRA + vocab extension" 조합이
> **중의학 LLM 커뮤니티에서 대다수 관행인가?**
> 답: **아님**. 대다수는 vocab extension 자체를 안 한다. 우리 HanMed 가 특이 교차점.

---

## 1. 핵심 결론 (먼저)

| 항목 | TCM LLM 대다수 | HanMed (우리) |
|---|---|---|
| Base | **중국어 native** (Baichuan2, Qwen, Ziya-LLaMA) | **한국어 native** (Bllossom-8B = Korean Llama-3) |
| Vocab extension | **거의 안 함** | **함** (128,256 → 128,260, 4 special tokens) |
| Special tokens | **사례 극소** | 한↔중 bilingual 구조 마커 4개 |
| 학습 경로 | CPT + SFT (기본) / LoRA (65% in scoping review) | CPT (Stage 1, 현재) |
| LoRA target | q_proj + v_proj (minimal) ~ q/k/v/o (중간) | **q/k/v/o + gate/up/down + embed_tokens + lm_head** |
| embed/lm_head full-train / LoRA target 포함 | **공개 사례 거의 없음** | 함 (§5.3 ver4) |

**요지**: 중국에서 만드는 TCM LLM 들은 base 를 이미 중국어가 강한 모델로 골라서 "언어 문제"가 없으므로 **vocab 확장을 아예 안 하는 게 관행**. 따라서 `modules_to_save=[embed_tokens,lm_head]` 도 불필요. HanMed 는 **한국어 base + 중국어/한중 bilingual corpus** 라는 이중 언어 교차점이라 vocab extension 이 등장한 것이고, 이 패턴의 참고 모델은 **TCM 쪽이 아니라 Chinese-LLaMA-Alpaca (Cui et al. 2023, arXiv:2304.08177) 같은 "언어 레벨 vocab extension" 계열**이다.

---

## 2. 조사된 TCM LLM 구성 (공개 paper 기준)

| 모델 | Base | 주요 학습 | Vocab 확장 | LoRA target | 출처 |
|---|---|---|---|---|---|
| **HuatuoGPT-II** (2023) | Baichuan2-7B/13B | **Full FT**, one-stage (CPT+SFT 통합) | 명시 없음 | — (LoRA 안 씀으로 보임) | [arXiv:2311.09774](https://arxiv.org/abs/2311.09774) |
| **BianCang** (2024) | Qwen-2/2.5 | CPT + SFT (2-stage) | 명시 없음 | 명시 없음 | [arXiv:2411.11027](https://arxiv.org/abs/2411.11027) |
| **TCMChat** (2024) | Baichuan2-7B-Chat | CPT + SFT | 명시 없음 | 명시 없음 | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1043661824004754) |
| **Zhongjing** (2023) | Ziya-LLaMA-13B | CPT + SFT + RLHF | 명시 없음 | 명시 없음 | [arXiv:2308.03549](https://arxiv.org/abs/2308.03549) |
| **BenTsao / Huatuo** (2023) | LLaMA-7B + Alpaca-Chinese | LoRA only (instruction tuning) | 없음 (LLaMA tokenizer 그대로) | **q_proj, v_proj** (최소) | [arXiv:2304.06975](https://arxiv.org/abs/2304.06975) |
| **ShenNong-TCM** (2023) | LLaMA-7B | LoRA fine-tune (11만 instruction) | 명시 없음 | 명시 없음 | [HF: michaelwzhu/ShenNong-TCM-LLM](https://huggingface.co/michaelwzhu/ShenNong-TCM-LLM) |
| **Jingfang** (2025) | Qwen2.5-7B-Instruct | LoRA (rank=64) + multi-agent | 없음 | 명시 없음 | [arXiv:2502.04345](https://arxiv.org/abs/2502.04345) |
| **MedChatZH** (2024) | LLaMA (Chinese tuned) | CPT + SFT | 명시 없음 | 명시 없음 | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0010482524003743) |
| **TCM-GPT** (2024) | (Chinese base) | 도메인 적응 CPT | 명시 없음 | 명시 없음 | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2666990024000259) |

**Scoping review 2025** (TCM LLM 27편 대상, PMC12922203): LoRA **65.2%**, prompt engineering 47.8%, CPT **43.5%**, RAG 39.1%. LoRA 가 지배적이지만 **target module 공개 사례 적고 q_proj/v_proj 수준이 다수**.

### 이 표에서 드러나는 3가지 패턴

1. **Base 선택 패턴**: 거의 모두 Baichuan / Qwen / Ziya-LLaMA (chinese-native). Llama / Mistral 같이 영어-default base 에 중국어 덧붙이는 경우는 BenTsao / ShenNong 초기 LLaMA-7B 세대뿐이고 이들도 Alpaca-Chinese 토크나이저를 그대로 씀 (즉 **기존 확장된 토크나이저 재사용**, 본인이 추가 확장 안 함).
2. **Vocab / Special token 추가**: 공개 논문·리포지토리 범위에서 TCM LLM 이 **자체적으로 vocab extension 을 수행한 사례를 찾지 못함** (unverified for unpublished models). 중의학 용어는 base 의 BBPE 가 중국어 이미 잘 처리하므로 굳이 할 이유가 없음.
3. **embed_tokens / lm_head 를 full-train 하거나 LoRA target 으로 포함한 공개 사례**: 없음 (조사 범위 내). `modules_to_save` 라는 옵션이 등장하는 공개 TCM LLM 소스 미발견.

---

## 3. "그러면 HanMed 의 구성은 어디서 온 패턴인가"

**Chinese-LLaMA-Alpaca** (Cui et al. 2023, [arXiv:2304.08177](https://arxiv.org/abs/2304.08177)) 의 vocab extension 절차와 **정확히 동형**.

Chinese-LLaMA Stage 2 에서:

> "LoRA weights are added to the attention mechanisms and **the embeddings, LM heads, and newly added LoRA parameters are trained**"

HanMed 의 `modules_to_save=["embed_tokens","lm_head"]` (원안) 또는 현재의 `target_modules += ["embed_tokens","lm_head"]` (B 안) 은 이 문장을 그대로 구현한 것. 즉 **TCM 관행이 아니라 "언어 레벨 vocab extension" 관행**.

이 계열에서 우리와 구조가 유사한 선례:
- **Chinese-LLaMA-Alpaca** — LLaMA vocab 32K → 49,953 (+17,953 Chinese tokens), 2-stage: Stage 1 embedding only, Stage 2 LoRA + embed/lm_head trainable.
- **Bllossom 자체** — Llama-3 을 Korean 으로 확장. Bllossom repo 의 "vocab-expansion version" 버전이 존재 (단 저자는 최종 릴리즈에서 vocab-expansion 버전을 **철회**하고 **non-expansion + 250GB 데이터** 경로를 택했다고 HF README 에 기록). Bllossom 팀 자체가 vocab extension 의 이점이 제한적이라 판단한 신호.

**암시**: 4 개만 추가하는 소규모 vocab extension 이 HanMed 의 목적(한↔중 bilingual 구조 마커)에 진짜 가치가 있는지 재검토 여지가 있음. Bllossom 팀이 large-scale vocab extension 도 걷어낸 판단을 참고하면, 4-token 추가가 **정말 필수인지** 혹은 **기존 vocab 조합으로 우회 가능한지** 검토할 가치 있음. 별개 결정 사항.

---

## 4. "대다수 관행"이 우리에게 시사하는 것

### 4.1 규모로 본 대세

- 27편 scoping review 중 LoRA 65.2% / CPT 43.5% → **LoRA + CPT 조합이 TCM LLM 표준**
- 우리도 이 조합은 따르고 있음 (맞음)

### 4.2 우리 대비 차이가 나는 지점

- 대다수 TCM LLM: LoRA 를 **attention (q/v) 정도만** 씀 → trainable ~0.1% 수준
- HanMed: LoRA 7 modules + embed/lm_head → **1.14%** (B안 기준 92.4M)
- 차이의 이유: 우리가 한국어 base → 중의학(대부분 중국어) 도메인 적응을 해야 해서, LoRA 로 덮어야 하는 weight 가 더 많음. **단순히 과하게 잡은 게 아니라 문제 구조상 정당화 가능**.

### 4.3 참조 사례 재정렬 (우리가 따라갈 만한 구성)

1. **Chinese-LLaMA-Alpaca** (arXiv:2304.08177) — 가장 가까운 구조. 대형 vocab extension + 2-stage.
2. **HuatuoGPT-II** (arXiv:2311.09774) — one-stage CPT+SFT 통합은 HanMed §C.5 (CPT) 다음 단계 (§D SFT) 설계에 참고.
3. **BianCang** (arXiv:2411.11027) — Qwen-2 기반 TCM 2-stage, 학습 데이터 구성 (ChP-TCM) 참고.
4. **Bllossom 자체** — vocab extension 의 제한된 이점에 대한 경고로 읽을 것.

---

## 5. 권고 (결정 아님, 참고용)

1. **현재 B 안 (LoRA on embed/lm_head) 유지**는 중국어-native base 를 쓰는 TCM LLM 관행과는 다르지만, **Chinese-LLaMA-Alpaca 관행**에는 정확히 부합. 논문 정당화는 "한국어 base + 중국어 도메인 코퍼스"라는 이중 언어 설정에서 new token 을 학습시키기 위함 으로 가능. 근거: arXiv:2304.08177 §3.2.
2. **향후 TCM LLM 대조 표**를 논문/보고서에 삽입 시 위 표를 그대로 쓰되, HanMed 를 "Korean base + TCM domain" 교차점으로 별도 행에 표기하는 게 가장 정확한 포지셔닝.
3. **special token 4개 자체가 필요한지** 재검토는 선택사항. Bllossom 의 non-expansion 경로 선택은 "vocab extension 이 데이터 양으로 상쇄될 수 있다"는 약한 방증. 우리 경우엔 bilingual 구조 마커로 쓰이므로 성격이 다르지만, "자연어 prefix (`<KO>`, `<ZH>`) + 기존 vocab"으로 우회 가능 여부는 검토 가치가 있음.

---

## 6. 조사 범위 한계 (과장 금지)

- 표 대상은 **공개 논문 + 공개 repo** 기준. 내부/미공개 상업 모델 (예: 중의학 특화 상용 모델) 은 조사 범위 밖 (`unverified`).
- LoRA target module 명시가 대다수 TCM paper 에서 누락되어 있어 "q_proj/v_proj 만 씀"이 확증이라기보단 BenTsao 한 편의 구체 수치에 근거한 합리적 추정.
- Scoping review 의 65.2% 수치는 논문 채택 여부 기준이고, 각 논문의 LoRA 구성 상세는 추가 검토 필요.
- 본 조사는 1 세션 web search 범위. 완전 정량 review 아님.

---

## 7. 참고문헌

### TCM LLM
- HuatuoGPT-II — [arXiv:2311.09774](https://arxiv.org/abs/2311.09774)
- BianCang — [arXiv:2411.11027](https://arxiv.org/abs/2411.11027), [GitHub QLU-NLP/BianCang](https://github.com/QLU-NLP/BianCang)
- TCMChat — [ScienceDirect S1043661824004754](https://www.sciencedirect.com/science/article/pii/S1043661824004754), [GitHub ZJUFanLab/TCMChat](https://github.com/ZJUFanLab/TCMChat)
- Zhongjing — [arXiv:2308.03549](https://arxiv.org/abs/2308.03549)
- BenTsao / Huatuo — [arXiv:2304.06975](https://arxiv.org/abs/2304.06975), [GitHub SCIR-HI/Huatuo-Llama-Med-Chinese](https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese)
- ShenNong-TCM-LLM — [HF michaelwzhu/ShenNong-TCM-LLM](https://huggingface.co/michaelwzhu/ShenNong-TCM-LLM)
- Jingfang — [arXiv:2502.04345](https://arxiv.org/abs/2502.04345)
- TCM-GPT — [ScienceDirect S2666990024000259](https://www.sciencedirect.com/science/article/pii/S2666990024000259)
- MedChatZH — [ScienceDirect S0010482524003743](https://www.sciencedirect.com/science/article/abs/pii/S0010482524003743)
- TCM-FTP — [arXiv:2407.10510](https://arxiv.org/abs/2407.10510)
- TCM-R1 — [Springer chapter](https://link.springer.com/chapter/10.1007/978-981-95-5640-3_21)
- Scoping review 2025 — [PMC12922203](https://pmc.ncbi.nlm.nih.gov/articles/PMC12922203/)
- Chinese medical LLM review — [arXiv:2509.18690](https://arxiv.org/html/2509.18690v1)

### 언어 레벨 Vocab extension (우리의 진짜 계보)
- Chinese-LLaMA-Alpaca — [arXiv:2304.08177](https://arxiv.org/abs/2304.08177)
- Bllossom-8B — [HF MLP-KTLim/llama-3-Korean-Bllossom-8B](https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B) (vocab-expansion 경로 철회 기록)

### 기반 모델
- Baichuan2 — (HuatuoGPT-II, TCMChat 의 base)
- Qwen / Qwen-2.5 — (BianCang, Jingfang 의 base)
- Ziya-LLaMA — (Zhongjing 의 base)
