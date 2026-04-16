# 06. Infrastructure — GPU · 프레임워크 · 재현성 (ver2)

> ver1 대비 변경: 메모리 예산에 DUS LoRA 독립 리스크(R14) 반영, 재현성 체크리스트에 eval contamination check 훅과 DVC remote 정책 추가, §6.6 체크포인트 tie-break 규칙(D11), §6.9 Prompt format 표준(D9) 신설.

## 6.1 하드웨어

| 구분 | 사양 |
|---|---|
| GPU (보유) | RTX A6000 48GB × 2 (Ampere CC 8.6, bf16 native) |
| 주 학습 경로 | **단일 A6000로 Solar-10.7B bf16 LoRA** |
| 확장 경로 | DDP 2장 (throughput 2배) |
| Fallback | A100 40GB ×8 클러스터 임차 (SFT 가속 필요 시) |
| 공유 주의 | gene-synthesis-project가 동일 GPU 사용 중 → 스케줄 조율 (§08 R10) |

## 6.2 메모리 예산 재확인

§04.3 표와 동일 가정. **M3 pilot에서 실측 후 이 문서 업데이트**.

| 항목 | 예상 | 실측 (M3 이후) |
|---|---|---|
| base bf16 | 21.4 GB | TBD |
| LoRA adapter + optimizer state | ~1.2 GB (**R14 시 ~2.4 GB**) | TBD |
| activations — grad ckpt **on** | 5~10 GB | TBD |
| activations — grad ckpt **off** | 10~14 GB | TBD |
| 총 peak (grad ckpt on, R14 고려) | ~30~35 GB | TBD |

**주석 (R14)**: Solar-10.7B는 **DUS(Depth Up-Scaled)** 구조로 일부 layer가 복제 형태로 삽입되어 있다. PEFT의 기본 동작은 복제 layer에 대해 **독립 LoRA adapter**를 할당하므로 adapter 수와 메모리가 약 2배가 될 수 있다. M3 pilot에서 실측하고 필요 시 **복제 layer 간 LoRA 공유 정책** ablation 진행 (§08 R14).

측정 명령: `nvidia-smi --query-gpu=memory.used --format=csv -l 5` 로 5초 간격 수집, `outputs/{run_id}/mem_profile.csv` 저장.

## 6.3 프레임워크

### 1차: **Llama-Factory**
- LoRA CPT / SFT / DPO 원스톱
- bf16, DeepSpeed ZeRO, FSDP, accelerate 내장
- YAML config 기반 → 재현성 좋음
- Solar 지원 확인됨 (2025 기준)
- **ChatML template 설정**: YAML에서 `template: chatml` 1줄 (D9, §6.9)

### 대안: **torchtune**
- 공식 PyTorch, 최신 모델 대응 빠름
- config는 dataclass 기반, LoRA rank 세밀 조정 용이
- Solar 공식 지원 여부는 도입 전 검증

**선택 기준**: M0에서 두 프레임워크로 각각 1 epoch dummy run → 속도·안정성 높은 쪽 채택. 두 프레임워크 모두 **ChatML template 지원 여부** 함께 확인.

## 6.4 실험 추적

- **wandb** (rank 0 only)
  - project: `HanMed-LLM`
  - group: `cpt` / `sft` / `dpo` / `eval`
  - tags: `solar-10.7b`, `bllossom-8b`, `lora-r32`, `replay-30`, `dus-shared` 등
  - `WANDB_MODE=offline` 옵션 (폐쇄망 대비)
- 로그 항목: train_loss, val_loss, lr, throughput (tok/s), grad_norm, GPU mem, eval 지표(T1 chrF, T5 Δacc 포함)
- 체크포인트와 wandb run id 상호 링크

## 6.5 재현성 체크리스트

| 항목 | 방법 |
|---|---|
| 데이터 버전 | **DVC** — `data/parsed/`, `data/stats/` 스냅샷, git과 함께 commit |
| **DVC remote** | **로컬 NFS 또는 기관 내부 스토리지만 사용. 외부 S3/GCS 금지** — KIOM 비상업 재배포 해석 리스크 (§07.1) (D12) |
| 코드 버전 | git SHA, `requirements.lock` (uv lock) |
| 난수 seed | torch/numpy/random 고정, `PYTHONHASHSEED=0` |
| 결정성 | `torch.use_deterministic_algorithms(True)` 시도 (실패 시 경고 기록) |
| Config snapshot | YAML 전체를 `outputs/{run_id}/config.yaml` 에 복사 |
| 체크포인트 | adapter + optimizer state + trainer state 같이 저장 |
| **Eval contamination check** | **데이터 prep 시 `eval/hashes/heldout_{T1,T2,T5}.txt` 와 학습 셋 hash 교집합 검사, 발견 시 빌드 fail** (§05.7, D8) |
| **Prompt template snapshot** | **`configs/prompt_template.py`를 run_id 폴더에 복사** (§6.9) |
| 환경 | `nvidia-smi`, `nvcc --version`, `python -V` 시작 시 stdout 기록 |

## 6.6 체크포인트 정책

| 설정 | 값 |
|---|---|
| save interval | every 500 steps |
| keep | last 3 + stage별 best (아래) |
| 경로 | `outputs/{run_id}/adapters/step-{N}/` |
| best 승격 | eval 결과가 기존 best를 넘을 때 `best/` symlink 갱신 |

**Stage별 best 승격 규칙 (D11, ver2.1 — Stage 2 primary 역전)**:

| Stage | Primary | Tie-break |
|---|---|---|
| Stage 1 CPT | val_loss 최소 | (없음, tie 시 더 늦은 step) |
| Stage 2 SFT | **전문가 선호 승률 (lag 수용)** | T1 chrF (tentative best 경로) |
| Stage 3 DPO | 전문가 선호 승률 최고 | T1 chrF |

- Stage 1은 전문가 평가 없이 진행되므로 val_loss 단독 기준.
- **Stage 2 승격 2단계 프로토콜** (ver2.1):
  1. 체크포인트 저장 시점에는 **T1 chrF 최고 step을 `tentative_best/` symlink**로 기록 (전문가 평가 도착 전 임시 레퍼런스).
  2. 전문가 평가 결과 도착 후 **전문가 선호 승률 최고 step** 으로 `best/` symlink를 재설정 → 이것이 논문 main table과 model card에 게재될 최종 best.
  - 평가 lag을 수용하면서도 초기 모니터링용 reference를 유지하기 위한 구조.
  - ver2 원본은 primary=chrF / tie-break=전문가 선호였으나, §05.3.1이 chrF를 monitoring-only로 강등한 것과 모순되었다 (ver2.1 patch).
- Stage 3 DPO는 전문가 선호 승률 단독 기준, chrF는 tie-break로만.
- Model card에는 stage별 **final best** adapter의 step·지표·run_id 만 기록 (tentative best는 outputs 내부에만 유지).

## 6.7 데이터·모델 버전 관계

- `data/parsed/` 스키마 변경 시 `schema_version` 증가
- `schema_version` 또는 `corpus_stats.json` hash 변경 시 **기존 학습 결과는 invalid** 플래그 → 재학습 유도
- 학습 config에 `data_hash` 를 포함, 실행 시 불일치 시 abort

## 6.8 저장소 구조 (예상)

```
korean-medicine-llm/
├── configs/
│   ├── stage1_cpt.yaml
│   ├── stage2_sft.yaml
│   └── prompt_template.py     # ChatML 표준 (§6.9)
├── data/
├── docs/
├── eval/
│   ├── hanmed_eval_v0/{T1..T5}.jsonl
│   ├── hashes/heldout_*.txt   # contamination check (§5.7)
│   ├── prompts/
│   └── rubric/
├── outputs/{run_id}/
│   ├── adapters/
│   ├── config.yaml
│   ├── prompt_template.py     # snapshot
│   ├── mem_profile.csv
│   ├── train.log
│   └── wandb_run_id.txt
├── scripts/
├── src/
└── tests/
```

## 6.9 Prompt Format 표준 (신규, D9)

- **학습 (CPT·SFT·DPO), 평가, 추론 전 단계에서 동일 ChatML 사용** (Solar default).
- 구현: `configs/prompt_template.py`에 단일 함수 `to_chatml(system, user, assistant=None)` 정의. Llama-Factory `template: chatml`, eval 스크립트, inference wrapper 모두 이 함수를 import.
- CPT 단계의 HanMed raw 블록(§04.5 D2: `<ZH>…</ZH>\n<KO>…</KO>`)은 ChatML 래퍼 없이 **raw**로 학습되지만, 동일 special token 집합(`<ZH>`, `<KO>`, ChatML 토큰)을 토크나이저에 등록해 SFT 전환 시 재정의 없이 이어감.
- 비교군(Bllossom, Qwen, GPT-4o, Claude) 호출도 동일 ChatML 래퍼로 통일 → 포맷 편향 제거.
- `scripts/eval.sh` 시작부에서 wrapper signature 검사, 불일치 시 run 무효.

## 6.10 네트워킹·폐쇄망

- wandb offline 가능
- HuggingFace Hub: 사전 모델 다운로드는 온라인 필요 → M0에서 전부 캐시
- 평가 시 GPT-4o / Claude API는 키 분리, **평가 데이터가 API 제공자 로그에 남을 수 있음** 주의 → 안전성 태스크(T4)는 API 호출 로그 contamination 감수하고 publish 시 별도 안내

## 6.11 GPU 공유 조율

- A6000 × 2는 gene-synthesis-project와 공유
- 본 프로젝트는 **주중 야간 / 주말** 우선
- 긴 CPT run 은 사전 캘린더 blocking
- `nvidia-smi` 점유 확인 후 시작
