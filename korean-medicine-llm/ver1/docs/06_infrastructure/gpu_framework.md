# 06. Infrastructure — GPU·프레임워크·재현성

## 6.1 하드웨어

| 구분 | 사양 |
|---|---|
| GPU (보유) | RTX A6000 48GB × 2 (Ampere CC 8.6, bf16 native) |
| 주 학습 경로 | **단일 A6000로 Solar-10.7B bf16 LoRA** |
| 확장 경로 | DDP 2장 (throughput 2배) |
| Fallback | A100 40GB ×8 클러스터 임차 (SFT 가속 필요 시) |
| 공유 주의 | gene-synthesis-project가 동일 GPU를 사용 중 → 스케줄 조율 (§08 R10) |

## 6.2 메모리 예산 재확인

§04.3 표와 동일. **Stage 1 pilot에서 실측 후 이 문서 업데이트**.

| 항목 | 예상 | 실측 (M3 이후) |
|---|---|---|
| base bf16 | 21.4 GB | TBD |
| LoRA + opt state | ~1.2 GB | TBD |
| activations (bs 2, seq 2048, grad ckpt) | ~10~14 GB | TBD |
| 총 peak | ~35 GB | TBD |

`nvidia-smi --query-gpu=memory.used --format=csv -l 5` 로 5초 간격 수집.

## 6.3 프레임워크

### 1차: **Llama-Factory**
- LoRA CPT / SFT / DPO 원스톱
- bf16, DeepSpeed ZeRO, FSDP, accelerate 내장
- YAML config 기반 → 재현성 좋음
- 커뮤니티 크고 이슈 트래킹 활발
- Solar 지원 확인됨 (2025 기준)

### 대안: **torchtune**
- 공식 PyTorch, 최신 모델 대응 빠름
- config는 dataclass 기반, LoRA rank 등 세밀 조정 용이
- Solar 공식 지원 여부는 도입 전 검증

**선택 기준**: M0에서 두 프레임워크로 각각 1 epoch dummy run → 속도·안정성 높은 쪽 채택.

## 6.4 실험 추적

- **wandb** (rank 0 only)
  - project: `HanMed-LLM`
  - group: `cpt` / `sft` / `dpo` / `eval`
  - tags: `solar-10.7b`, `bllossom-8b`, `lora-r32` 등
  - `WANDB_MODE=offline` 옵션 (폐쇄망 대비)
- 로그 항목: train_loss, val_loss, lr, throughput (tok/s), grad_norm, GPU mem, eval 지표
- 체크포인트와 wandb run id 상호 링크

## 6.5 재현성 체크리스트

| 항목 | 방법 |
|---|---|
| 데이터 버전 | **DVC** — `data/parsed/`, `data/stats/` 스냅샷, git과 함께 commit |
| 코드 버전 | git SHA, `requirements.lock` (uv lock) |
| 난수 seed | torch/numpy/random 고정, `PYTHONHASHSEED=0` |
| 결정성 | `torch.use_deterministic_algorithms(True)` 시도 (실패 시 경고 기록) |
| Config snapshot | YAML 전체를 `outputs/{run_id}/config.yaml` 에 복사 |
| 체크포인트 | adapter + optimizer state + trainer state 같이 저장 |
| 환경 | `nvidia-smi`, `nvcc --version`, `python -V` 시작 시 stdout 기록 |

## 6.6 체크포인트 정책

| 설정 | 값 |
|---|---|
| save interval | every 500 steps |
| keep | last 3 + best by val_loss + best by T1 chrF |
| 경로 | `outputs/{run_id}/adapters/step-{N}/` |
| best 승격 | eval 결과가 기존 best를 넘을 때 `best/` symlink 갱신 |
| model card 포함 | `best/` 만 포함 |

## 6.7 데이터·모델 버전 관계

- `data/parsed/` 스키마 변경 시 `schema_version` 증가
- `schema_version` 또는 `corpus_stats.json` hash 변경 시 **기존 학습 결과는 invalid** 플래그 → 재학습 유도
- 학습 config에 `data_hash` 를 포함, 실행 시 불일치 시 abort

## 6.8 저장소 구조 (예상)

```
korean-medicine-llm/
├── configs/           # YAML configs (stage1_cpt.yaml, stage2_sft.yaml, ...)
├── data/              # §02, §03 참고
├── docs/              # 본 기획서
├── eval/              # §05 태스크 파일 + rubric + 프롬프트
├── outputs/
│   └── {run_id}/
│       ├── adapters/
│       ├── config.yaml
│       ├── train.log
│       └── wandb_run_id.txt
├── scripts/
│   ├── download_mediclassics.py
│   ├── parse_markup.py
│   ├── train.sh
│   └── eval.sh
├── src/
│   ├── data/          # 파서, 정제, 믹스
│   ├── training/      # trainer wrapper
│   └── evaluation/    # 지표 계산
└── tests/
```

## 6.9 네트워킹·폐쇄망

- wandb는 offline 가능
- HuggingFace Hub: 사전 모델 다운로드는 온라인 필요 → M0에서 전부 캐시
- 평가 시 GPT-4o / Claude API는 키 분리, **평가 데이터가 API 제공자 로그에 남을 수 있음** 감안

## 6.10 GPU 공유 조율

현재 A6000 × 2는 gene-synthesis-project (diffusion 학습) 에서도 사용. 충돌 방지 정책:
- 본 프로젝트는 **주중 야간 / 주말** 우선
- 긴 CPT run은 사전 캘린더 blocking
- `nvidia-smi`로 점유 확인 후 시작
