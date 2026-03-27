# Phase 0.5: 실험 운영 및 재현성 규칙

- **입력**: `configs/default.yaml`, `split_manifest.json`, run 디렉토리
- **출력**: 재현 가능한 실험 실행 규칙
- **의존 문서**: `01_overview`, `02_preprocessing`, `07_project_structure`

---

## 1. 목적

이 문서는 모델 설계가 아니라 **실험 운영 계약**을 정의한다.
같은 모델이라도 split, seed, naming, checkpoint 보존 규칙이 흔들리면 비교가 무의미해진다.

---

## 2. Seed 규칙

| 단계 | 기본 seed |
|------|-----------|
| preprocessing | `20260327` |
| training baseline | `20260327` |
| training repeat 2 | `20260328` |
| training repeat 3 | `20260329` |
| inference generation | 학습 seed와 동일 또는 명시적 override |
| evaluation bootstrap | `20260401` |

규칙:

- 전처리 split seed는 고정하고 바꾸지 않는다.
- 최종 성능 비교는 최소 3 seeds 평균으로 수행한다.
- ablation과 baseline은 동일 seed set을 사용한다.

---

## 3. Run Naming 규칙

권장 포맷:

```text
YYYYMMDD_<phase>_<tag>_seed<seed>
```

예시:

```text
20260327_train_baseline_seed20260327
20260327_train_ablation-no-film_seed20260328
20260328_eval_baseline_seed20260327
```

최소 포함 정보:

- 날짜
- 단계(train / infer / eval / sweep)
- 실험 tag
- seed

---

## 4. Split 재사용 규칙

`data/processed/split_manifest.json`은 모든 baseline, ablation, proposal이 공유하는 기준 파일이다.

금지:

- 모델마다 임의로 train/test 재분할
- baseline과 proposal에 서로 다른 split 사용
- 보고 단계에서 가장 잘 나온 split만 선택

허용:

- 완전히 새로운 데이터 표현을 도입해 전처리를 다시 설계하는 경우

단, 이 경우에도 새 split manifest를 별도 이름으로 저장하고 이전 결과와 직접 비교하지 않는다.

---

## 5. Checkpoint 보존 및 Resume

보존 규칙:

- 항상 `best_model.pth` 유지
- 주기 checkpoint는 최근 3개만 유지
- 중단 복구는 가장 최근 `checkpoint_epoch*.pth`에서 재개

권장 저장 항목:

- `model_state_dict`
- `ema_state_dict`
- `optimizer_state_dict`
- `scheduler_state_dict`
- `epoch`
- `best_val_loss`
- `config`

resume 규칙:

- resume 시 seed, config, split manifest 경로를 로그에 다시 기록
- config가 다르면 경고를 띄우고 새 run으로 분기한다

---

## 6. 결과 보고 규칙

최종 보고에는 아래 항목이 포함되어야 한다.

- run name
- git 상태 또는 문서 버전 날짜
- config hash 또는 config 파일 경로
- split manifest 경로
- seed
- 평가 샘플 수
- 핵심 지표와 95% CI 또는 seed 반복 표준편차

권장 산출물:

- `summary_metrics.json`
- `per_population_metrics.csv`
- `bootstrap_intervals.json`
- figure PNG/PDF

---

## 7. 외부 공유 및 배포 정책

기본 정책은 `internal_only`다.

외부 공유 가능:

- aggregate metric table
- figure
- 실험 설정 요약
- 비식별 run metadata

외부 공유 금지:

- raw genotype
- 개별 synthetic sample tensor
- sample-level nearest-neighbor 결과
- membership inference 개별 판정 결과

개별 synthetic sample export가 필요하면 아래 조건을 모두 만족해야 한다.

1. privacy 평가(NNAA, DUPI, membership AUC) 완료
2. 내부 검토 완료
3. export 목적과 범위 문서화

---

## 8. 실패 시 처리 원칙

- OOM: batch size 또는 gradient accumulation 조정 후 동일 run tag로 재시도하지 말고 suffix를 붙인다
- 데이터 정렬 오류: 즉시 중단, 산출물 삭제 후 전처리 재실행
- NaN loss: checkpoint resume보다 설정 조정 우선
- 평가 실패: partial metric만 보고하지 말고 실패 원인과 누락 지표를 함께 기록
