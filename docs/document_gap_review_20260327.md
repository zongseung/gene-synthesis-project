# 숫자별 기획서 보강 리뷰

작성일: 2026-03-27
대상: `docs/01_overview` ~ `docs/09_performance_optimization`
**보강 완료일: 2026-03-27**

## 요약

현재 숫자별 문서는 개별 설명은 매우 상세하지만, 서로 연결되는 계약이 아직 고정되지 않았다.
특히 `데이터 표현`, `설정 파일 스키마`, `산출물 형식`, `평가 가능 지표`가 문서마다 다르게 전제된다.

가장 먼저 보강해야 할 것은 "무엇을 실제로 학습/생성/평가하는가"에 대한 단일 기준 문서다.
이 기준이 고정되지 않으면 이후 문서의 코드 예시도 계속 어긋난다.

## 우선순위 P0

### 1. 데이터 표현을 하나로 고정

현재 두 개의 전제가 섞여 있다.

- Gene PCA 텐서 `(B, K, gene_size)` 기반 파이프라인
- 원시 SNP/하플로타입 기반 아키텍처 및 분포 손실

보강 필요:

- `01_overview` 맨 앞에 "이번 구현 범위"를 명시
- `03_model`에 입력 표현을 한 줄로 고정
- `06_evaluation`에서 해당 표현으로 계산 가능한 지표와 불가능한 지표를 분리
- `09_performance_optimization`에서 원시 SNP 전제가 필요한 전략은 "후속 단계"로 분리

권장 방향:

- 현재 레포와 `architecture_review.md` 기준으로는 1차 구현 범위를 `Gene PCA 기반`으로 고정하는 편이 현실적이다.
- 원시 SNP 기반 손실은 별도 확장안으로 분리하는 것이 맞다.

### 2. 설정 파일 스키마를 단일화

문서에는 nested YAML 구조가 나오지만, 학습/추론 코드 예시는 flat key 접근을 사용한다.
이 상태로는 그대로 구현해도 문서 간 인터페이스가 깨진다.

보강 필요:

- canonical config schema 1개 선정
- 예시 코드 전체를 그 스키마에 맞게 통일
- `config loader`에서 nested → flat 변환을 할지 여부 명시

최소 합의 항목:

- `gene_size`
- `num_channels`
- `zero_mask_path`
- `label_hierarchy_path`
- `max_timesteps`
- `batch_size`
- `save_dir`

### 3. 전처리 산출물 계약을 명확히 정의

현재 `02_preprocessing`에서 가장 중요한 계약이 모호하다.

- PCA 최적 K 선택 기준이 `90% threshold`와 `elbow/marginal gain` 사이에서 흔들림
- `label_hierarchy.pkl`에 무엇이 들어가는지 문서마다 다름
- `zero_mask` shape가 K와 함께 바뀌는데 일부 예시는 `8`로 고정

보강 필요:

- K 선택 규칙을 하나로 확정
- `label_hierarchy.pkl` 필드 정의 추가
- `normalization_stats.pkl`, `train_data.pkl`, `test_data.pkl`의 내부 스키마 명시
- `gene_size` 계산식과 padding 규칙을 overview / model / training에 전파

추천 필드:

- `label_hierarchy.pkl`
  - `pop_to_idx`
  - `idx_to_pop`
  - `pop_to_superpop`
  - `superpop_to_idx`
  - `pop_sizes`

## 우선순위 P1

### 4. 추론 문서의 I/O 계약 보강

`05_inference`는 실행 예시와 코드 예시가 완전히 일치하지 않는다.

보강 필요:

- `best_model.pth` 내부 필드 고정
- EMA 사용 시 어떤 state를 추론에 적용하는지 명시
- `generation_meta.json` 생성 로직 추가 또는 산출물 목록에서 제거
- `oversample_minority` CLI 예시를 실제 함수 시그니처와 맞춤
- 역정규화 시 텐서 shape broadcasting 규칙 명시

### 5. 평가 지표를 "계산 가능" 기준으로 재정리

현재 평가는 논문형 지표는 풍부하지만, PCA 기반 생성물에서 바로 계산 가능한지 설명이 부족하다.

보강 필요:

- 직접 계산 가능:
  - PCA 공간 분포 거리
  - 인구군 분리도/겹침도
  - recovery rate
  - privacy / nearest-neighbor 계열
- 추가 역변환 또는 별도 데이터가 필요:
  - SNP 단위 AF
  - MAF 저빈도 영역 평가
  - LD 감쇠
  - haplotype diversity

문서에는 "현재 평가 세트"와 "후속 평가 세트"를 분리해 두는 편이 낫다.

### 6. 계획 구조와 현재 레포 상태를 분리

`07_project_structure`는 목표 구조를 잘 정리했지만, 현재 레포에는 아직 대부분 존재하지 않는다.
그래서 독자가 현재 구현 상태와 목표 구조를 혼동할 수 있다.

보강 필요:

- 섹션 제목을 `목표 디렉토리 구조`로 변경
- 별도 표에 `현재 존재`, `예정`, `후속` 상태 표시
- 실행 명령어에 "아직 미구현" 또는 "작성 예정" 표시

### 7. 의존성 문서와 실제 `pyproject.toml` 정렬

문서에는 `wandb`, `pyyaml`, `joblib`, `cyvcf2` 등이 필수처럼 등장하지만 실제 `pyproject.toml`에는 빠져 있다.
반대로 문서 기준 버전과 실제 버전 범위도 다르다.

보강 필요:

- `07_project_structure`의 의존성 표를 현재 파일 기준으로 갱신
- 문서에서 요구하는 패키지를 `pyproject.toml`에 반영할지 결정
- CPU/GPU 환경별 추가 설치가 필요한 패키지는 별도 섹션으로 분리

## 우선순위 P2

### 8. Phase별 완료 조건 추가

각 Phase에 "문서 설명"은 충분하지만, 완료 판정 기준이 약하다.

추가 권장:

- Phase 1 완료 조건
  - 전처리 산출물 생성 완료
  - shape / label integrity 검사 통과
- Phase 2 완료 조건
  - shape test 통과
  - 파라미터 수 / VRAM 추정 기록
- Phase 3 완료 조건
  - 1 epoch smoke test
  - DDP 2GPU 정상 동작
  - checkpoint 저장/로드 검증
- Phase 4 완료 조건
  - 소량 샘플 생성
  - zero_mask 적용 검증
- Phase 5 완료 조건
  - 핵심 지표 json 저장
  - 인구군별 결과 표 생성

### 9. 문서 간 의존성 링크 강화

현재는 서로 참조는 하지만, 어느 문서가 source of truth인지 약하다.

추가 권장:

- `01_overview`에 각 문서의 source of truth 범위 표 추가
- 각 README 상단에 입력/출력/의존 문서 3줄 요약 추가

## 바로 수정하면 좋은 순서

1. ✅ `01_overview`에 구현 범위와 canonical config schema 추가
2. ✅ `02_preprocessing`에서 K 선택 규칙과 산출물 스키마 확정
3. ✅ `03_model`에 입력 표현 한 줄 고정
4. ✅ `05_inference`에서 checkpoint / metadata / CLI 정합성 수정
5. ✅ `06_evaluation`에서 "현재 가능 지표 vs 후속 지표" 분리
6. ✅ `07_project_structure`를 "현재 상태"와 "목표 구조"로 분리
7. ✅ `09_performance_optimization`에서 원시 SNP 전제 전략을 "후속 확장"으로 분리
