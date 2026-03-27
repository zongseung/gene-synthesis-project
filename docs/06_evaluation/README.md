# Phase 5: 평가 파이프라인

- **입력 표현**: Gene PCA 텐서 `(B, K, gene_size)` — `01_overview` "1차 구현 범위" 참조
- **의존 문서**: `02_preprocessing` (산출물 스키마), `05_inference` (생성 샘플 형식)

---

## 0. 지표 분류: 현재 평가 세트 vs 후속 평가 세트

> PCA 기반 생성물에서 **직접 계산 가능한 지표**와 **SNP 역변환이 필요한 지표**를 구분한다.

### 현재 평가 세트 (1차 구현, PCA 공간에서 직접 계산)

| 카테고리 | 지표 | 입력 | 비고 |
|----------|------|------|------|
| **Fidelity** | PCA 채널별 Wasserstein distance | PCA 텐서 | 채널별 분포 비교 |
| **Fidelity** | AF 상관 (전체) — PCA 공간 유전자 평균 | PCA 텐서 | `real_mean` vs `syn_mean` Pearson r |
| **Structure** | PCA 겹침도 (Silhouette score) | PCA 텐서 → 2D PCA | real/syn 구분 불가 = 이상적 |
| **Structure** | Sliced Wasserstein Distance | PCA 텐서 → 2D PCA | test set 대비 ≤2x |
| **Utility** | Recovery Rate | PCA 텐서 + 레이블 | 분류기 학습→테스트 정확도 비율 |
| **Utility** | 증강 효과 (5%, 50%) | PCA 텐서 + 레이블 | real+syn 혼합 학습 |
| **Privacy** | NNAA (train/test) | PCA 텐서 (flatten) | ≈ 0.5 이상적 |
| **Privacy** | DUPI (Jeong et al. 2023) | PCA 텐서 (flatten) | 이론 벤치마크 기반 |
| **Privacy** | 멤버십 추론 AUC | PCA 텐서 (flatten) | ≈ 0.5 이상적 |
| **Robustness** | 인구군 크기 vs 품질 상관 (|r|) | PCA 텐서 + 레이블 | |r| → 0 이상적 |
| **Robustness** | 인구군별 DUPI gap | PCA 텐서 + 레이블 | FiLM 강건성 |

### 후속 평가 세트 (SNP 역변환 또는 별도 데이터 필요)

| 카테고리 | 지표 | 필요 조건 | 비고 |
|----------|------|----------|------|
| Fidelity | SNP 단위 AF 상관 | PCA → SNP 역변환 | 개별 변이 수준 비교 |
| Fidelity | MAF 저빈도 영역 상관 (MAF≤0.05) | PCA → SNP 역변환 | 저빈도 변이 복원 |
| Fidelity | LD 감쇠 상관 | PCA → SNP 역변환 + 물리적 위치 | 변이 간 거리별 상관 |
| Diversity | Haplotype diversity | PCA → SNP 역변환 | 하플로타입 수준 |
| Diversity | k-mer 엔트로피 | PCA → SNP 역변환 | 서열 패턴 |

> **후속 평가는 PCA 역변환 모듈 구현 후 추가한다.** 1차 논문에서는 "현재 평가 세트"로 충분하며, 후속 평가는 확장 실험으로 기술한다.

---

## 0.5 평가 프로토콜 (확정)

평가 지표의 종류뿐 아니라 **계산 절차**도 고정한다.

| 항목 | 규칙 |
|------|------|
| split 사용 | `02_preprocessing`의 `split_manifest.json` 재사용 |
| 평가 대상 real | test split만 사용 |
| 평가 대상 syn | 기본은 test split과 동일 총 샘플 수 |
| per-pop 평가 | 가능한 한 real test의 population 비율과 동일하게 맞춤 |
| bootstrap CI | 1,000회 bootstrap, 95% CI |
| seed 반복 | 최종 표/그림은 3 seeds 평균 + 표준편차 권장 |
| baseline 비교 | 동일 split, 동일 샘플 수, 동일 classifier/eval 설정 사용 |

```python
EVAL_BOOTSTRAP_ROUNDS = 1000
EVAL_SEEDS = [20260327, 20260328, 20260329]
```

평가 보고 규칙:

- 단일 스칼라만 보고하지 않고 `mean ± std` 또는 `95% CI`를 함께 기록
- population-wise 결과는 전체 평균과 함께 별도 표로 저장
- oversampling 실험은 기본 생성 실험과 분리하여 비교한다

---

## 1. 평가 지표 (병렬 실행)

모든 지표 계산은 병렬로 실행한다:

```python
from concurrent.futures import ProcessPoolExecutor

def evaluate_all(real_data, syn_data, real_labels, syn_labels, config):
    """전체 평가 파이프라인 (병렬)"""
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = {
            'fidelity': executor.submit(evaluate_fidelity, real_data, syn_data),
            'structure': executor.submit(evaluate_structure, real_data, syn_data, real_labels, syn_labels),
            'utility': executor.submit(evaluate_utility, real_data, syn_data, real_labels, syn_labels, config),
            'privacy': executor.submit(evaluate_privacy, real_data, syn_data),
            'diversity': executor.submit(evaluate_diversity, real_data, syn_data),
            'robustness': executor.submit(evaluate_robustness, real_data, syn_data, real_labels, syn_labels),
        }
        results = {name: f.result() for name, f in futures.items()}
    return results
```

---

## 2. 지표 상세

### 2.1 충실도 (Fidelity)

| 지표 | 측정 | 목표 |
|------|------|------|
| AF 상관 (전체) | Pearson r (유전자별 평균값 real vs syn) | r ≥ 0.95 |
| AF 상관 (저빈도) | MAF ≤ 0.05 영역만 | r ≥ 0.85 | ⚠️ 후속 (SNP 역변환 필요) |
| LD 감쇠 상관 | 유전자 간 거리별 상관 비교 | r ≥ 0.90 | ⚠️ 후속 (SNP 역변환 필요) |
| 채널별 분포 | PCA 성분별 Wasserstein distance | 작을수록 좋음 |

### 2.2 구조 (Structure)

| 지표 | 측정 | 목표 |
|------|------|------|
| PCA 겹침도 | 2D PCA 투영 후 Silhouette score | real과 유사 |
| Sliced Wasserstein | 2D PCA 공간에서 분포 거리 | test set 대비 ≤2x |

### 2.3 유용성 (Utility)

| 지표 | 측정 | 목표 |
|------|------|------|
| Recovery Rate | 합성 학습→실제 테스트 정확도 / 실제 학습 정확도 | ≥ 0.93 |
| 증강 효과 (5%) | 5% real + 95% syn 학습 시 정확도 향상 | GeneDiffusion 대비 개선 |
| 증강 효과 (50%) | 50% real + 50% syn 학습 시 | 상동 |

### 2.4 프라이버시 (Privacy)

| 지표 | 측정 | 목표 |
|------|------|------|
| NNAA (train) | Nearest Neighbor Adversarial Accuracy | ≈ 0.5 |
| NNAA (test) | 상동 | ≈ 0.5 |
| 멤버십 추론 AUC | 공격 모델 AUC | ≈ 0.5 |
| **DUPI<1>** | Jeong et al. (2023) utility-privacy 통합 지표 | ≈ benchmark |

#### 2.4.1 DUPI: 수학적 정의

**출처**: Jeong, D., Kim, J. H. T., & Im, J. (2023). "Synthetic Data — What, Why and How?" *IEEE Transactions on Information Forensics and Security*.
- DOI: https://doi.org/10.1109/TIFS.2022.3228753
- Preprint: https://d197for5662m48.cloudfront.net/documents/publicationstatus/165729/preprint_pdf/494aa2f52af2f4de735b1849f8707b8d.pdf

**표기법**:
- `X_n = {x_1, ..., x_n}`: 원본(real) 데이터 (n개)
- `Y_m = {y_1, ..., y_m}`: 합성(synthetic) 데이터 (m개)
- `d(a, b)`: 거리 함수 (L2 Euclidean 사용)
- `d^{<k>}_S(c)`: 집합 S에서 점 c까지의 k번째 최근접 이웃 거리

**정의 (DUPI^{<k>})**:

```
DUPI^{<k>} = (1/n) × Σ_{i=1}^{n} I( d^{<k>}_{Y_m}(x_i)  ≤  d^{<k>}_{X_n\i}(x_i) )

각 원본 점 x_i에 대해:
  - d^{<k>}_{Y_m}(x_i):    합성 데이터에서 x_i까지의 k번째 최근접 거리
  - d^{<k>}_{X_n\i}(x_i):  x_i를 제외한 원본 데이터에서의 k번째 최근접 거리
  - I(.):                  지시 함수 (조건 만족 시 1)

→ "합성 데이터가 원본보다 더 가까운 원본 점의 비율"
```

**이론 벤치마크 (Theorem 4)**:

X_n과 Y_m이 같은 분포에서 독립 추출된 경우:

```
k=1:    DUPI_0 = m / (n + m - 1)
n=m:    DUPI_0 = n / (2n - 1) ≈ 0.5

일반 k:  DUPI_0 = Σ_{s=k}^{2k-1} C(s-1,k-1)·C(n-1+m-s,m-k) / C(n-1+m,m)
```

**해석**:

| DUPI 값 | 의미 | 진단 |
|---------|------|------|
| ≈ 1 | 합성이 원본에 지나치게 가까움 | utility↑ privacy↓ (data leakage 위험) |
| **≈ DUPI_0 (≈0.5)** | **최적 균형** | **같은 분포의 독립 샘플처럼 동작** |
| ≈ 0 | 합성이 원본에서 너무 멀어짐 | utility↓ privacy↑ (정보 손실) |

#### 2.4.2 UI / PI 분해 (시각화용)

DUPI를 Utility Index와 Privacy Index로 분해하여 2D 플롯으로 비교:

```
리스케일링 함수 g:
  DUPI ≤ DUPI_0:  g = DUPI / (2 · DUPI_0)
  DUPI > DUPI_0:  g = (DUPI - DUPI_0) / (2·(1 - DUPI_0)) + 0.5

UI = arctan(τ · g) / arctan(τ)          τ = 5 (기본값)
PI = arctan(τ - τ · g) / arctan(τ)

최적 조건 (Theorem 5):
  UI · PI ≤ [arctan(τ/2) / arctan(τ)]²
  등호 ⟺ DUPI = DUPI_0  (최적 균형점)

τ=5, DUPI_0≈0.5 일 때 최적 (UI_0, PI_0) ≈ (0.867, 0.867)
```

#### 2.4.3 DUPI와 NNAA의 관계

```
NNAA = 0.5 × [ (real→real 더 가까운 비율) + (syn→syn 더 가까운 비율) ]

DUPI^{<1>} ≈ 1 - NNAA의 첫 번째 항

차이점:
- NNAA: 양방향 대칭 (real→syn + syn→real)
- DUPI: 단방향 (real→syn만), 대신 이론 벤치마크가 존재

→ 둘 다 ≈ 0.5가 이상적이나, DUPI는 정확한 통계적 기준(DUPI_0)을 제공
→ NNAA는 기존 유전형 생성 논문과의 비교용, DUPI는 정량적 판정용으로 병행
```

#### 2.4.4 구현 코드

**파일 구조**:
```
src/evaluation/
├── privacy.py            # evaluate_privacy() — NNAA + DUPI + 멤버십 추론
├── privacy_dupi.py       # DUPI 전용 구현 (아래 코드)
└── privacy_nnaa.py       # NNAA 전용 구현
```

**`src/evaluation/privacy_dupi.py`**:

```python
"""
DUPI (Data Utility and Privacy Index) 구현

Jeong, D., Kim, J. H. T., & Im, J. (2023).
"Synthetic Data — What, Why and How?"
IEEE Transactions on Information Forensics and Security.
https://doi.org/10.1109/TIFS.2022.3228753

사용법:
    results = evaluate_dupi(real_data, syn_data, k=1, metric='euclidean')
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import comb
import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────
# 1. 핵심: DUPI 계산
# ──────────────────────────────────────────────────────

def compute_dupi(X_real: np.ndarray, Y_syn: np.ndarray,
                 k: int = 1, metric: str = 'euclidean') -> dict:
    """
    DUPI^{<k>} 계산

    Args:
        X_real: (n, d) 원본 데이터
        Y_syn:  (m, d) 합성 데이터
        k: 최근접 이웃 차수 (기본 1, 논문 권고)
        metric: scipy cdist 거리 함수 (L2 기본)

    Returns:
        dict with dupi, benchmark, gap, per_sample_indicators
    """
    n, m = X_real.shape[0], Y_syn.shape[0]

    if X_real.ndim != 2 or Y_syn.ndim != 2:
        raise ValueError(
            f"Expected 2D arrays, got X_real={X_real.ndim}D, Y_syn={Y_syn.ndim}D. "
            f"Flatten PCA features first: data.reshape(n, -1)"
        )
    if X_real.shape[1] != Y_syn.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: X_real={X_real.shape[1]}, Y_syn={Y_syn.shape[1]}"
        )

    logger.info(f"DUPI 계산: n={n}, m={m}, k={k}, metric={metric}")

    # ── 거리 행렬 계산 ──
    # X_real → Y_syn: (n, m)
    dist_xy = cdist(X_real, Y_syn, metric=metric)

    # X_real → X_real: (n, n), 자기 자신 제외
    dist_xx = cdist(X_real, X_real, metric=metric)
    np.fill_diagonal(dist_xx, np.inf)  # x_i 자신은 제외

    # ── k번째 최근접 거리 ──
    if k == 1:
        d_y_k = np.min(dist_xy, axis=1)   # (n,)
        d_x_k = np.min(dist_xx, axis=1)   # (n,)
    else:
        # np.partition: k번째 작은 값을 O(n) 시간에 찾음
        d_y_k = np.partition(dist_xy, k - 1, axis=1)[:, k - 1]
        d_x_k = np.partition(dist_xx, k - 1, axis=1)[:, k - 1]

    # ── DUPI ──
    indicators = (d_y_k <= d_x_k).astype(np.float64)  # 샘플별 지시 함수
    dupi = float(np.mean(indicators))
    benchmark = _compute_benchmark(n, m, k)
    gap = abs(dupi - benchmark)

    logger.info(f"  DUPI^{{<{k}>}} = {dupi:.4f}  (benchmark = {benchmark:.4f}, gap = {gap:.4f})")

    return {
        'dupi': dupi,
        'benchmark': benchmark,
        'gap': gap,
        'k': k,
        'n_real': n,
        'n_syn': m,
        'per_sample_indicators': indicators,  # (n,) — 샘플별 결과, 인구군별 분석에 활용
    }


def _compute_benchmark(n: int, m: int, k: int = 1) -> float:
    """
    DUPI 이론 벤치마크 (Theorem 4)

    k=1: m / (n + m - 1)
    일반 k: 이항 계수 합
    """
    if k == 1:
        return m / (n + m - 1)

    total = 0.0
    denom = comb(n - 1 + m, m, exact=True)
    if denom == 0:
        raise ValueError(f"Benchmark computation overflow: n={n}, m={m}, k={k}")
    for s in range(k, 2 * k):
        total += comb(s - 1, k - 1, exact=True) * comb(n - 1 + m - s, m - k, exact=True)
    return total / denom


# ──────────────────────────────────────────────────────
# 2. UI / PI 분해
# ──────────────────────────────────────────────────────

def compute_ui_pi(dupi: float, benchmark: float, tau: float = 5.0) -> dict:
    """
    DUPI → Utility Index + Privacy Index 분해

    Args:
        dupi: 관측된 DUPI 값
        benchmark: 이론 벤치마크 (DUPI_0)
        tau: 형상 파라미터 (기본 5, 논문 권고)

    Returns:
        dict with ui, pi, ui_pi_product, optimal_product
    """
    # 리스케일링
    if dupi <= benchmark:
        g = dupi / (2.0 * benchmark)
    else:
        g = (dupi - benchmark) / (2.0 * (1.0 - benchmark)) + 0.5

    ui = float(np.arctan(tau * g) / np.arctan(tau))
    pi = float(np.arctan(tau - tau * g) / np.arctan(tau))

    # 최적 곱 (Theorem 5)
    optimal = float((np.arctan(tau / 2) / np.arctan(tau)) ** 2)

    return {
        'ui': ui,
        'pi': pi,
        'ui_pi_product': ui * pi,
        'optimal_product': optimal,
        'is_near_optimal': abs(ui * pi - optimal) < 0.05,
        'tau': tau,
    }


# ──────────────────────────────────────────────────────
# 3. NNAA (비교용)
# ──────────────────────────────────────────────────────

def compute_nnaa(X_real: np.ndarray, Y_syn: np.ndarray,
                 metric: str = 'euclidean') -> dict:
    """
    NNAA (Nearest Neighbor Adversarial Accuracy)

    AA = 0.5 × [ P(real→real 더 가까움) + P(syn→syn 더 가까움) ]
    이상적 값: 0.5

    DUPI와의 차이:
    - NNAA는 양방향 대칭, DUPI는 단방향
    - NNAA는 이론 벤치마크 없음, DUPI는 정확한 DUPI_0 존재
    """
    n, m = X_real.shape[0], Y_syn.shape[0]

    dist_xy = cdist(X_real, Y_syn, metric=metric)

    dist_xx = cdist(X_real, X_real, metric=metric)
    np.fill_diagonal(dist_xx, np.inf)

    dist_yy = cdist(Y_syn, Y_syn, metric=metric)
    np.fill_diagonal(dist_yy, np.inf)

    # 방향 1: 각 real 점에서 nearest real이 nearest syn보다 가까운 비율
    d_real2syn = np.min(dist_xy, axis=1)
    d_real2real = np.min(dist_xx, axis=1)
    term1 = float(np.mean(d_real2syn > d_real2real))

    # 방향 2: 각 syn 점에서 nearest syn이 nearest real보다 가까운 비율
    d_syn2real = np.min(dist_xy.T, axis=1)
    d_syn2syn = np.min(dist_yy, axis=1)
    term2 = float(np.mean(d_syn2real > d_syn2syn))

    nnaa = 0.5 * (term1 + term2)

    return {
        'nnaa': nnaa,
        'nnaa_real_term': term1,
        'nnaa_syn_term': term2,
        'ideal': 0.5,
        'gap': abs(nnaa - 0.5),
    }


# ──────────────────────────────────────────────────────
# 4. 인구군별 DUPI (강건성 분석용)
# ──────────────────────────────────────────────────────

def compute_dupi_per_population(X_real: np.ndarray, Y_syn: np.ndarray,
                                 real_labels: np.ndarray, syn_labels: np.ndarray,
                                 k: int = 1, metric: str = 'euclidean') -> dict:
    """
    인구군별 DUPI — FiLM 강건성 분석의 핵심

    소수 인구군(ASW 61명)과 대규모 인구군(YRI 108명)의
    DUPI가 모두 벤치마크 근처이면 → FiLM이 균일한 프라이버시 보호 제공

    Args:
        X_real: (n, d) 원본 flattened
        Y_syn: (m, d) 합성 flattened
        real_labels: (n,) 인구군 인덱스
        syn_labels: (m,) 인구군 인덱스
    """
    pop_results = {}

    for pop_idx in np.unique(real_labels):
        real_mask = real_labels == pop_idx
        syn_mask = syn_labels == pop_idx

        n_real = int(real_mask.sum())
        n_syn = int(syn_mask.sum())

        if n_real < 5 or n_syn < 5:
            logger.warning(f"Pop {pop_idx}: too few samples (real={n_real}, syn={n_syn}), skipping")
            continue

        result = compute_dupi(X_real[real_mask], Y_syn[syn_mask], k=k, metric=metric)
        pop_results[int(pop_idx)] = {
            'dupi': result['dupi'],
            'benchmark': result['benchmark'],
            'gap': result['gap'],
            'n_real': n_real,
            'n_syn': n_syn,
        }

    # 인구군 크기 vs DUPI gap 상관 분석
    if len(pop_results) >= 5:
        sizes = [v['n_real'] for v in pop_results.values()]
        gaps = [v['gap'] for v in pop_results.values()]
        size_gap_corr = float(np.corrcoef(sizes, gaps)[0, 1])
    else:
        size_gap_corr = float('nan')

    return {
        'per_population': pop_results,
        'size_gap_correlation': size_gap_corr,
        # |r|이 작을수록 = 인구군 크기와 무관한 프라이버시 = FiLM 강건성
    }


# ──────────────────────────────────────────────────────
# 5. 통합 프라이버시 평가 함수
# ──────────────────────────────────────────────────────

def evaluate_privacy(real_data: np.ndarray, syn_data: np.ndarray,
                     real_labels: np.ndarray = None, syn_labels: np.ndarray = None,
                     k: int = 1, metric: str = 'euclidean') -> dict:
    """
    프라이버시 평가 통합 함수

    1. DUPI (primary — 이론적 벤치마크 기반 판정)
    2. UI/PI (시각화용 분해)
    3. NNAA (기존 논문 비교용)
    4. 인구군별 DUPI (FiLM 강건성 — optional)

    Args:
        real_data: (n, genes, pca_k) → 내부에서 (n, genes*pca_k)로 flatten
        syn_data: (m, genes, pca_k) → 동일
        real_labels: (n,) optional, 인구군별 분석 시 필요
        syn_labels: (m,) optional
    """
    # ── Flatten (PCA 텐서 → 1D 벡터) ──
    X = real_data.reshape(real_data.shape[0], -1).astype(np.float64)
    Y = syn_data.reshape(syn_data.shape[0], -1).astype(np.float64)

    results = {}

    # 1. DUPI
    dupi_result = compute_dupi(X, Y, k=k, metric=metric)
    results['dupi'] = dupi_result

    # 2. UI / PI
    ui_pi = compute_ui_pi(dupi_result['dupi'], dupi_result['benchmark'])
    results['ui_pi'] = ui_pi

    # 3. NNAA
    nnaa_result = compute_nnaa(X, Y, metric=metric)
    results['nnaa'] = nnaa_result

    # 4. 인구군별 DUPI (레이블이 있을 때만)
    if real_labels is not None and syn_labels is not None:
        pop_dupi = compute_dupi_per_population(X, Y, real_labels, syn_labels, k=k, metric=metric)
        results['per_population_dupi'] = pop_dupi

    # ── 종합 판정 ──
    dupi_val = dupi_result['dupi']
    bench = dupi_result['benchmark']
    nnaa_val = nnaa_result['nnaa']

    if dupi_result['gap'] < 0.05 and nnaa_result['gap'] < 0.05:
        verdict = "GOOD — utility-privacy 균형 양호"
    elif dupi_val > bench + 0.1:
        verdict = "WARNING — 원본에 너무 가까움 (privacy risk)"
    elif dupi_val < bench - 0.1:
        verdict = "WARNING — 원본에서 너무 멀어짐 (utility loss)"
    else:
        verdict = "ACCEPTABLE — 벤치마크 근처"

    results['verdict'] = verdict

    logger.info(f"\n{'='*60}")
    logger.info(f"프라이버시 평가 결과:")
    logger.info(f"  DUPI^{{<{k}>}} = {dupi_val:.4f}  (benchmark = {bench:.4f}, gap = {dupi_result['gap']:.4f})")
    logger.info(f"  NNAA       = {nnaa_val:.4f}  (ideal = 0.5, gap = {nnaa_result['gap']:.4f})")
    logger.info(f"  UI = {ui_pi['ui']:.4f},  PI = {ui_pi['pi']:.4f},  UI×PI = {ui_pi['ui_pi_product']:.4f}")
    logger.info(f"  판정: {verdict}")
    logger.info(f"{'='*60}\n")

    return results
```

#### 2.4.5 시각화: UI-PI 플롯 (모델 간 비교)

```python
# src/evaluation/privacy_plot.py

import matplotlib.pyplot as plt
import numpy as np

def plot_ui_pi_comparison(model_results: dict, save_path: str = None):
    """
    여러 모델의 UI-PI 좌표를 하나의 2D 플롯에 표시

    Args:
        model_results: {모델명: evaluate_privacy() 결과}
        save_path: 저장 경로 (None이면 plt.show())

    예:
        plot_ui_pi_comparison({
            'GeneDiffusion': baseline_results,
            'HybridGenoDiT': proposed_results,
            'HybridGenoDiT+AuxLoss': aux_results,
        }, save_path='figures/ui_pi_comparison.png')
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # 최적점 표시
    tau = 5.0
    opt_val = np.arctan(tau / 2) / np.arctan(tau)
    ax.scatter([opt_val], [opt_val], marker='*', s=200, c='gold',
               edgecolors='black', zorder=10, label=f'Optimal ({opt_val:.3f}, {opt_val:.3f})')

    # 각 모델
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, (model_name, result) in enumerate(model_results.items()):
        ui = result['ui_pi']['ui']
        pi = result['ui_pi']['pi']
        dupi = result['dupi']['dupi']
        c = colors[i % len(colors)]
        ax.scatter([ui], [pi], s=100, c=c, zorder=5,
                   label=f'{model_name} (DUPI={dupi:.3f})')
        ax.annotate(model_name, (ui, pi), textcoords="offset points",
                    xytext=(8, 8), fontsize=8)

    ax.set_xlabel('Utility Index (UI)', fontsize=12)
    ax.set_ylabel('Privacy Index (PI)', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='lower left')
    ax.set_title('Utility-Privacy Trade-off (Jeong et al., 2023)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
```

#### 2.4.6 대용량 데이터 최적화

유전형 데이터는 차원이 높으므로 (2504 × ~170K) cdist가 느릴 수 있다.

```python
# 최적화 1: PCA 축소 후 거리 계산 (차원 축소)
from sklearn.decomposition import PCA

def compute_dupi_with_pca_reduction(X_real, Y_syn, n_components=50, k=1):
    """고차원 데이터에서 PCA 축소 후 DUPI 계산 (속도 최적화)"""
    combined = np.vstack([X_real, Y_syn])
    pca = PCA(n_components=n_components)
    combined_pca = pca.fit_transform(combined)

    X_pca = combined_pca[:len(X_real)]
    Y_pca = combined_pca[len(X_real):]

    return compute_dupi(X_pca, Y_pca, k=k, metric='euclidean')


# 최적화 2: 배치 단위 거리 계산 (메모리 절약)
def compute_dupi_batched(X_real, Y_syn, k=1, batch_size=500):
    """메모리 제한 환경에서 배치 단위 DUPI 계산"""
    n = X_real.shape[0]
    all_d_y_k = []
    all_d_x_k = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_x = X_real[start:end]

        # batch → Y_syn 거리
        dist_xy = cdist(batch_x, Y_syn)
        d_y_k = np.min(dist_xy, axis=1) if k == 1 else np.partition(dist_xy, k-1, axis=1)[:, k-1]
        all_d_y_k.append(d_y_k)

        # batch → X_real (자기 제외) 거리
        dist_xx = cdist(batch_x, X_real)
        # 자기 자신 제외: 배치 내 인덱스 → 전체 인덱스
        for i, global_i in enumerate(range(start, end)):
            dist_xx[i, global_i] = np.inf
        d_x_k = np.min(dist_xx, axis=1) if k == 1 else np.partition(dist_xx, k-1, axis=1)[:, k-1]
        all_d_x_k.append(d_x_k)

    d_y_k = np.concatenate(all_d_y_k)
    d_x_k = np.concatenate(all_d_x_k)

    dupi = float(np.mean(d_y_k <= d_x_k))
    benchmark = Y_syn.shape[0] / (n + Y_syn.shape[0] - 1)

    return {'dupi': dupi, 'benchmark': benchmark, 'gap': abs(dupi - benchmark)}
```

#### 2.4.7 wandb 로깅 연동

```python
# src/evaluation/run_evaluation.py 내에서

def log_privacy_to_wandb(privacy_results):
    """프라이버시 평가 결과를 wandb에 로깅"""
    wandb.log({
        # DUPI
        'eval/dupi': privacy_results['dupi']['dupi'],
        'eval/dupi_benchmark': privacy_results['dupi']['benchmark'],
        'eval/dupi_gap': privacy_results['dupi']['gap'],

        # UI / PI
        'eval/utility_index': privacy_results['ui_pi']['ui'],
        'eval/privacy_index': privacy_results['ui_pi']['pi'],
        'eval/ui_pi_product': privacy_results['ui_pi']['ui_pi_product'],

        # NNAA
        'eval/nnaa': privacy_results['nnaa']['nnaa'],
        'eval/nnaa_gap': privacy_results['nnaa']['gap'],

        # 판정
        'eval/privacy_verdict': privacy_results['verdict'],
    })

    # 인구군별 DUPI 테이블
    if 'per_population_dupi' in privacy_results:
        pop_data = privacy_results['per_population_dupi']['per_population']
        table = wandb.Table(columns=["pop_idx", "n_real", "n_syn", "dupi", "benchmark", "gap"])
        for pop_idx, info in sorted(pop_data.items()):
            table.add_data(pop_idx, info['n_real'], info['n_syn'],
                           info['dupi'], info['benchmark'], info['gap'])
        wandb.log({"eval/per_pop_dupi_table": table})
```

### 2.5 다양성 (Diversity)

| 지표 | 측정 | 목표 |
|------|------|------|
| k-mer 엔트로피 | 4-mer, 8-mer 모티프 분포 | real과 유사 | ⚠️ 후속 (SNP 역변환 필요) |

### 2.6 강건성 (Robustness) — **핵심 신규 지표**

```python
def evaluate_robustness(real_data, syn_data, real_labels, syn_labels):
    """
    핵심 실험: 인구군 크기와 생성 품질 간의 관계

    FiLM 가설: 소수 인구군일수록 계층적 임베딩 효과가 크다.
    검증: 인구군 크기(n)와 품질(AF 상관) 간 Pearson r 계산
    → FiLM 적용 후 |r|이 감소하면 = 크기 독립적 품질 = 강건성 확보
    """
    per_pop_quality = {}

    for pop_idx in range(26):
        real_mask = real_labels == pop_idx
        syn_mask = syn_labels == pop_idx

        if real_mask.sum() == 0 or syn_mask.sum() == 0:
            continue

        real_pop = real_data[real_mask]
        syn_pop = syn_data[syn_mask]

        # 인구군별 AF 상관
        real_mean = real_pop.mean(axis=0)
        syn_mean = syn_pop.mean(axis=0)
        af_corr = np.corrcoef(real_mean.flatten(), syn_mean.flatten())[0, 1]

        per_pop_quality[pop_idx] = {
            'n_samples': int(real_mask.sum()),
            'af_correlation': af_corr,
        }

    # 크기 vs 품질 상관
    sizes = [v['n_samples'] for v in per_pop_quality.values()]
    qualities = [v['af_correlation'] for v in per_pop_quality.values()]
    size_quality_corr = np.corrcoef(sizes, qualities)[0, 1]

    return {
        'per_pop_quality': per_pop_quality,
        'size_quality_correlation': size_quality_corr,
        # |r| 이 0에 가까울수록 = 크기 독립적 품질 = 강건
    }
```

---

## 3. Ablation Study

| ID | 실험 | 변경 | 파일명 |
|----|------|------|--------|
| A1 | DiT 제거 | `n_dit_blocks=0` | `configs/ablation_no_dit.yaml` |
| A2 | FiLM 제거 | one-hot 곱셈으로 교체 | `configs/ablation_no_film.yaml` |
| A3 | 계층 제거 | superpop_emb 비활성화 | `configs/ablation_no_hierarchy.yaml` |
| A4 | CNN FiLM 제거 | CNN 블록 FiLM 비활성화 | `configs/ablation_no_cnn_film.yaml` |
| A5 | 보조 손실 제거 | `lambda_aux=0` | `configs/ablation_no_aux.yaml` |
| A6 | Zero-init 제거 | AdaLN → 일반 FiLM | `configs/ablation_no_zero_init.yaml` |

---

## 4. 비교 모델 (Baselines)

| 모델 | 구현 방법 |
|------|----------|
| GeneDiffusion (UnetCombined) | 기존 GeneDiffusion 코드 실행 |
| GeneDiffusion + CFG | cfg_dropout 추가 |
| HybridGenoDiT (제안) | 본 구현 |
| HybridGenoDiT + AuxLoss | 보조 손실 포함 |

---

## 5. 통계 보고 형식

최종 결과 표는 아래 형식을 기본으로 한다.

| 지표 | 보고 형식 | 예 |
|------|----------|----|
| AF 상관 | `mean ± std` across seeds | `0.947 ± 0.006` |
| Recovery Rate | `mean ± std` across seeds | `0.935 ± 0.011` |
| DUPI | `point estimate (95% CI)` | `0.512 (0.488, 0.536)` |
| NNAA | `point estimate (95% CI)` | `0.503 (0.491, 0.516)` |
| 인구군별 지표 | `per-pop table + macro average` | 별도 csv/json |

권장 산출물:

- `outputs/<run>/evaluation/summary_metrics.json`
- `outputs/<run>/evaluation/per_population_metrics.csv`
- `outputs/<run>/evaluation/bootstrap_intervals.json`
