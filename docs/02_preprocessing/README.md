# Phase 1: 데이터 전처리 파이프라인

---

## 1. 전체 흐름

```
ALL.autosomes.phase3.genotypes.vcf.gz (13.9 GB)
  + integrated_call_samples_v3.20130502.ALL.panel (인구군 레이블)
         │
    [Step 1] VCF 파싱 + 유전자 어노테이션 (22 염색체 병렬)
         │
    [Step 2a] PCA 최적 성분 수(K) 그리드 서치 (멀티스레딩)
         │   후보: [4, 6, 8, 10, 12, 16]
         │   기준: 평균 explained variance ≥ 90%
         │
    [Step 2b] 최적 K로 전체 Gene-level PCA (joblib 병렬)
         │
    [Step 3] 토큰화 + 정규화 + zero_mask 생성
         │
    [Step 4] Train/Test 분할 + 계층적 레이블 생성
         │
         ▼
    data/processed/
    ├── pca_grid_search_results.csv   K별 설명 분산 요약
    ├── pca_per_gene_stats.csv        유전자별 PCA 상세 통계
    ├── pca_information_loss_analysis.json  정보 손실 분석
    ├── gene_pca_features.pkl     (2504, ~21819, K)  K=그리드서치 결과
    ├── labels.pkl                (2504,) int64
    ├── label_hierarchy.pkl       pop→superpop 매핑
    ├── train_data.pkl            90% 분할
    ├── test_data.pkl             10% 분할
    ├── split_manifest.json       split seed, 비율, 샘플 ID 기록
    ├── normalization_stats.pkl   (mean, std) fp32
    └── zero_mask.pt              (26624, K) bool
```

---

## 2. Step 1: VCF 파싱 (22 염색체 병렬)

### 병렬화 전략

```python
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor

N_WORKERS = 22  # 염색체당 1 워커

def process_chromosome(args):
    """단일 염색체 처리 (worker function)"""
    chrom_num, gene_list, vcf_path = args
    # ... VCF 읽기, MAF 필터, 유전자 매핑 ...
    return gene_matrices, sample_ids, chrom_num

# 병렬 실행
with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = [executor.submit(process_chromosome, task) for task in tasks]
    results = [f.result() for f in futures]
```

### 필터 조건

| 필터 | 값 | 근거 |
|------|-----|------|
| Biallelic SNP only | `len(ALT)==1, len(REF)==1, len(ALT[0])==1` | 다중 대립유전자 제외 |
| MAF ≥ 0.01 | `min(af, 1-af) >= 0.01` | 희귀 변이 제거 |
| 유전자당 최대 변이 수 | 500 | 계산량 제한 |
| Missing rate | mean imputation | 결측값 처리 |

### 에러 처리

```python
def process_chromosome(args):
    chrom_num, gene_list, vcf_path = args

    if not Path(vcf_path).exists():
        logger.warning(f"[chr{chrom_num}] VCF not found: {vcf_path}")
        return {}, [], chrom_num

    try:
        vcf = VCF(vcf_path)
    except Exception as e:
        logger.error(f"[chr{chrom_num}] VCF open failed: {e}")
        return {}, [], chrom_num

    # ... 처리 로직 ...

    for variant in vcf:
        try:
            # 변이별 처리
            gt = variant.gt_types
            dosage = gt.copy().astype(np.float32)
            dosage[gt == 3] = 2.0      # unknown homozygous → 2.0
            dosage[gt == 2] = np.nan    # missing → NaN

            if np.all(np.isnan(dosage)):
                continue
            # ...
        except Exception:
            continue  # 개별 변이 오류는 스킵

    return gene_matrices, sample_ids, chrom_num
```

---

## 3. Step 2: Gene-level PCA (최적 성분 수 자동 탐색 + 유전자별 병렬)

### 3.1 PCA의 역할과 생물학적 의미

```
유전자 1개 (예: BRCA1)
  └→ 해당 유전자 영역 내 SNP 수: 2~500개
  └→ 2,504명 × N_snps dosage 행렬
  └→ PCA 압축 → K개 주성분

각 주성분의 의미:
  PC1: 해당 유전자의 인구군 간 주요 분화 축 (가장 큰 변이 방향)
  PC2: PC1에 직교하는 두 번째 분화 축
  ...
  PCK: K번째 변이 방향

→ K가 클수록 정보 보존↑, 차원↑, 모델 부담↑, 패딩 0 비율↑
→ K가 작을수록 정보 손실↑, 차원↓, 모델 부담↓, 압축 효율↑
```

**핵심 trade-off**:

| 성분 수(K) | 정보 보존 | 모델 입력 차원 | zero_mask 비율 | 학습 부담 |
|-----------|----------|--------------|---------------|----------|
| 4 | ? | (4, gene_size) | 낮음 | 가벼움 |
| 8 | ? | (8, gene_size) | 중간 | 기준 |
| 12 | ? | (12, gene_size) | 높음 | 무거움 |
| 16 | ? | (16, gene_size) | 매우 높음 | 매우 무거움 |

**물음표(?)인 이유**: 유전체 데이터의 분산 구조는 일반 데이터와 다르다.
SNP 간 LD(연관 불균형)가 약한 유전자는 분산이 넓게 분산되어 K=8로도 60%일 수 있고,
LD가 강한 유전자는 K=4로도 95%를 설명할 수 있다.
**→ 임계값을 하드코딩하지 않고, 데이터가 스스로 최적 K를 결정하게 한다.**

---

### 3.2 PCA 최적 성분 수 그리드 서치

#### 전략: Elbow + Marginal Gain 기반 자동 선택

**90% 같은 절대 임계값을 쓰지 않는 이유:**
- 유전체에서 Gene-level PCA의 explained variance는 유전자의 LD 구조에 크게 좌우됨
- 변이 수 200+인 유전자는 K=16으로도 80%에 미달할 수 있음 (분산이 넓게 퍼짐)
- 변이 수 5 이하인 유전자는 K=4로 100% → 평균을 왜곡
- **절대 임계값 대신 "K를 늘렸을 때의 한계 이득(marginal gain)"이 수렴하는 지점**을 찾는다

```
선택 기준: Marginal Gain 수렴
  marginal_gain(K) = mean_explained(K) - mean_explained(K-step)

  K=4→6: gain = 0.12  (큰 개선)
  K=6→8: gain = 0.06  (보통 개선)
  K=8→10: gain = 0.03 (작은 개선)  ← elbow 근처
  K=10→12: gain = 0.02
  K=12→16: gain = 0.01

  → gain < MARGINAL_GAIN_THRESHOLD(기본 0.03) 이 처음 되는 K를 선택
  → 또는 gain이 직전 대비 50% 이하로 떨어지는 K를 선택

이 방식의 장점:
  - 데이터의 실제 분산 구조에 적응
  - 유전체에서 90%가 비현실적이어도 문제 없음
  - elbow point가 자연스럽게 정보 보존 vs 차원 사이 균형을 잡음
```

```
후보 K: [4, 6, 8, 10, 12, 16]
평가 기준: K별 marginal gain이 수렴하는 elbow point
보조 기준: 변이 수 구간별 가중 평균 explained variance
방법: 각 K를 멀티스레딩으로 동시에 평가 → elbow 자동 감지
```

#### 구현

```python
from concurrent.futures import ThreadPoolExecutor
from sklearn.decomposition import PCA
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────
# 그리드 서치 후보
# ──────────────────────────────────────────────────────
PCA_CANDIDATES = [4, 6, 8, 10, 12, 16]
MARGINAL_GAIN_THRESHOLD = 0.03  # gain이 이 이하로 떨어지면 수렴으로 판단
MARGINAL_GAIN_DECAY_RATIO = 0.5  # gain이 직전 대비 50% 이하면 수렴


def evaluate_pca_k_for_gene(gene_name, matrix, k):
    """
    단일 유전자에 대해 특정 K값의 explained variance 계산

    Args:
        gene_name: 유전자 이름
        matrix: (n_samples, n_variants) dosage 행렬
        k: PCA 성분 수 후보
    Returns:
        (gene_name, k, explained_ratio, actual_k)
    """
    n_vars = matrix.shape[1]
    n_samples = matrix.shape[0]
    actual_k = min(k, n_vars, n_samples)

    if actual_k < 2:
        return (gene_name, k, 0.0, 0)

    try:
        pca = PCA(n_components=actual_k)
        pca.fit(matrix)
        explained = float(np.sum(pca.explained_variance_ratio_))
        return (gene_name, k, explained, actual_k)
    except Exception as e:
        logger.warning(f"PCA eval failed for {gene_name} (k={k}): {e}")
        return (gene_name, k, 0.0, 0)


def grid_search_optimal_pca(gene_matrices, candidates=PCA_CANDIDATES,
                             marginal_threshold=MARGINAL_GAIN_THRESHOLD,
                             decay_ratio=MARGINAL_GAIN_DECAY_RATIO,
                             n_sample_genes=500):
    """
    그리드 서치: 최적 PCA 성분 수(K) 자동 결정

    전략 (Marginal Gain Elbow):
    1. 대표 유전자 N개를 랜덤 샘플링 (전체 ~21,819개 중 500개로 빠르게 탐색)
    2. 각 K 후보에 대해 멀티스레딩으로 동시 평가
    3. K를 늘렸을 때의 marginal gain이 수렴하는 elbow point를 자동 감지
    4. 전체 유전자에 대해 선택된 K로 최종 PCA 실행

    선택 기준 (절대 임계값 90%를 사용하지 않음):
    - 조건 1: marginal_gain(K→K+step) < marginal_threshold (기본 0.03) → 직전 K 선택
    - 조건 2: gain이 직전 대비 decay_ratio (기본 50%) 이하로 급감 → 직전 K 선택
    - 둘 중 먼저 발생하는 지점이 elbow

    Args:
        gene_matrices: {gene_name: (n_samples, n_variants)} 딕셔너리
        candidates: K 후보 리스트 [4, 6, 8, 10, 12, 16]
        marginal_threshold: marginal gain 수렴 임계값 (기본 0.03)
        decay_ratio: gain 급감 판정 비율 (기본 0.5)
        n_sample_genes: 탐색용 샘플 유전자 수 (기본 500)
    Returns:
        optimal_k: 최적 성분 수
        search_results: K별 상세 결과 DataFrame
    """
    all_genes = list(gene_matrices.keys())

    # 대표 유전자 샘플링 (변이 수 분포를 보존하는 stratified 샘플링)
    if len(all_genes) > n_sample_genes:
        np.random.seed(42)
        sample_genes = np.random.choice(all_genes, n_sample_genes, replace=False)
    else:
        sample_genes = all_genes

    logger.info(f"PCA 그리드 서치 시작: K 후보={candidates}, "
                f"샘플 유전자={len(sample_genes)}/{len(all_genes)}")

    # ── 멀티스레딩으로 모든 (유전자, K) 조합 동시 평가 ──
    tasks = []
    for gene_name in sample_genes:
        for k in candidates:
            tasks.append((gene_name, gene_matrices[gene_name], k))

    logger.info(f"총 {len(tasks)}개 태스크 ({len(sample_genes)} 유전자 × {len(candidates)} K값)")

    results = []
    with ThreadPoolExecutor(max_workers=min(32, len(candidates))) as executor:
        futures = [executor.submit(evaluate_pca_k_for_gene, *task) for task in tasks]
        for future in futures:
            results.append(future.result())

    # ── K별 통계 집계 ──
    import pandas as pd
    results_df = pd.DataFrame(results,
                               columns=['gene', 'k', 'explained_ratio', 'actual_k'])

    k_summary = {}
    for k in candidates:
        k_data = results_df[results_df['k'] == k]
        valid = k_data[k_data['actual_k'] > 0]

        mean_explained = valid['explained_ratio'].mean()
        median_explained = valid['explained_ratio'].median()
        p10_explained = valid['explained_ratio'].quantile(0.10)  # 하위 10%
        p25_explained = valid['explained_ratio'].quantile(0.25)
        genes_above_threshold = (valid['explained_ratio'] >= threshold).sum()
        pct_above_threshold = genes_above_threshold / len(valid) * 100

        k_summary[k] = {
            'mean_explained': mean_explained,
            'median_explained': median_explained,
            'p10_explained': p10_explained,
            'p25_explained': p25_explained,
            'pct_genes_above_threshold': pct_above_threshold,
            'n_valid_genes': len(valid),
        }

        logger.info(
            f"  K={k:2d}: mean={mean_explained:.4f}, median={median_explained:.4f}, "
            f"p10={p10_explained:.4f}, ≥{threshold:.0%}인 유전자={pct_above_threshold:.1f}%"
        )

    # ── 최적 K 선택: Elbow (Marginal Gain 수렴) 기반 ──
    sorted_k = sorted(candidates)
    gains = {}
    for i in range(1, len(sorted_k)):
        prev_k, curr_k = sorted_k[i - 1], sorted_k[i]
        gain = k_summary[curr_k]['mean_explained'] - k_summary[prev_k]['mean_explained']
        gains[curr_k] = gain
        logger.info(f"  K={prev_k}→{curr_k}: marginal gain = {gain:.4f}")

    # 방법 1: gain < MARGINAL_GAIN_THRESHOLD 이 처음 되는 직전 K
    # 방법 2: gain이 직전 대비 50% 이하로 떨어지는 직전 K
    # 둘 중 먼저 발생하는 지점 선택
    optimal_k = sorted_k[0]  # fallback: 가장 작은 K
    prev_gain = None
    for i in range(1, len(sorted_k)):
        curr_k = sorted_k[i]
        gain = gains[curr_k]

        # 조건 1: 절대 gain이 임계값 미만
        if gain < MARGINAL_GAIN_THRESHOLD:
            optimal_k = sorted_k[i - 1]
            logger.info(f"  Elbow (절대 gain < {MARGINAL_GAIN_THRESHOLD}): K={optimal_k}")
            break

        # 조건 2: gain이 직전 대비 급감
        if prev_gain is not None and gain < prev_gain * MARGINAL_GAIN_DECAY_RATIO:
            optimal_k = sorted_k[i - 1]
            logger.info(f"  Elbow (gain 급감 {prev_gain:.4f}→{gain:.4f}): K={optimal_k}")
            break

        prev_gain = gain
        optimal_k = curr_k  # 아직 수렴 안 됨 → 계속 증가
    else:
        # 끝까지 수렴하지 않음 → 마지막 K 사용
        logger.warning(f"  Marginal gain이 수렴하지 않음. 최대 K={optimal_k} 사용")

    logger.info(f"\n{'='*60}")
    logger.info(f"최적 PCA 성분 수: K={optimal_k}")
    logger.info(f"  평균 설명 분산: {k_summary[optimal_k]['mean_explained']:.4f}")
    logger.info(f"  ≥{threshold:.0%} 유전자 비율: "
                f"{k_summary[optimal_k]['pct_genes_above_threshold']:.1f}%")
    logger.info(f"{'='*60}\n")

    # ── 결과 저장 ──
    summary_df = pd.DataFrame(k_summary).T
    summary_df.index.name = 'k'
    summary_df.to_csv("data/processed/pca_grid_search_results.csv")

    # 상세 결과도 저장 (논문 Supplementary용)
    results_df.to_csv("data/processed/pca_grid_search_detail.csv", index=False)

    return optimal_k, summary_df
```

#### 실행 예시 (예상 출력 — 실제 값은 데이터 의존)

```
PCA 그리드 서치 시작: K 후보=[4, 6, 8, 10, 12, 16], 샘플 유전자=500/21819
총 3000개 태스크 (500 유전자 × 6 K값)

  K= 4: mean=0.6834, median=0.7101, p10=0.4212
  K= 6: mean=0.7856, median=0.8112, p10=0.5634
  K= 8: mean=0.8512, median=0.8734, p10=0.6656
  K=10: mean=0.8945, median=0.9101, p10=0.7312
  K=12: mean=0.9134, median=0.9278, p10=0.7856
  K=16: mean=0.9456, median=0.9534, p10=0.8412

  Marginal Gain 분석:
  K=4→6:  gain = 0.1022  (큰 개선)
  K=6→8:  gain = 0.0656  (보통)
  K=8→10: gain = 0.0433  (보통)
  K=10→12: gain = 0.0189 (< 0.03 threshold)  ← Elbow!

============================================================
최적 PCA 성분 수: K=10
  선택 근거: marginal gain 수렴 (K=10→12에서 0.0189 < 0.03)
  평균 설명 분산: 0.8945
  중앙 설명 분산: 0.9101
============================================================

주의: 위 수치는 예시입니다.
  - LD가 강한 유전자(예: HLA 영역): K=4로도 90%+ 가능
  - LD가 약한 유전자(변이 300+, 독립적): K=16으로도 80% 미달 가능
  - 실제 값은 1KG Phase 3 데이터의 LD 구조에 따라 결정됩니다.
```

---

### 3.3 최적 K로 전체 PCA 실행 (유전자별 병렬)

```python
from joblib import Parallel, delayed


def pca_single_gene(gene_name, matrix, n_components):
    """
    단일 유전자에 대해 최적 K로 PCA 수행

    Args:
        gene_name: 유전자 이름
        matrix: (n_samples, n_variants) dosage 행렬
        n_components: 그리드 서치에서 결정된 최적 K
    Returns:
        (features_dict, explained_ratio, n_variants) 또는 None
    """
    n_vars = matrix.shape[1]
    n_comp = min(n_components, n_vars, matrix.shape[0])

    if n_comp < 2:
        logger.debug(f"Skipping {gene_name}: only {n_vars} variants")
        return None

    try:
        pca = PCA(n_components=n_comp)
        transformed = pca.fit_transform(matrix)
        explained = float(np.sum(pca.explained_variance_ratio_))
        per_component = pca.explained_variance_ratio_.tolist()

        features = {}
        for k in range(n_comp):
            features[f"{gene_name}:{k}"] = transformed[:, k]

        return {
            'features': features,
            'explained_total': explained,
            'explained_per_component': per_component,
            'n_variants': n_vars,
            'actual_k': n_comp,
        }
    except Exception as e:
        logger.warning(f"PCA failed for {gene_name}: {e}")
        return None


def run_full_pca(gene_matrices, optimal_k):
    """
    전체 유전자에 대해 최적 K로 PCA 실행 (joblib 병렬)

    Args:
        gene_matrices: {gene_name: matrix} 딕셔너리
        optimal_k: 그리드 서치에서 결정된 최적 성분 수
    Returns:
        features_df: DataFrame (2504, N_features)
        pca_stats_df: DataFrame (유전자별 PCA 통계)
    """
    logger.info(f"전체 PCA 실행: {len(gene_matrices)} 유전자, K={optimal_k}")

    # joblib 병렬 (CPU 전체 코어)
    results = Parallel(n_jobs=-1, backend='loky', verbose=10)(
        delayed(pca_single_gene)(name, matrix, optimal_k)
        for name, matrix in sorted(gene_matrices.items())
    )

    # 결과 취합
    all_features = {}
    pca_stats = []

    for gene_name, result in zip(sorted(gene_matrices.keys()), results):
        if result is None:
            continue
        all_features.update(result['features'])
        pca_stats.append({
            'gene': gene_name,
            'n_variants': result['n_variants'],
            'actual_k': result['actual_k'],
            'explained_total': result['explained_total'],
            **{f'explained_pc{i+1}': v
               for i, v in enumerate(result['explained_per_component'])},
        })

    features_df = pd.DataFrame(all_features)
    pca_stats_df = pd.DataFrame(pca_stats)

    # ── 요약 통계 ──
    logger.info(f"PCA 완료:")
    logger.info(f"  유효 유전자: {len(pca_stats)}/{len(gene_matrices)}")
    logger.info(f"  총 피처 수: {len(all_features)}")
    logger.info(f"  평균 설명 분산: {pca_stats_df['explained_total'].mean():.4f}")
    logger.info(f"  중앙 설명 분산: {pca_stats_df['explained_total'].median():.4f}")
    logger.info(f"  하위 10% 설명 분산: {pca_stats_df['explained_total'].quantile(0.10):.4f}")

    # ── 저장 ──
    pca_stats_df.to_csv("data/processed/pca_per_gene_stats.csv", index=False)
    logger.info(f"  유전자별 통계 저장: data/processed/pca_per_gene_stats.csv")

    return features_df, pca_stats_df
```

---

### 3.4 다운스트림 영향: K 변경 시 자동 전파

PCA 성분 수(K)가 변경되면 다음이 **자동으로** 연쇄 변경되어야 한다:

```
K 결정 (그리드 서치)
  │
  ├→ num_channels = K                     (모델 입력 채널 수)
  ├→ 토큰화: (2504, N_genes, K)           (텐서 shape)
  ├→ zero_mask: (gene_size, K)            (마스크 shape)
  ├→ 모델: CNNStemEncoder(in_channels=K)  (CNN 입력)
  ├→ 모델: gene_size 패딩 계산            (26624는 K=8 기준, 재계산 필요)
  └→ 정규화 통계: (N_genes × K,) 벡터     (mean, std shape)
```

```python
def compute_gene_size(n_genes, optimal_k):
    """
    gene_size를 K와 유전자 수에 맞게 자동 계산

    조건: CNN 다운샘플링(stride 2) 횟수에 맞춰 2의 배수로 정렬
    """
    raw_size = n_genes  # ~21,819
    # 3단계 다운샘플(÷8) 후 패치 크기(16)로 나누어 떨어져야 함
    # → gene_size는 8 × 16 = 128의 배수여야 함
    alignment = 128
    gene_size = ((raw_size + alignment - 1) // alignment) * alignment
    logger.info(f"gene_size: {raw_size} → {gene_size} (aligned to {alignment})")
    return gene_size


def build_config_from_pca(optimal_k, n_genes):
    """그리드 서치 결과에서 다운스트림 설정 자동 생성"""
    gene_size = compute_gene_size(n_genes, optimal_k)
    return {
        'num_channels': optimal_k,
        'gene_size': gene_size,
        'pca_components': optimal_k,
        # 이 값들이 모델, 토큰화, zero_mask에 전파됨
    }
```

---

### 3.5 PCA 정보 손실 분석 (논문 Methods/Supplementary용)

```python
def analyze_pca_information_loss(pca_stats_df, optimal_k):
    """
    PCA 정보 손실 분석 — 논문에 보고할 통계 생성

    출력:
    1. 전체 요약 통계
    2. 변이 수 구간별 설명 분산
    3. 염색체별 설명 분산
    4. 정보 손실이 큰 유전자 목록 (하위 5%)
    """
    stats = {}

    # 1. 전체 요약
    stats['overall'] = {
        'optimal_k': optimal_k,
        'mean_explained': pca_stats_df['explained_total'].mean(),
        'median_explained': pca_stats_df['explained_total'].median(),
        'std_explained': pca_stats_df['explained_total'].std(),
        'min_explained': pca_stats_df['explained_total'].min(),
        'max_explained': pca_stats_df['explained_total'].max(),
        'p5_explained': pca_stats_df['explained_total'].quantile(0.05),
        'p10_explained': pca_stats_df['explained_total'].quantile(0.10),
        'p25_explained': pca_stats_df['explained_total'].quantile(0.25),
        'n_genes_total': len(pca_stats_df),
        'n_genes_above_90pct': (pca_stats_df['explained_total'] >= 0.90).sum(),
        'n_genes_above_95pct': (pca_stats_df['explained_total'] >= 0.95).sum(),
        'n_genes_below_70pct': (pca_stats_df['explained_total'] < 0.70).sum(),
    }

    # 2. 변이 수 구간별 분석
    bins = [0, 5, 10, 20, 50, 100, 200, 500]
    pca_stats_df['variant_bin'] = pd.cut(pca_stats_df['n_variants'], bins=bins)
    stats['by_variant_count'] = (
        pca_stats_df.groupby('variant_bin')['explained_total']
        .agg(['mean', 'median', 'count'])
        .to_dict()
    )

    # 3. 정보 손실 최대 유전자 (하위 5%)
    threshold_5pct = pca_stats_df['explained_total'].quantile(0.05)
    worst_genes = pca_stats_df[
        pca_stats_df['explained_total'] <= threshold_5pct
    ].sort_values('explained_total')
    stats['worst_genes'] = worst_genes[['gene', 'n_variants', 'explained_total']].to_dict('records')

    # 4. 성분별 기여도 분석 (PC1이 몇 % 차지하는지 등)
    pc_cols = [c for c in pca_stats_df.columns if c.startswith('explained_pc')]
    if pc_cols:
        stats['per_component_contribution'] = {
            col: pca_stats_df[col].mean() for col in pc_cols
        }

    # 저장
    with open("data/processed/pca_information_loss_analysis.json", 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    # 논문용 한 줄 요약 출력
    o = stats['overall']
    logger.info(
        f"\n논문 보고용 (Methods):\n"
        f"  \"Gene-level PCA with K={o['optimal_k']} components explained "
        f"{o['mean_explained']:.1%} (mean) / {o['median_explained']:.1%} (median) "
        f"of per-gene variance across {o['n_genes_total']:,} genes. "
        f"{o['n_genes_above_90pct']:,} genes ({o['n_genes_above_90pct']/o['n_genes_total']:.1%}) "
        f"exceeded the 90% threshold.\"\n"
    )

    return stats
```

---

### 3.6 출력 파일 (PCA 관련)

| 파일 | 설명 | 용도 |
|------|------|------|
| `pca_grid_search_results.csv` | K별 (mean, median, p10, 임계값 충족률) | 최적 K 선정 근거 |
| `pca_grid_search_detail.csv` | (유전자, K, explained) 전체 조합 | 상세 분석 |
| `pca_per_gene_stats.csv` | 유전자별 (n_variants, actual_k, explained, PC별) | 논문 Supplementary |
| `pca_information_loss_analysis.json` | 전체 요약 + 구간별 + worst genes | 논문 Methods |
| `gene_pca_features.pkl` | 최종 PCA 피처 DataFrame | 학습 데이터 |

---

## 4. Step 3: 토큰화 + 정규화

### 토큰화

```python
def tokenize_dataset(df, gene_names):
    """
    DataFrame → (N_samples, N_genes, max_pca_components) numpy array

    컬럼명 형식: "geneName:chrN:componentK"
    → "geneName:chrN"로 그룹핑 → 유전자당 1개 토큰 (길이 = max_components)
    """
    # 유전자별 그룹핑
    grouped_names = []
    for name in df.columns:
        parts = name.split(":")
        grouped_names.append(f"{parts[0]}:{parts[1]}")

    unique_genes, counts = np.unique(grouped_names, return_counts=True)
    max_length = int(np.max(counts))  # max PCA components across genes

    tokenized = np.zeros((len(df), len(unique_genes), max_length), dtype=np.float32)

    gene_idx = 0
    comp_idx = 0
    last_gene = grouped_names[0]

    for col_idx, gene_name in enumerate(grouped_names):
        if gene_name != last_gene:
            gene_idx += 1
            comp_idx = 0
            last_gene = gene_name
        for sample_idx in range(len(df)):
            tokenized[sample_idx, gene_idx, comp_idx] = df.iloc[sample_idx, col_idx]
        comp_idx += 1

    return tokenized  # (2504, ~21819, 8)
```

### 정규화 (fp32 통계량 → 학습 시 bf16 캐스팅)

```python
def normalize_data(x_train, x_test):
    """
    통계량은 fp32로 정확히 계산,
    정규화된 데이터는 그대로 저장 (학습 시 bf16 autocast 적용)
    """
    # fp32로 통계량 계산
    xmean = x_train.mean(axis=0).astype(np.float32)
    xstd = x_train.std(axis=0).astype(np.float32)
    xstd[xstd == 0.0] += 1  # zero division 방지

    x_train_norm = (x_train - xmean) / xstd
    x_test_norm = (x_test - xmean) / xstd

    # 통계량 저장 (역변환에 필요)
    stats = {'mean': xmean, 'std': xstd}
    with open("data/processed/normalization_stats.pkl", 'wb') as f:
        pickle.dump(stats, f)

    return x_train_norm, x_test_norm, stats
```

### zero_mask 생성

```python
def generate_zero_mask(x_train, gene_size=26624):
    """
    학습 데이터에서 모든 샘플이 0인 위치를 마스크로 기록
    → 생성 시 이 위치를 0으로 강제
    """
    # 패딩 후 (N, gene_size, 8)
    padded = np.zeros((len(x_train), gene_size, x_train.shape[2]), dtype=np.float32)
    padded[:, :x_train.shape[1], :] = x_train

    # 모든 샘플에서 0인 위치 = True
    zero_mask = np.all(padded == 0, axis=0)  # (gene_size, 8)
    torch.save(torch.tensor(zero_mask), "data/processed/zero_mask.pt")

    n_zeros = zero_mask.sum()
    total = zero_mask.numel()
    logger.info(f"Zero mask: {n_zeros}/{total} ({n_zeros/total*100:.1f}%) positions always zero")

    return zero_mask
```

---

## 5. Step 4: 계층적 레이블 생성

```python
def create_hierarchical_labels(panel_path):
    """
    panel 파일에서 pop(26) + superpop(5) 계층 레이블 생성
    """
    panel = pd.read_csv(panel_path, sep='\t')

    # 슈퍼인구군 매핑
    superpop_to_idx = {sp: i for i, sp in enumerate(sorted(panel['super_pop'].unique()))}
    pop_to_idx = {p: i for i, p in enumerate(sorted(panel['pop'].unique()))}
    pop_to_superpop = {}

    for _, row in panel.drop_duplicates(subset='pop').iterrows():
        pop_to_superpop[pop_to_idx[row['pop']]] = superpop_to_idx[row['super_pop']]

    idx_to_pop = {v: k for k, v in pop_to_idx.items()}
    idx_to_superpop = {v: k for k, v in superpop_to_idx.items()}

    labels = {
        'pop_labels': np.array([pop_to_idx[p] for p in panel['pop']]),           # (2504,) 0-25
        'superpop_labels': np.array([superpop_to_idx[sp] for sp in panel['super_pop']]),  # (2504,) 0-4
        'pop_to_idx': pop_to_idx,
        'idx_to_pop': idx_to_pop,               # 역매핑
        'superpop_to_idx': superpop_to_idx,
        'idx_to_superpop': idx_to_superpop,      # 역매핑
        'pop_to_superpop': pop_to_superpop,      # {pop_idx: superpop_idx}
        'pop_sizes': dict(panel['pop'].value_counts()),
    }

    with open("data/processed/label_hierarchy.pkl", 'wb') as f:
        pickle.dump(labels, f)

    logger.info(f"Populations: {len(pop_to_idx)}, Superpopulations: {len(superpop_to_idx)}")
    logger.info(f"Pop→Superpop mapping: {pop_to_superpop}")

    return labels
```

### 5.1 Train/Test 분할 규칙 (확정)

train/test 분할은 단순 random split이 아니라 **population stratified split**으로 고정한다.

| 항목 | 값 |
|------|-----|
| test ratio | `0.1` |
| split 단위 | sample |
| stratify 기준 | `pop_labels` (26개 인구군) |
| split seed | `20260327` |
| 재사용 정책 | 최초 생성 후 `split_manifest.json`을 저장하고 이후 모든 모델/ablation에서 재사용 |

```python
from sklearn.model_selection import train_test_split
import json

PREPROCESS_SEED = 20260327


def split_dataset_stratified(tokenized, labels, sample_ids, test_ratio=0.1, seed=PREPROCESS_SEED):
    """
    population stratified split

    Args:
        tokenized: (N, n_genes, K)
        labels: dict from create_hierarchical_labels()
        sample_ids: list[str] aligned with tokenized rows
    Returns:
        x_train, x_test, y_train, y_test, split_manifest
    """
    pop_labels = labels['pop_labels']

    idx = np.arange(len(tokenized))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True,
        stratify=pop_labels,
    )

    x_train = tokenized[train_idx]
    x_test = tokenized[test_idx]
    y_train = pop_labels[train_idx]
    y_test = pop_labels[test_idx]

    split_manifest = {
        'seed': seed,
        'test_ratio': test_ratio,
        'n_total': int(len(idx)),
        'n_train': int(len(train_idx)),
        'n_test': int(len(test_idx)),
        'train_indices': train_idx.tolist(),
        'test_indices': test_idx.tolist(),
        'train_sample_ids': [sample_ids[i] for i in train_idx],
        'test_sample_ids': [sample_ids[i] for i in test_idx],
        'train_pop_counts': {
            str(k): int(v) for k, v in pd.Series(y_train).value_counts().sort_index().items()
        },
        'test_pop_counts': {
            str(k): int(v) for k, v in pd.Series(y_test).value_counts().sort_index().items()
        },
    }

    with open("data/processed/split_manifest.json", "w") as f:
        json.dump(split_manifest, f, indent=2)

    return x_train, x_test, y_train, y_test, split_manifest
```

추가 규칙:

- preprocessing 재실행 시 split seed를 바꾸지 않는다.
- baseline / ablation / proposal은 동일 `split_manifest.json`을 공유한다.
- train/test sample ID가 panel 파일 순서와 어긋나면 즉시 `ValueError`를 발생시킨다.

---

## 6. 전처리 실행 스크립트 구조

```python
# src/preprocessing/run_pipeline.py

def main():
    """전처리 전체 파이프라인 (병렬)"""
    t_start = time.time()

    # Step 0: 설정 검증
    validate_input_files()

    # Step 1: 유전자 어노테이션 + VCF 병렬 파싱
    gene_coords = parse_refgene()
    gene_matrices, sample_ids = parallel_vcf_processing(gene_coords)  # 22 workers

    # Step 2a: PCA 최적 성분 수 그리드 서치 (Marginal Gain Elbow, 멀티스레딩)
    optimal_k, search_results = grid_search_optimal_pca(
        gene_matrices,
        candidates=[4, 6, 8, 10, 12, 16],
        marginal_threshold=0.03,    # gain < 0.03이면 수렴 판정
        decay_ratio=0.5,            # gain이 직전 대비 50% 이하면 수렴
        n_sample_genes=500,
    )

    # Step 2b: 최적 K로 전체 PCA (joblib 병렬)
    features_df, pca_stats = run_full_pca(gene_matrices, optimal_k)

    # Step 2c: PCA 정보 손실 분석 (논문용)
    pca_analysis = analyze_pca_information_loss(pca_stats, optimal_k)

    # Step 3: 레이블 + 샘플 정렬
    labels = create_hierarchical_labels(PANEL_FILE)
    features_df, labels = align_samples(features_df, labels, sample_ids)

    # Step 4: 토큰화
    tokenized = tokenize_dataset(features_df)

    # Step 5: stratified split + 정규화
    x_train, x_test, y_train, y_test, split_manifest = split_dataset_stratified(
        tokenized,
        labels,
        sample_ids=sample_ids,
        test_ratio=0.1,
        seed=PREPROCESS_SEED,
    )
    x_train, x_test, stats = normalize_data(x_train, x_test)

    # Step 6: zero_mask
    zero_mask = generate_zero_mask(x_train)

    # Step 7: 저장
    save_all(x_train, x_test, y_train, y_test, stats, zero_mask)

    elapsed = time.time() - t_start
    logger.info(f"전처리 완료: {elapsed:.0f}s ({elapsed/60:.1f}min)")
```

---

## 7. 출력 파일 명세

| 파일 | 형태 | 크기 (추정) | 설명 |
|------|------|------------|------|
| `gene_pca_features.pkl` | DataFrame | ~400 MB | (2504, ~168K) PCA 피처 |
| `processed_tokenized.pkl` | ndarray | ~400 MB | (2504, ~21819, K) |
| `train_data.pkl` | (ndarray, ndarray) | ~360 MB | (x_train, y_train) |
| `test_data.pkl` | (ndarray, ndarray) | ~40 MB | (x_test, y_test) |
| `split_manifest.json` | JSON | ~100 KB | split seed, sample IDs, 인구군 분포 |
| `normalization_stats.pkl` | dict | ~3 MB | mean, std (fp32) |
| `label_hierarchy.pkl` | dict | ~1 KB | pop/superpop 매핑 |
| `zero_mask.pt` | Tensor bool | ~200 KB | (26624, K) |
| `pca_explained_variance.csv` | CSV | ~500 KB | 유전자별 설명 분산 |

---

## 8. 산출물 내부 스키마 (계약 정의)

> 이 섹션은 전처리 산출물의 **정확한 내부 구조**를 정의한다. 다운스트림 모듈(모델, 학습, 추론, 평가)은 이 계약에 의존한다.

### 8.1 K 선택 규칙 (확정)

| 항목 | 값 |
|------|-----|
| **방법** | Marginal Gain Elbow (절대 임계값 90% 미사용) |
| 후보 | `[4, 6, 8, 10, 12, 16]` |
| 조건 1 | `marginal_gain(K→K+step) < 0.03` → 직전 K 선택 |
| 조건 2 | `gain < 직전_gain × 0.5` (급감) → 직전 K 선택 |
| 결정 | 둘 중 먼저 발생하는 elbow point |
| fallback | 끝까지 수렴하지 않으면 최대 K(16) 사용 |
| 전파 | 결정된 K → `num_channels`, `zero_mask` shape, 모델 입력 채널에 자동 반영 |

> **참고**: `gene_size`는 K와 독립적으로 유전자 수(~21,819)에서 128의 배수로 패딩하여 결정된다 (현재 26624). K가 바뀌어도 `gene_size`는 동일하다.

### 8.2 `label_hierarchy.pkl` 내부 필드

```python
{
    # 인구군 매핑
    'pop_to_idx': dict,         # {"ACB": 0, "ASW": 1, ..., "YRI": 25}  (26개)
    'idx_to_pop': dict,         # {0: "ACB", 1: "ASW", ..., 25: "YRI"}  (역매핑)
    'pop_to_superpop': dict,    # {0: 0, 1: 0, ..., 25: 0}  pop_idx → superpop_idx

    # 슈퍼인구군 매핑
    'superpop_to_idx': dict,    # {"AFR": 0, "AMR": 1, "EAS": 2, "EUR": 3, "SAS": 4}
    'idx_to_superpop': dict,    # {0: "AFR", 1: "AMR", ...}

    # 크기 정보
    'pop_sizes': dict,          # {"ACB": 96, "ASW": 61, ..., "YRI": 108}
    'pop_labels': np.ndarray,   # (2504,) int64 — 샘플별 pop 인덱스
    'superpop_labels': np.ndarray,  # (2504,) int64 — 샘플별 superpop 인덱스
}
```

**사용처**:
- `HierarchicalPopulationEmbedding`: `pop_to_superpop`으로 superpop 임베딩 공유
- `PopulationBalancedSampler`: `pop_sizes`로 sqrt 비례 오버샘플링 가중치 계산
- 추론: `pop_to_idx`로 인구군명 → 인덱스 변환

### 8.3 `normalization_stats.pkl` 내부 필드

```python
{
    'mean': np.ndarray,  # (gene_size, K) float32 — 학습 데이터 채널별 평균
    'std': np.ndarray,   # (gene_size, K) float32 — 학습 데이터 채널별 표준편차 (0인 곳은 1로 대체)
}
```

**역정규화** (추론 후처리):
```python
# sample shape: (B, K, gene_size) → 전치 후 broadcasting
x_denorm = sample * std.T + mean.T  # std/mean을 (K, gene_size)로 전치하여 적용
```

### 8.4 `train_data.pkl` / `test_data.pkl` 내부 필드

```python
# pickle.load() 결과: tuple
(
    x_data,   # np.ndarray, shape (N, gene_size, K), dtype float32 — 정규화된 PCA 텐서
    y_labels, # np.ndarray, shape (N,), dtype int64 — 인구군 인덱스 (0~25)
)
# 학습 시 DataLoader에서 x_data를 (N, K, gene_size)로 전치하여 모델에 입력
```

### 8.5 `split_manifest.json` 내부 필드

```json
{
  "seed": 20260327,
  "test_ratio": 0.1,
  "n_total": 2504,
  "n_train": 2253,
  "n_test": 251,
  "train_indices": [0, 1, 2],
  "test_indices": [3, 4, 5],
  "train_sample_ids": ["HG00096", "HG00097"],
  "test_sample_ids": ["HG00100", "HG00101"],
  "train_pop_counts": {"0": 86, "1": 55},
  "test_pop_counts": {"0": 10, "1": 6}
}
```

**용도**:

- 모든 실험에서 동일 split 강제
- sample alignment 추적
- 평가 시 train/test leakage 방지 확인

### 8.6 `zero_mask.pt` 내부 필드

```python
# torch.load() 결과: torch.Tensor
# shape: (gene_size, K), dtype: torch.bool
# True = 모든 학습 샘플에서 항상 0인 위치 (패딩 포함)
# 모델에서는 (K, gene_size)로 전치하여 사용
```

### 8.7 `gene_pca_features.pkl` 내부 필드

```python
# pickle.load() 결과: pd.DataFrame
# shape: (2504, N_features)
# 컬럼명 형식: "geneName:componentIdx" (예: "BRCA1:0", "BRCA1:1", ..., "TP53:7")
# dtype: float32
# 토큰화 전 원본 PCA 피처 (정규화 전)
```
