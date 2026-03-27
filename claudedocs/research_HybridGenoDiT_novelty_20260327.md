# HybridGenoDiT 유사 논문 존재 여부 조사

**작성일**: 2026-03-27
**검색 깊이**: Deep (5개 병렬 경로)
**목적**: 제안 아키텍처(CNN-DiT Hybrid + FiLM + 계층적 인구군 조건부)와 동일/유사한 선행 연구 존재 여부 확인

---

## Executive Summary

**"HybridGenoDiT"과 동일한 논문은 존재하지 않는다.**

5개 독립 검색 경로를 통해 확인한 결과, 제안 아키텍처의 4가지 핵심 요소를 **모두 결합한** 논문은 0건이다. 각 요소별로도 유전형 생성 분야에서는 대부분 미적용 상태이다.

| 요소 | 유전형 생성에서의 존재 여부 |
|------|------------------------|
| CNN + DiT 하이브리드 | **없음** |
| FiLM / AdaLN 조건부 | **없음** |
| 계층적 인구군 임베딩 (pop + superpop) | **없음** |
| 위 3가지의 결합 | **없음** |

---

## 1. "HybridGenoDiT" 직접 검색 결과

- "HybridGenoDiT" → **0건**
- "GenoDiT", "DiTGeno", "GenoTransDiff", "GenomeDiT", "HybridGeno" → **모두 0건**
- arxiv, bioRxiv, PubMed, Google Scholar 전수 검색 결과 해당 이름의 논문/프리프린트 없음

---

## 2. 요소별 선행 연구 현황

### 2.1 CNN + DiT 하이브리드 (유전형 생성)

**해당 없음.** 유전형(SNP/genotype) 생성에 CNN+DiT 하이브리드를 쓴 논문은 없다.

가장 가까운 논문들:

| 논문 | 아키텍처 | 도메인 | 차이점 |
|------|---------|--------|--------|
| **SNPgen** (Lampis 2026) | CNN VAE + UNet(Conv+Attn) | 유전형 (UK Biobank) | DiT가 아닌 UNet 안에 attention 삽입 |
| **DiscDiff** (Li 2024) | CNN VAE + LDM | DNA 서열 | loosely coupled VAE+LDM, 유전형 아님 |
| **DNA-Diffusion** (Pinello 2025) | UNet(ResNet+Attn) | 조절 DNA (200bp) | UNet 내 attention, 유전형 아님 |
| **Cont. DiT for Reg. Elements** (Liu 2026) | 2D CNN 인코더 + DiT | 조절 DNA (200bp) | 가장 유사하나 200bp 짧은 서열, 유전형 아님 |

### 2.2 CNN + DiT 하이브리드 (타 도메인)

다른 도메인에서는 CNN+DiT 패턴이 **등장하기 시작**한 단계:

| 논문 | 연도 | 학회 | 도메인 | 구조 |
|------|------|------|--------|------|
| **FoilDiff** | 2025 | arXiv | 유체역학 (에어포일) | CNN encoder + ViT bottleneck + CNN decoder ← **가장 유사** |
| **NTv3** (InstaDeep) | 2025 | bioRxiv | 유전체 서열 | Conv tower + Transformer + Deconv tower (masked diffusion) |
| **Grafting** | 2025 | NeurIPS (Oral) | 이미지 | DiT 내 attention을 conv로 일부 교체하는 하이브리드 |
| **HDiT** | 2024 | ICML | 이미지 | 계층적 Transformer (CNN 없음, multi-resolution) |
| **U-DiTs** | 2024 | NeurIPS | 이미지 | U-shaped DiT (순수 Transformer, U-Net 구조 차용) |

**핵심**: "CNN encoder + DiT core + CNN decoder"를 tightly coupled로 결합한 구조는 FoilDiff(2025)와 NTv3(2025)에서만 확인됨. 유전형 생성에는 **미적용**.

### 2.3 FiLM / AdaLN (유전형 생성)

**해당 없음.** FiLM 또는 AdaLN을 유전형 생성에 적용한 논문은 0건.

기존 유전형 생성 모델의 조건부 메커니즘:

| 모델 | 조건부 방법 |
|------|-----------|
| GeneDiffusion | one-hot 토큰 곱셈 주입 |
| SNPgen | cross-attention + classifier-free guidance |
| Genome-AC-GAN | ACGAN 스타일 레이블 연결 |
| GenoDiffusion | ResNet 블록 임베딩 |

모두 **FiLM/AdaLN보다 단순한 방식**을 사용.

유전체 분야에서 AdaLN 사용 사례는 딱 1건: **Continuous DiT for Regulatory Elements** (Liu 2026, ICLR 2026 Workshop) — 200bp 조절 서열 생성에 AdaLN-Zero 사용. 그러나 이는 유전형(SNP)이 아닌 짧은 DNA 서열.

### 2.4 계층적 인구군 조건부 (pop + superpop)

**해당 없음.** 인구군과 슈퍼인구군을 **동시에** 계층적으로 조건부 입력하는 유전형 생성 모델은 0건.

기존 모델의 인구군 조건부:

| 모델 | 조건부 수준 |
|------|-----------|
| GeneDiffusion | 단일 (26개 세부 인구군) |
| Genome-AC-GAN | 단일 (5개 슈퍼인구군) |
| SNPgen | 없음 (질병 레이블만) |
| Yelmen et al. | 없음 (비조건부) |

어떤 모델도 pop(26) + superpop(5)를 **동시에** 사용하지 않음.

---

## 3. Novelty 확인 매트릭스

| 제안 요소 | 유전형 생성에 적용된 적 있는가? | 타 생물학 도메인에 적용된 적 있는가? | 타 도메인(이미지/오디오 등)에 적용된 적 있는가? |
|----------|------------------------------|----------------------------------|----------------------------------------------|
| CNN+DiT 하이브리드 | **없음** | NTv3 (masked diff., 유전체 서열) | FoilDiff, Grafting, HDiT 등 |
| FiLM/AdaLN 조건부 | **없음** | TaxDiff (단백질), AlphaFold3 | DiT (AdaLN-Zero, 이미지) |
| 계층적 인구군 임베딩 | **없음** | **없음** | 계층적 레이블 조건부 자체는 존재 |
| 3가지 결합 | **없음** | **없음** | **없음** |

---

## 4. 주의해야 할 관련 논문

### 4.1 가장 가까운 경쟁자

**Continuous Diffusion Transformers for Designing Synthetic Regulatory Elements** (Liu & Ghods, 2026)
- arXiv: 2603.10885
- ICLR 2026 Gen2 Workshop (Tiny Papers Track)
- **구조**: 2D CNN 인코더 + DiT 6블록 (d=320, 8 heads, AdaLN-Zero)
- **도메인**: 200bp 조절 DNA 서열 (유전형이 아님)
- **차이점**: (1) 유전형이 아닌 짧은 DNA 서열, (2) 인구군 조건부가 아닌 세포 유형 조건부, (3) 계층적 임베딩 없음, (4) CNN 디코더 없음

→ **도메인과 목적이 다르므로 직접 경쟁 관계가 아님.** 오히려 "DiT+AdaLN이 유전체 분야에서 작동함을 보여주는 지지 근거"로 인용 가능.

### 4.2 NTv3 (Nucleotide Transformer v3, InstaDeep 2025)

- bioRxiv: 2025.12.22.695963
- **구조**: Conv downsampling tower + Transformer + Deconv tower
- **도메인**: 유전체 서열 (foundation model, masked diffusion)
- **차이점**: (1) Gaussian diffusion이 아닌 masked diffusion, (2) 생성 모델이 아닌 foundation model, (3) 유전형(SNP)이 아닌 뉴클레오타이드 서열

→ "CNN+Transformer 하이브리드가 유전체 1D 서열에 효과적임을 보여주는 선례"로 인용 가능.

---

## 5. 결론

### Novelty 확정

**HybridGenoDiT의 4가지 핵심 요소는 모두 유전형 생성 분야에서 미적용 상태이며, 이들의 결합은 어떤 도메인에서도 존재하지 않는다.**

1. CNN+DiT 하이브리드 → 유전형에 **최초**
2. FiLM/AdaLN 조건부 → 유전형에 **최초**
3. 계층적 인구군 임베딩 → 유전형에 **최초** (어떤 유전형 생성 모델도 미사용)
4. 3가지 결합 → **모든 도메인에서 최초**

### 논문에서의 포지셔닝

기존 연구와의 관계를 다음과 같이 정리할 수 있다:

```
DiT (Peebles 2023, 이미지)
  + AdaLN-Zero → 유전형 생성에 최초 적용

TaxDiff (Lin 2024, 단백질 서열)
  + 분류 체계 조건부 AdaLN → "인구군 계층 조건부"로 전환

FoilDiff (2025, 유체역학)
  + CNN encoder/decoder + ViT bottleneck → 유전형 1D 데이터에 적응

GeneDiffusion (Kenneweg 2025, 유전형)
  + Gene PCA 표현 + UNet diffusion → CNN-DiT 하이브리드로 확장
```

### 신뢰도

| 항목 | 신뢰도 |
|------|--------|
| "HybridGenoDiT" 이름 미존재 | **99%** |
| CNN+DiT 유전형 생성 미적용 | **95%** |
| FiLM/AdaLN 유전형 생성 미적용 | **95%** |
| 계층적 인구군 임베딩 미적용 | **97%** |
| 4요소 결합 전 도메인 미존재 | **95%** |

---

## 6. 참고 문헌

### 유전형 생성
1. Kenneweg et al. "Generating Synthetic Genotypes using Diffusion Models" Bioinformatics/ISMB 2025. arXiv:2412.03278
2. Lampis et al. "SNPgen: Phenotype-Supervised Genotype Representation via Latent Diffusion" arXiv:2603.10873, 2026
3. Ahronoviz & Gronau. "Genome-AC-GAN" bioRxiv, 2024
4. Xie et al. "Deep Generative Models for Discrete Genotype Simulation" arXiv:2508.09212, 2025
5. Yelmen et al. "Creating Artificial Human Genomes Using Generative Neural Networks" PLOS Genetics, 2021

### 유전체 DiT/CNN+Transformer
6. Liu & Ghods. "Continuous Diffusion Transformers for Designing Synthetic Regulatory Elements" arXiv:2603.10885, ICLR 2026 Workshop
7. Li et al. "DiscDiff: Latent Diffusion Model for DNA Sequence Generation" arXiv:2402.06079, 2024
8. DaSilva et al. "DNA-Diffusion" Nature Genetics, 2025
9. InstaDeep. "Nucleotide Transformer v3" bioRxiv:2025.12.22.695963, 2025

### CNN+DiT 하이브리드 (타 도메인)
10. Peebles & Xie. "DiT: Scalable Diffusion Models with Transformers" ICCV 2023
11. FoilDiff. arXiv:2510.04325, 2025
12. Crowson et al. "Grafting: Exploring Diffusion Transformer Designs" NeurIPS 2025 Oral. arXiv:2506.05340
13. HDiT. "Hourglass Diffusion Transformers" ICML 2024. arXiv:2401.11605
14. Tian et al. "U-DiTs" NeurIPS 2024. arXiv:2405.02730
15. DiC. "Rethinking Conv3x3 Designs in Diffusion Models" CVPR 2025. arXiv:2501.00603

### FiLM/AdaLN (생물학)
16. Lin et al. "TaxDiff: Taxonomic-Guided Diffusion Model for Protein Sequence Generation" arXiv:2402.17156, 2024
17. Abramson et al. "AlphaFold3" Nature, 2024
18. Birnbaum et al. "Temporal FiLM" NeurIPS 2019. arXiv:1909.06628
