┌────────────────────────────────────────────────────────────────────────────────────┐
│      Population-Conditional Hybrid Genotype Generator (FiLM + Distribution)       │
└────────────────────────────────────────────────────────────────────────────────────┘

 Input
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │  SNP / Haplotype Vector x                                                       │
 │  - fixed-length 1D sequence                                                     │
 │  - e.g., 10K SNP positions                                                      │
 │                                                                                  │
 │  Condition c                                                                     │
 │  - superpopulation / population label                                            │
 │  - optional metadata                                                             │
 └──────────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                    ┌────────────────────────────────┐
                    │     1D CNN Stem Encoder        │
                    │  - local SNP pattern extraction│
                    │  - short-range LD features     │
                    │  - haplotype chunk signals     │
                    └────────────────────────────────┘
                                   │
                                   ▼
                    ┌────────────────────────────────┐
                    │    Token / Latent Projection   │
                    │  - feature -> latent tokens    │
                    │  - position embedding          │
                    │  - DiT input formatting        │
                    └────────────────────────────────┘
                                   │
                  ┌────────────────┴────────────────┐
                  │                                 │
                  ▼                                 ▼
   ┌────────────────────────────┐     ┌────────────────────────────────────┐
   │ Diffusion Timestep Encoder │     │      Condition Encoder             │
   │ - t embedding              │     │ - population embedding             │
   │ - denoising stage info     │     │ - superpopulation hierarchy        │
   └────────────────────────────┘     │ - optional class prior             │
                  │                   └────────────────────────────────────┘
                  └────────────────┬────────────────┘
                                   ▼
                    ┌────────────────────────────────┐
                    │     FiLM Parameter Generator   │
                    │  - generates γ_l, β_l          │
                    │  - block-wise modulation       │
                    │  - population-aware control    │
                    └────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────────────────┐
│                               Conditional DiT Backbone                             │
│                                                                                    │
│   Shared Transformer Blocks                                                        │
│                                                                                    │
│   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐              │
│   │ Block 1 + FiLM   │→→│ Block 2 + FiLM   │→→│ Block N + FiLM   │              │
│   │ h1' = γ1⊙h1 + β1 │   │ h2' = γ2⊙h2 + β2 │   │ hN' = γN⊙hN + βN │              │
│   └──────────────────┘   └──────────────────┘   └──────────────────┘              │
│                                                                                    │
│   * same backbone shared across all populations                                    │
│   * only modulation coefficients change by condition                               │
└────────────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                    ┌────────────────────────────────┐
                    │     Diffusion Denoising Head   │
                    │  - predict noise / state       │
                    │  - reconstruct genotype tokens │
                    │  - iterative denoising         │
                    └────────────────────────────────┘
                                   │
                                   ▼
                    ┌────────────────────────────────┐
                    │ Generated Sequence x_hat       │
                    │ - synthetic genotype/haplotype │
                    └────────────────────────────────┘
                                   │
                    ┌──────────────┼───────────────┐
                    │              │               │
                    ▼              ▼               ▼
      ┌────────────────────┐ ┌────────────────┐ ┌────────────────────────┐
      │ SFS Matching Head   │ │ LD Matching    │ │ Haplotype Diversity    │
      │ - allele freq dist  │ │ - local r²/dist│ │ - unique pattern dist  │
      │ - windowed spectrum │ │ - pairwise dep │ │ - local diversity      │
      └────────────────────┘ └────────────────┘ └────────────────────────┘
                    │              │               │
                    └──────────────┴───────────────┘
                                   ▼
                    ┌────────────────────────────────┐
                    │ Distribution Matching Loss     │
                    │ L_dist = λ1 L_sfs + λ2 L_ld    │
                    │        + λ3 L_hap              │
                    └────────────────────────────────┘
                                   │
                                   ▼
                    ┌────────────────────────────────┐
                    │ Final Objective                │
                    │ L_total = L_diffusion + L_dist │
                    └────────────────────────────────┘