# HiPoDiT-T 학습량 스케일업 측정 — 2026-09-16

결과 보고서 `docs/reports/hipodit_t_results_20260916.md` §13 항목 3의 후속 측정이다. 그 보고서
§5.3은 1,000 update 진단 설정에서 **end-to-end 오차를 디코더가 아니라 latent generator가 지배한다**고
결론지었다. 그 결론이 학습량을 올려도 유지되는지를 본다.

**이 문서는 확증이 아니다.** 사전등록(`docs/superpowers/plans/2026-09-15-hipodit-ld.md` 개정 2 §9.4)은
test split을 Gate 3′에서 한 번만 열도록 규정했고 그 한 번은 이미 소진됐다. 따라서 아래 모든 수치는
**dev(val) split** 이며, Gate 판정이 아니라 진단이다. 새 구조도, 새 하이퍼파라미터 선택도 하지 않았다.

## 설정

동결 패널(`outputs/diagnostics/hipodit_fisher_20260915_unique`)과 Gate 1′ 산출물에서 로드한 동결
디코더(`outputs/diagnostics/hipodit_t_oracle_20260915_pooled`)를 그대로 쓴다. 디코더는 재적합하지
않았고 diffusion kernel과 GLM-PCA도 바뀌지 않았다. 바뀐 값은 `--steps` 하나뿐이다.

update 수 1,000 / 5,000 / 20,000 / 80,000 × seed 20260327·20260328·20260329 = 12회.
나머지는 Gate 2′와 동일하다(batch 64, samples 251, DDIM 100 step, standard schedule).
산출물은 `outputs/training/hipodit_t_scaleup_20260916/`에 있고, 동결 기록인
`outputs/diagnostics/`는 건드리지 않았다.

**재현성 확인.** 1,000 update 실행은 Gate 2′ 설정과 같으며, 그 실행의 af/cohort AF/het MAE가
동결된 Gate 2′ 보고서 값과 **16자리까지 동일**했다. 현재 HEAD에서 기존 end-to-end 기록이 그대로
재현된다는 뜻이다.

## 결과 (seed 3개 평균, dev)

| updates | seeds | 마지막 loss | AF MAE | cohort AF MAE | het MAE | 런타임(초) |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 3 | 0.1621 | 0.023678 | 0.070364 | 0.064367 | 23 |
| 5,000 | 3 | 0.1312 | 0.022409 | 0.067159 | 0.064345 | 100 |
| 20,000 | 3 | 0.0784 | 0.017919 | 0.065125 | 0.055961 | 394 |
| 80,000 | 3 | 0.0491 | 0.017327 | 0.063632 | 0.036368 | 1682 |

| updates | AF (2)/(1) | cohort AF (2)/(1) | het (2)/(1) |
|---:|---:|---:|---:|
| 1,000 | 14.0배 | 11.8배 | 1.64배 |
| 5,000 | 28.9배 | 10.0배 | 1.58배 |
| 20,000 | 66.9배 | 10.0배 | 1.16배 |
| 80,000 | 18.2배 | 7.8배 | 0.42배 |

두 번째 표의 (1)과 (2)는 §5.3의 분해를 그대로 따른다. (1) = 같은 generated latent에서 B0 − T,
즉 디코더 선택이 만드는 차이. (2) = 같은 T 디코더에서 generated − real latent, 즉 생성기가 만드는
차이. 비율이 크면 생성기가 지배한다.

## 읽는 법

1. **학습은 실제로 진행됐다.** 마지막 loss가 0.162 → 0.049로 3.3배 줄었고, 지표도 같이 움직인다.
   80,000 update에서 AF MAE는 27 %, heterozygosity MAE는 44 % 개선된다.
2. **§5.3의 결론은 allele frequency에서 유지된다.** cohort AF의 생성기/디코더 비율은 11.8배에서
   7.8배로 완만히 줄 뿐이며, 학습량을 80배 늘려도 여전히 생성기가 한 자릿수 배수로 지배한다.
   cohort AF MAE 자체도 0.0704 → 0.0636, 10 % 개선에 그친다.
3. **heterozygosity에서는 결론이 뒤집힌다.** 비율이 1.64 → 0.42로 1 아래를 통과한다. 즉 80,000
   update에서는 het 오차에 관한 한 **디코더 선택이 생성기보다 더 큰 차이를 만든다.** 이는 het가
   `τ_j`가 직접 조작하는 양이라는 §10 항목 11의 지적과 일관되며, 학습량을 올리면 생성기 쪽 기여가
   먼저 소진되기 때문으로 읽는 것이 자연스럽다.
4. **수확체감이 보인다.** AF MAE는 20,000 → 80,000에서 0.017919 → 0.017327로 3 % 개선에 그친다.
   반면 het는 같은 구간에서 계속 개선된다. 학습량을 더 올린다면 AF가 아니라 het가 움직일 여지다.

## 주장하지 않는 것

이 측정은 dev split 한 번의 진단이며, 결과 보고서 §11의 주장 범위를 넓히지 않는다. 특히 어떤 수치도
Gate 판정으로 쓰지 않는다. LD 관련 주장은 여전히 금지다(개정 2 §9.5). seed 3개는 분산 추정에
충분하지 않으므로 신뢰구간을 붙이지 않았다.

## 재현

```bash
CUDA_VISIBLE_DEVICES=0 /home/user/Envs/csdi/bin/python scripts/hipodit_rebuild_check.py train \
  --output-dir outputs/diagnostics/hipodit_fisher_20260915_unique \
  --run-dir outputs/training/hipodit_t_scaleup_20260916/seed_<SEED>_steps_<STEPS> \
  --decoder-dir outputs/diagnostics/hipodit_t_oracle_20260915_pooled \
  --seed <SEED> --schedule standard --steps <STEPS> --batch-size 64 --samples 251 \
  --ddim-steps 100 --eval-split val --device cuda
```

STEPS ∈ {1000, 5000, 20000, 80000}, SEED ∈ {20260327, 20260328, 20260329}.
