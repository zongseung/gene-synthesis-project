# ADR-0003 · Stage-1 별도 분류기(classifier/)를 폐기한다

날짜 2026-07-28 · 상태 채택

## 맥락

`src/hanmed_mm/classifier/`(labels.py·model.py·dataset.py·infer.py·calibrate.py·train.py,
약 545줄)는 초기 설계에서 종 분류를 담당할 별도 ConvNeXt 계열 Stage-1 분류기로
기획되었다. VARCO-VISION 14B SFT 와는 별개로, 비전 특징을 저차원 종 라벨로
매핑하는 전용 분류 헤드를 학습·보정(calibrate)하는 경로였다.

이후 설계가 `02_decisions.md` §2.2 의 **2단 SFT**(1차 텍스트 CPT/SFT → 2차
멀티모달 SFT)로 수렴하면서, 별도 분류기 스테이지는 파이프라인에서 빠졌다.
그 결과 이 패키지는 한 번도 실행되지 않았다:

- `train.py` 가 만들어야 할 `outputs/clf_run1/` 이 존재하지 않는다.
- `labels.py` 가 만들어야 할 `data/shards/label_vocab.json` 이 존재하지 않는다.
- `train.py --config` 가 참조할 `configs/classifier.yaml` 자체가 없다.
- 현재 계획 인벤토리(`claudedocs/vlm_plan/04_code.md`)에 이 패키지가 등재되어 있지
  않다 — 살아있는 계획에서 이미 빠진 코드다.

외부에서 이 패키지를 임포트하는 곳도 없다(`git grep -n hanmed_mm.classifier`
전량이 `classifier/` 내부 파일 간 참조뿐). 죽은 코드가 옆에 있으면 다음 사람이
"이미 있는 분류기"로 오인해 존재하지 않는 산출물을 전제로 이어받는 함정이 된다.

`siglip_probe.py` 만은 예외다 — `outputs/siglip_probe/` 에 실제 산출물이 있고,
`classifier/` 의 다른 파일을 하나도 임포트하지 않는다(내부 의존은
`hanmed_mm.data.shard_image_reader` 뿐). 이 파일은 삭제하지 않고
`eval/siglip_probe.py` 로 이동한다.

## 결정

`classifier/` 6파일(`__init__.py`·`labels.py`·`model.py`·`dataset.py`·
`infer.py`·`calibrate.py`·`train.py`)을 삭제한다. `siglip_probe.py` 는
`git mv` 로 `eval/siglip_probe.py` 로 옮기고 내부 임포트를 갱신한다.

게이트 3단(불확실성 탐지, `02_decisions.md` §2.3)이 필요로 하는 신호는 별도
분류기의 softmax 가 아니라, **동결된 SigLIP 비전 타워의 top-k 마진**이다.
`siglip_probe.py` 가 이미 이 경로를 측정한다(`extract_features` → 선형 프로브
→ `margin = top2[:,1] - top2[:,0]`). 3단은 이 마진 분포 위에 abstain 임계값을
얹는 방향으로 설계되어 있으며(`06_training.md` §6.3), 별도 학습 분류기를
전제하지 않는다.

## 대안

**보관만 하고 계속 방치** — 임포트가 없어 당장 깨지진 않지만, "학습된 분류기가
있다"는 착각을 다음 사람에게 남긴다. `outputs/clf_run1/` 을 찾다가 없다는 걸
알게 되는 시점에야 죽은 코드였음이 드러난다.

**계획에 재편입해 실제로 학습** — 게이트 3단이 이미 frozen-SigLIP 마진 경로로
설계·검증(`siglip_probe.py` 실측)돼 있어 별도 분류기를 새로 학습할 근거가 없다.
추가 학습 비용 대비 이득이 없다.

## 결과

- `classifier/` 패키지 소멸. `siglip_probe.py` 만 `eval/` 로 이동해 생존.
- 게이트 3단은 `siglip_probe.py` 의 마진 측정 경로를 그대로 근거로 쓴다.
- 복구가 필요하면 `git show 4105ed3:korean-medicine-llm/src/hanmed_mm/classifier/<파일명>`
  으로 삭제 시점 커밋에서 파일별로 되살릴 수 있다(예:
  `git show 4105ed3:korean-medicine-llm/src/hanmed_mm/classifier/train.py`).
</content>
