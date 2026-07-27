"""Stage 1 분류기 — 약초 식별(species) · 부위(part) · 독성(toxic) 멀티태스크.

768px WebDataset 샤드(`hanmed_mm.data.shard` 산출)를 입력으로 ConvNeXt-V2 / DINOv2
백본 + 3-헤드를 학습한다. 학습 후 temperature scaling + abstain 임계를 보정한다.
출력 계약은 `pipeline/types.ClassifierResult` 와 정렬.
"""
