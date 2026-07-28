"""동결 SigLIP linear probe — VARCO 의 비전 타워가 158종을 구분하는가.

claudedocs/vlm_plan/06_training.md §6.3 선행 측정.

두 가지를 재려고 만든다.
  1. 비전 도메인 적응이 필요한가        → 158종 정확도 (95%+ 면 불필요)
  2. 게이트 3단 abstain 임계값의 근거    → top-k 마진 분포, 특히 유사종

**유사종 정확도가 실질 지표다.** 개당귀를 참당귀로 인식하면 잘못된 종의 근거가
정상적으로 조회되어 출처까지 완벽한 위험한 답이 나온다 — 1·2단이 못 막는 유일한 경로다.

14B 전체를 올리지 않는다. 비전 타워 가중치(421키)는 첫 safetensors 샤드에만 있어
그것만 읽어 SiglipVisionModel 에 적재한다 (약 0.9GB). 학습 중인 SFT 와 GPU 를 나눠 쓴다.

사용:
  PYTHONPATH=src .venv/bin/python -m hanmed_mm.eval.siglip_probe \
      --per_species 200 --out outputs/siglip_probe
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import time

BASE = "models/VARCO-VISION-2.0-14B"
SHARD_INDEX = "data/shards/herb_shard_index.json"
SHARD_DIR = "data/shards"
SPECIES_ANN = "data/annotations/species_annotation.jsonl"


def load_vision_tower(base: str = BASE, device: str = "cuda", dtype=None):
    """비전 타워만 적재. LLM 40층은 건드리지 않는다."""
    import torch
    from safetensors.torch import load_file
    from transformers import SiglipVisionConfig, SiglipVisionModel

    cfg = json.load(open(os.path.join(base, "config.json"), encoding="utf-8"))
    model = SiglipVisionModel(SiglipVisionConfig(**cfg["vision_config"]))
    index = json.load(open(os.path.join(base, "model.safetensors.index.json"),
                          encoding="utf-8"))["weight_map"]
    shards = {v for k, v in index.items() if k.startswith("vision_tower.")}
    state = {}
    for s in sorted(shards):
        blob = load_file(os.path.join(base, s))
        state.update({k[len("vision_tower."):]: v
                      for k, v in blob.items() if k.startswith("vision_tower.")})
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"예상치 못한 키 {len(unexpected)}개 — 체크포인트 구조 확인 필요")
    # vision_use_head=false 라 pooling head 는 원래 없다. 그 외 결측은 오류다.
    hard = [k for k in missing if not k.startswith("vision_model.head.")]
    if hard:
        raise RuntimeError(f"가중치 결측 {len(hard)}개: {hard[:5]}")
    return model.to(device=device, dtype=dtype or torch.bfloat16).eval()


def species_of(logical: str) -> str:
    """논리경로 → 종명. 151 은 「001_칡」, 612 는 「가는장구채」라 번호를 떼어 통일한다.

    이걸 맞추지 않으면 유사종 판정이 151 쪽에서 전부 어긋난다 — 유사종 정확도가
    이 측정의 핵심 지표이므로 이름 표기부터 종 주석과 일치시켜야 한다.
    """
    name = logical.split("/")[1]
    head, sep, tail = name.partition("_")
    return tail if sep and head.isdigit() else name


def sample_paths(per_species: int, seed: int = 0) -> dict[str, list[str]]:
    """종당 최대 per_species 장을 논리경로로 표본."""
    index = json.load(open(SHARD_INDEX, encoding="utf-8"))
    by_species = collections.defaultdict(list)
    for logical in index:
        if len(logical.split("/")) >= 3:
            by_species[species_of(logical)].append(logical)
    rng = random.Random(seed)
    return {sp: rng.sample(paths, min(per_species, len(paths)))
            for sp, paths in sorted(by_species.items())}


def _preprocess(img, size: int = 384):
    """SigLIP 전처리 — [-1, 1] 정규화. ImageNet mean/std 가 아니다."""
    import numpy as np
    import torch
    a = np.asarray(img.resize((size, size)), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1) * 2.0 - 1.0


def extract_features(samples: dict[str, list[str]], device: str, batch_size: int,
                     workers: int = 8):
    """(특징 행렬, 라벨, 종 목록). 이미지 디코딩은 스레드로 겹친다."""
    import torch
    from concurrent.futures import ThreadPoolExecutor

    from hanmed.shared.shard_image_reader import ShardImageReader

    reader = ShardImageReader(SHARD_INDEX, SHARD_DIR)
    species = sorted(samples)
    flat = [(sp, p) for sp in species for p in samples[sp]]
    model = load_vision_tower(device=device)

    def load_one(item):
        try:
            img = reader.get(item[1])
            return (_preprocess(img), species.index(item[0])) if img is not None else None
        except Exception:
            return None

    feats, labels, t0, done = [], [], time.time(), 0
    with ThreadPoolExecutor(workers) as pool:
        for start in range(0, len(flat), batch_size):
            chunk = [r for r in pool.map(load_one, flat[start:start + batch_size]) if r]
            if not chunk:
                continue
            x = torch.stack([c[0] for c in chunk]).to(device, dtype=torch.bfloat16)
            with torch.no_grad():
                out = model(pixel_values=x).last_hidden_state.mean(1)   # 평균 풀링
            feats.append(out.float().cpu())
            labels.extend(c[1] for c in chunk)
            done += len(chunk)
            if start % (batch_size * 50) == 0:
                rate = done / max(time.time() - t0, 1e-9)
                print(f"  {done}/{len(flat)}  {rate:.0f} img/s  "
                      f"남은 {(len(flat) - done) / max(rate, 1e-9) / 60:.0f}분", flush=True)
    return torch.cat(feats).numpy(), labels, species


def similar_species() -> set[str]:
    out = set()
    with open(SPECIES_ANN, encoding="utf-8") as fh:
        for line in fh:
            a = json.loads(line)
            if a.get("has_similar_class"):
                out.add(a["species_ko"])
    return out


def main() -> int:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    p = argparse.ArgumentParser(description="동결 SigLIP linear probe.")
    p.add_argument("--per_species", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--out", default="outputs/siglip_probe")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    samples = sample_paths(a.per_species, a.seed)
    print(f"[1/3] 표본 {sum(len(v) for v in samples.values())}장 / {len(samples)}종")

    # 특징은 캐시한다 — 추출에 5분 걸리는데 뒤 단계가 실패하면 전부 다시 뽑아야 한다.
    os.makedirs(a.out, exist_ok=True)
    cache = os.path.join(a.out, f"features_{a.per_species}_{a.seed}.npz")
    if os.path.exists(cache):
        print(f"[2/3] 캐시 사용: {cache}")
        z = np.load(cache, allow_pickle=True)
        X, y, species = z["X"], z["y"], list(z["species"])
    else:
        print("[2/3] SigLIP 특징 추출…")
        X, y, species = extract_features(samples, a.device, a.batch_size)
        y = np.asarray(y)
        np.savez_compressed(cache, X=X, y=y, species=np.array(species, dtype=object))
        print(f"  캐시 저장: {cache}")

    print("[3/3] 선형 분류기 학습…")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y,
                                          random_state=a.seed)
    # sklearn 1.7+ 는 multi_class 인자를 없앴다. 다중분류는 기본이 multinomial 이다.
    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(Xtr, ytr)

    prob = clf.predict_proba(Xte)
    pred = prob.argmax(1)
    top2 = np.sort(prob, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]          # 게이트 3단 abstain 재료

    sim = similar_species()
    sim_idx = np.array([species[t] in sim for t in yte])
    acc = float((pred == yte).mean())
    acc_sim = float((pred[sim_idx] == yte[sim_idx]).mean()) if sim_idx.any() else None
    acc_non = float((pred[~sim_idx] == yte[~sim_idx]).mean()) if (~sim_idx).any() else None

    per_species_acc = {}
    for i, sp in enumerate(species):
        m = yte == i
        if m.any():
            per_species_acc[sp] = float((pred[m] == yte[m]).mean())
    worst = sorted(per_species_acc.items(), key=lambda kv: kv[1])[:15]

    report = {
        "n_species": len(species), "n_images": int(len(y)),
        "acc_overall": acc, "acc_similar": acc_sim, "acc_non_similar": acc_non,
        "n_similar_species": int(sum(1 for s in species if s in sim)),
        "margin": {q: float(np.quantile(margin, q / 100)) for q in (5, 25, 50, 75, 95)},
        "margin_correct_median": float(np.median(margin[pred == yte])),
        "margin_wrong_median": float(np.median(margin[pred != yte])),
        "worst_species": worst,
    }
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "probe_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print(f"\n158종 정확도      {acc:.4f}")
    print(f"유사종 정확도     {acc_sim}")
    print(f"비유사종 정확도   {acc_non}")
    print(f"top-2 마진 중앙   정답 {report['margin_correct_median']:.3f} / "
          f"오답 {report['margin_wrong_median']:.3f}")
    print(f"최저 종: " + ", ".join(f"{k}={v:.2f}" for k, v in worst[:5]))
    print(f"✔ {a.out}/probe_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
