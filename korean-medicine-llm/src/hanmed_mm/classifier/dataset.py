"""분류기용 WebDataset 로더 — 768px tar 샤드 → (이미지텐서, species, part, toxic).

샤드 한 샘플 = `{key}.jpg` + `{key}.json`. json에 species_ko·part·is_poisonous 포함.
DDP에서는 nodesplitter=split_by_node 로 샤드를 rank별 분배(워커별은 자동).
"""
from __future__ import annotations
import json

from torchvision import transforms as T

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def build_transform(img_size: int, train: bool):
    if train:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.75, 1.333)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2),
            T.ToTensor(),
            T.Normalize(_MEAN, _STD),
        ])
    return T.Compose([
        T.Resize(int(img_size * 1.15)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(_MEAN, _STD),
    ])


def make_loader(urls, vocab: dict, img_size: int, batch_size: int,
                num_workers: int = 8, train: bool = True, epoch_samples: int | None = None,
                shuffle_buf: int = 2000):
    """WebDataset 로더 생성. urls: brace glob 문자열 또는 리스트."""
    import webdataset as wds

    tf = build_transform(img_size, train)
    sp_map = vocab["species"]
    pt_map = vocab["parts"]
    unk_part = pt_map["unknown"]

    def to_sample(sample):
        d = sample["json"]
        if not isinstance(d, dict):
            d = json.loads(d)
        sp = sp_map.get(d.get("species_ko"), -1)        # 미등록 종은 -1 → 아래서 필터
        pt = pt_map.get(d.get("part", "unknown"), unk_part)
        tox = 1.0 if d.get("is_poisonous") is True else 0.0
        img = tf(sample["jpg"].convert("RGB"))
        return img, sp, pt, tox

    ds = (
        wds.WebDataset(urls, shardshuffle=(100 if train else False),
                       nodesplitter=wds.split_by_node,
                       empty_check=False,            # 워커>샤드 시 일부 워커 빈 분배 허용
                       handler=wds.warn_and_continue)
        .shuffle(shuffle_buf if train else 0)
        .decode("pil", handler=wds.warn_and_continue)
        .map(to_sample, handler=wds.warn_and_continue)
        .select(lambda x: x[1] >= 0)                    # vocab 미등록 species 제외
    )
    if epoch_samples:
        ds = ds.with_epoch(epoch_samples)
    loader = wds.WebLoader(ds, batch_size=batch_size, num_workers=num_workers,
                           drop_last=train, pin_memory=True)
    return loader
