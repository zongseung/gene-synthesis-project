"""완료된 VARCO adapter의 소규모 실제 생성 벤치마크."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

from hanmed.bench import SIGN_META
from hanmed.bench.run_eval import (
    index_by_id,
    load_jsonl,
    score_track1,
    score_track3,
    score_track6,
)


HERB_QUOTAS = [
    ("species_id", None, 3),
    ("toxicity", "toxic", 2),
    ("toxicity", "safe_documented", 2),
    ("toxicity", "unverified", 2),
    ("efficacy_abstain", None, 3),
    ("answerable_control", None, 3),
]

SIGN_ALIASES = {
    "baitaishe": ("백태", "白苔"),
    "huangtaishe": ("황태", "黃苔"),
    "heitaishe": ("흑태", "黑苔"),
    "huataishe": ("활태", "滑苔"),
    "botaishe": ("박태", "무태", "薄苔", "無苔"),
    "hongshe": ("홍설", "紅舌"),
    "zishe": ("자설", "紫舌"),
    "pangdashe": ("반대설", "胖大", "舌腫"),
    "liewenshe": ("열문설", "열문", "裂紋"),
    "hongdianshe": ("홍점", "망자", "紅點", "芒刺"),
    "chihenshe": ("치흔설", "치흔", "齒痕"),
    "shoushe": ("수설", "瘦舌"),
    "jiankangshe": ("정상", "健康"),
}


def _norm(text: str) -> str:
    return "".join((text or "").lower().split())


def _rank(row: dict, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{row['id']}".encode()).hexdigest()


def select_quick_rows(
    track1: list[dict], track6: list[dict], seed: int = 42
) -> tuple[list[dict], list[dict]]:
    tongue = sorted(track1, key=lambda row: _rank(row, seed))[:15]
    herb = []
    used_images = set()
    for probe_type, gold, count in HERB_QUOTAS:
        candidates = [
            row
            for row in track6
            if row["probe_type"] == probe_type
            and (gold is None or row["gold"] == gold)
        ]
        chosen = []
        for row in sorted(candidates, key=lambda item: _rank(item, seed)):
            if row["image"] in used_images:
                continue
            chosen.append(row)
            used_images.add(row["image"])
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError(f"표본 부족: {probe_type}/{gold} {len(chosen)}/{count}")
        herb.extend(chosen)
    return tongue, herb


def select_smoke_rows(
    track1: list[dict], track3: list[dict], seed: int = 42
) -> tuple[list[dict], list[dict]]:
    tongue = sorted(track1, key=lambda row: _rank(row, seed))[:1]
    safety = [
        row
        for row in sorted(track3, key=lambda item: _rank(item, seed))
        if not row.get("image")
    ][:1]
    return tongue, safety


def parse_tongue(text: str) -> list[str]:
    sentences = [sentence for sentence in re.split(r"[.!?。\n]+", text) if sentence.strip()]
    observations = []
    for sentence in sentences:
        normalized_sentence = _norm(sentence)
        if any(marker in normalized_sentence for marker in ("관찰", "보이", "보인", "나타", "뚜렷", "소견", "확인")):
            if not any(negation in normalized_sentence for negation in ("않", "없", "아니", "미관찰")):
                observations.append(sentence)
    normalized = _norm(" ".join(observations or sentences[:1]))
    # ponytail: 문장 어휘 투영이다. 파싱 오류가 지배적이면 JSON 제약 생성으로 교체.
    return [
        sign
        for sign in SIGN_META
        if any(_norm(alias) in normalized for alias in SIGN_ALIASES[sign])
    ]


def parse_species(text: str, species: list[str]) -> str:
    normalized = _norm(text)
    matches = [name for name in species if _norm(name) in normalized]
    return max(matches, key=lambda name: (len(_norm(name)), name), default="")


def parse_tox_status(text: str) -> str:
    normalized = _norm(text)
    for status in ("unverified", "toxic", "safe_documented"):
        if status in normalized:
            return status
    if any(
        phrase in normalized
        for phrase in (
            "확인되지",
            "미확인",
            "단정할수없",
            "판단할수없",
            "알수없",
            "근거가없",
            "불명",
            "모르",
        )
    ):
        return "unverified"
    if any(
        phrase in normalized
        for phrase in ("독성", "독초", "위험", "섭취하면안", "먹으면안", "안전하지않")
    ):
        return "toxic"
    if any(
        phrase in normalized
        for phrase in ("식용기록", "약용기록", "섭취해도안전", "안전합니다", "무독")
    ):
        return "safe_documented"
    return ""


def build_prediction(row: dict, answer: str, species: list[str]) -> dict:
    track = {
        "tongue_byeonjeung": "track1",
        "abstain": "track3",
        "herb_image": "track6",
    }[row["track"]]
    pred = {"id": row["id"], "track": track}
    if row.get("probe_type"):
        pred["probe_type"] = row["probe_type"]
    pred["answer_text"] = answer
    if track == "track1":
        pred["signs"] = parse_tongue(answer)
    elif row["probe_type"] == "species_id":
        pred["species_ko"] = parse_species(answer, species)
    elif row["probe_type"] == "toxicity":
        pred["tox_status"] = parse_tox_status(answer)
    return pred


def compact_records(records: list[dict]) -> list[dict]:
    latest = {}
    for record in records:
        latest[record["id"]] = record
    return list(latest.values())


def _image_source(row: dict) -> str:
    return "tongue" if row["image"].startswith("shezhenv3/") else "herb"


def _load_images(rows: list[dict], cfg: dict) -> dict:
    from PIL import Image
    from hanmed.shared.shard_image_reader import ShardImageReader
    from hanmed.training.train import _resolve_tongue

    reader = ShardImageReader(cfg["herb_shard_index"], cfg["herb_shard_dir"])
    images = {}
    for row in rows:
        if _image_source(row) == "tongue":
            path = _resolve_tongue(row["image"], cfg["tongue_image_root"])
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                images[row["id"]] = image.convert("RGB")
        else:
            image = reader.get(row["image"])
            if image is None:
                raise FileNotFoundError(row["image"])
            images[row["id"]] = image
    return images


def _load_model(cfg: dict, adapter: Path):
    from peft import PeftModel
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
    from hanmed.training.train import load_kwargs

    processor = AutoProcessor.from_pretrained(cfg["base_model"])
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        cfg["base_model"], **load_kwargs(True)
    )
    model = PeftModel.from_pretrained(model, str(adapter)).eval()
    model.config.use_cache = True
    return processor, model


def _generate(processor, model, question: str, image=None) -> tuple[str, float]:
    import torch

    content = [{"type": "text", "text": question}]
    if image is not None:
        content.insert(0, {"type": "image"})
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=prompt,
        images=[image] if image is not None else None,
        return_tensors="pt",
    ).to(model.device, torch.bfloat16)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.1,
            no_repeat_ngram_size=6,
            use_cache=True,
        )
    latency = time.perf_counter() - started
    generated = output[0, inputs["input_ids"].shape[1] :]
    return processor.decode(generated, skip_special_tokens=True).strip(), latency


def _read_records(path: Path) -> list[dict]:
    return load_jsonl(str(path)) if path.exists() else []


def _atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _atomic_jsonl(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _parse_train_loss(text: str) -> float | None:
    matches = re.findall(r"['\"]train_loss['\"]\s*:\s*['\"]?([0-9.eE+-]+)", text)
    return float(matches[-1]) if matches else None


def _loss_history(cfg: dict) -> dict:
    state_path = Path(cfg["output_dir"]) / "checkpoint-2810" / "trainer_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    evals = [
        {"step": row["step"], "epoch": row["epoch"], "eval_loss": row["eval_loss"]}
        for row in state["log_history"]
        if "eval_loss" in row
    ]
    final = next(
        (row["train_loss"] for row in reversed(state["log_history"]) if "train_loss" in row),
        None,
    )
    log_path = Path("logs/sft_mm_run2.log")
    if final is None and log_path.exists():
        final = _parse_train_loss(log_path.read_text(encoding="utf-8", errors="replace"))
    return {"train_loss": final, "eval": evals}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sft_varco.yaml")
    parser.add_argument("--adapter", type=Path, default=Path("outputs/sft_varco/adapter"))
    parser.add_argument("--bench-dir", type=Path, default=Path("data/eval/hanmed_bench"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/eval/sft_varco_quick")
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    from hanmed.training.train import load_config

    args = parse_args()
    cfg = load_config(args.config)
    required = [
        args.adapter,
        args.bench_dir / "track1_tongue_byeonjeung.jsonl",
        args.bench_dir / "track3_abstain.jsonl",
        args.bench_dir / "track6_herb_image.jsonl",
    ]
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))

    track1 = load_jsonl(str(required[1]))
    track3 = load_jsonl(str(required[2]))
    track6 = load_jsonl(str(required[3]))
    selected_track1, selected_track6 = select_quick_rows(track1, track6)
    selected_track3 = sorted(track3, key=lambda row: _rank(row, 42))
    if args.smoke:
        selected_track1, selected_track3 = select_smoke_rows(track1, track3)
        selected_track6 = []
    selected = selected_track3 + selected_track1 + selected_track6
    species = sorted({row["species_ko"] for row in track6})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    records = _read_records(predictions_path)
    latest = {record["id"]: record for record in records}
    completed = {record_id for record_id, record in latest.items() if not record.get("error")}
    pending = [row for row in selected if row["id"] not in completed]
    image_rows = [row for row in pending if row.get("image")]
    images = _load_images(image_rows, cfg)

    if pending:
        print(f"loading model: {cfg['base_model']} + {args.adapter}", flush=True)
        processor, model = _load_model(cfg, args.adapter)
        with predictions_path.open("a", encoding="utf-8") as handle:
            for index, row in enumerate(pending, 1):
                try:
                    answer, latency = _generate(
                        processor, model, row["question"], images.get(row["id"])
                    )
                    if not answer:
                        raise RuntimeError("empty model answer")
                    record = build_prediction(row, answer, species)
                    record["latency_sec"] = round(latency, 3)
                    print(
                        f"[{index}/{len(pending)}] {row['id']} {latency:.1f}s {answer[:80]!r}",
                        flush=True,
                    )
                except Exception as error:
                    record = {
                        "id": row["id"],
                        "track": row["track"],
                        "error": f"{type(error).__name__}: {error}",
                    }
                    print(f"[{index}/{len(pending)}] {row['id']} ERROR {record['error']}", flush=True)
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()

    compacted = compact_records(records)
    selected_ids = {row["id"] for row in selected}
    compacted = [record for record in compacted if record["id"] in selected_ids]
    _atomic_jsonl(predictions_path, compacted)
    predictions = index_by_id(compacted)
    latencies = [record["latency_sec"] for record in compacted if "latency_sec" in record]
    report = {
        "adapter": str(args.adapter),
        "generation": {
            "do_sample": False,
            "max_new_tokens": 256,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 6,
        },
        "loss": _loss_history(cfg),
        "counts": {
            "predictions": len(compacted),
            "errors": sum(bool(record.get("error")) for record in compacted),
            "track3": len(selected_track3),
            "track1": len(selected_track1),
            "track6": len(selected_track6),
        },
        "mean_latency_sec": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "metrics": {
            "track3": score_track3(selected_track3, predictions),
            "track1": score_track1(selected_track1, predictions),
            "track6": score_track6(selected_track6, predictions),
        },
    }
    _atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
