# Image-directory prediction CLI design

## Goal

Run the trained VARCO-VISION adapter over every supported image below a directory and save one JSONL record per image. The caller explicitly chooses `herb` or `tongue`; automatic task detection is out of scope.

## Interface

```bash
PYTHONPATH=src .venv/bin/python scripts/predict_image_dir.py \
  --image-root /path/to/images \
  --task herb \
  --output outputs/herb_predictions.jsonl
```

Required arguments are `--image-root` and `--task {herb,tongue}`. `--output` defaults to `outputs/image_predictions.jsonl`; existing config and adapter defaults match `quick_eval_sft_varco.py`.

## Behavior

- Recursively discover `.jpg`, `.jpeg`, `.png`, and `.webp` files in stable path order.
- Load the processor, base model, and adapter once by reusing the existing quick-evaluation inference helpers.
- For `herb`, ask only for the pictured medicinal plant or herb name.
- For `tongue`, ask only for directly visible tongue-body, coating, and shape observations; do not request diagnosis or pattern identification.
- Append `image`, `task`, `answer_text`, and `latency_sec` as JSONL and flush after every image.
- On rerun, skip image/task pairs already recorded successfully.
- Record per-image errors and continue. Fail before model loading if the root is invalid or contains no supported images.

## Verification

One focused test covers deterministic recursive discovery, task-specific prompts, and successful-record resume selection. A CLI help invocation verifies imports and argument wiring without loading the model.

## Deliberate omissions

No automatic herb-versus-tongue classification, label scoring, batching, UI, or shared inference refactor. Add those only if directory inference proves they are needed.
