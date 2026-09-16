# Quick SFT VARCO Evaluation Design

## Goal

Check the completed `outputs/sft_varco/adapter` for obvious quality or safety failures before spending hours on the full 4,848-item benchmark.

## Scope

- Run all 68 `track3_abstain` items: 44 text-only and 24 image-bearing ungrounded-sign probes.
- Run a deterministic, seed-42 image sample: 15 tongue items from track1 and 15 unique-image herb items from track6.
- The herb quota is 3 species-ID, 6 toxicity (2 each for `toxic`, `safe_documented`, and `unverified`), 3 efficacy-abstain, and 3 answerable-control items.
- Evaluate the adapter only. A base-model comparison and the full benchmark are follow-ups if this pass is healthy.

## Approach

Add one small quick-evaluation entry point under `scripts/`. It loads the local VARCO base in the same 4-bit NF4 configuration used for training, attaches the saved PEFT adapter, and keeps the model loaded for the entire run.

Generation is deterministic (`do_sample=False`) with the repetition controls already shown useful by `logs/rep_check.log`. Prompts remain in the benchmark/training style; the evaluator does not force an unfamiliar JSON response format.

Raw model answers are converted to the existing scorer schema with fixed, gold-independent rules:

- tongue signs: project `SIGN_META` aliases found in the answer;
- herb species: longest matching known `species_ko` name;
- toxicity: fixed status phrases mapped to `toxic`, `safe_documented`, or `unverified`;
- abstention: reuse `hanmed.bench.run_eval.detect_abstain`.

An ambiguous or unparsable answer stays empty and scores as incorrect. Every raw answer remains in JSONL for manual inspection.

## Data Flow

1. Read the frozen benchmark files from `data/eval/hanmed_bench`.
2. Select all track3 rows and deterministic stratified image rows.
3. Resolve track1 and image-bearing track3 tongue files from the configured tongue root, and herb files through the existing shard image reader.
4. Run one text and one image smoke item.
5. Generate the remaining answers, appending resumable JSONL records.
6. Call the existing `score_track1`, `score_track3`, and `score_track6` functions on the selected rows.
7. Write a compact JSON report with metrics, counts, latency, checkpoint, and generation settings.

## Outputs

- `outputs/eval/sft_varco_quick/predictions.jsonl`: raw answers and parsed fields.
- `outputs/eval/sft_varco_quick/report.json`: loss history plus track metrics and latency.

## Failure Handling

- Fail before generation if the adapter, benchmark, or an image cannot be loaded.
- Preserve completed JSONL rows and skip their IDs on rerun.
- Record per-item generation or parse errors without silently substituting an answer.
- Treat missing/invalid parsed fields as wrong rather than dropping them from denominators.

## Verification

- A small unit test covers deterministic selection and response parsing.
- A two-item smoke run proves text and image preprocessing, adapter loading, and generation.
- The full quick run must produce 98 unique predictions: 68 text, 15 tongue, and 15 herb.

## Report Interpretation

Report track3 appropriate-abstention, answer rate on answerable controls, over-refusal, and overall accuracy; track1 sign micro-F1 and literature substring recall; track6 per-probe accuracy and toxic recall where sampled. Image metrics are diagnostic only because each image subset has 15 items. Run the full benchmark only after this gate passes.
