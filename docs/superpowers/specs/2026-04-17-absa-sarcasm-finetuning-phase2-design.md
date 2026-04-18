# ABSA And Sarcasm Fine-Tuning Phase 2 Design

## Goal

Implement Phase 2 training and inference integration for:

- one English `sarcasm` LoRA adapter trained on `cardiffnlp/tweet_eval` (`irony`)
- one multilingual `sentiment` LoRA adapter trained on English and Vietnamese sentiment data

The API must remain backward compatible by keeping `baseline` mode as the default and adding a new opt-in `finetuned` mode.

## Delta vs spec gốc

- Replace the earlier single multi-label head with **two independent LoRA adapters** on one shared `xlm-roberta-base` backbone.
- Keep ABSA **zero-shot only** in Phase 2; do not fine-tune a dedicated ABSA model yet.
- Use raw `xlm-roberta-base` instead of a pre-finetuned sentiment checkpoint so label semantics stay under project control.
- Drop Yelp silver labels from the sentiment path.
- Add auto language detection at API time, but only as an additive response feature.
- Keep adapters attached and switch them with `set_adapter()` at inference time instead of merging weights in Phase 2.

## Verified Dataset Facts

- `tweet_eval/irony`
  - train/validation/test = `2862 / 955 / 784`
  - train split is nearly balanced
- `tyqiangz/multilingual-sentiments` (`english`)
  - train/validation/test = `1839 / 324 / 870`
  - all splits are perfectly balanced
  - this is much smaller than the original rough assumption
- `uitnlp/vietnamese_students_feedback`
  - train/validation/test = `11426 / 1583 / 3166`
  - heavy `neutral` underrepresentation confirms the need for class-weighted loss
- SemEval restaurants
  - local XML sources are not present in this repo checkout
  - train/test task references are roughly `3041 / 800`
  - final sentence-derived cardinality remains blocked on manual source placement

## Architecture

### Training

- Add `src/training/task_configs.py` for explicit task metadata.
- Add `src/training/lora_config.py` for the shared LoRA settings:
  - `r=8`
  - `lora_alpha=16`
  - `lora_dropout=0.05`
  - targets = `query`, `value`
- Add `src/training/dataset_builder.py` to normalize text fields, unify labels, deduplicate across sources, and stratify by `lang x class` where applicable.
- Add `src/training/metrics.py` for macro-F1, per-language F1, and confusion-matrix payloads.
- Add `src/training/mlflow_callback.py` for per-epoch logging, tags, and artifact recording.

### Inference

- Extend `ModelConfig` with:
  - `mode`
  - base model name for finetuned inference
  - adapter paths
  - expanded `supported_languages`
- Keep the existing `BaselineModelInference` behavior for `mode="baseline"`.
- Add a finetuned branch that:
  - loads the base encoder once
  - attaches both LoRA adapters
  - switches task adapters explicitly per call
- Add `src/model/language_detector.py` for best-effort language detection with a short-text fallback to `default_lang`.

### API

- Make `lang` optional on prediction requests.
- Return additive fields:
  - `detected_lang`
  - `lang_confidence`
- Keep old fields and response semantics intact when `lang` is explicitly supplied.

## Data Rules

- Normalize whitespace only; do not lowercase or strip emoji for model input.
- Deduplicate sentiment examples by normalized text hash before stratification.
- Map all sentiment labels into `{negative, neutral, positive}`.
- Log whether the SemEval-derived slice was present in the final merged sentiment training set.

## MLflow And DVC

- MLflow must use environment-driven credentials, with `file:./mlruns` as the default fallback.
- Required tags:
  - `task`
  - `git_sha`
  - `device`
  - `environment`
  - `dataset_version`
  - `seed`
  - `user`
- DVC stages must cover:
  - raw sarcasm download
  - raw sentiment download
  - eval set preparation
  - sarcasm finetuning
  - sentiment finetuning
  - finetuned evaluation

## Backward-compat

- `ModelConfig.mode` defaults to `baseline`.
- Existing `predict_single`, `predict_batch`, `/predict`, `/health`, and explainability flows must continue to work without new required inputs.
- A regression test must prove the baseline path still returns the same output semantics as before the finetuned branch exists.

## Perf gate

- Phase 2 acceptance gate:
  - CPU `P95 <= 400ms`
  - GPU `P95 <= 100ms`
- Record the measured values in `reports/perf-baseline-phase2.json`.
- If finetuned inference misses the gate, keep the feature but document Phase 3 follow-up for adapter merge or export optimization.

## Out of scope

- ABSA fine-tuning
- merged-adapter export
- ONNX export
- few-shot Vietnamese sarcasm adaptation
- guarantees for languages beyond `en` and `vi`

## Open questions

- Which exact `transformers`, `peft`, and `mlflow` versions should be pinned for stable Colab and local parity?
- Whether the final adapter artifact remote should stay on DagsHub DVC or move to a separate bucket later.
- Whether SemEval source XML files can be checked into a private remote or must remain manual-only.

## Success criteria

- English sarcasm F1 on `tweet_eval` test split: `>= 0.70`
- English sentiment macro-F1: `>= 0.80`
- Vietnamese sentiment macro-F1: `>= 0.70`
- Mixed-language eval F1: `>= 0.65`
- Vietnamese sarcasm eval F1:
  - informational target `>= 0.45`
  - if lower, gate Vietnamese sarcasm output behind a null-safe fallback
- Baseline regression test remains green
- MLflow captures at least one successful run per task with the required tags and artifacts

## Risks And Mitigations

- Small English sentiment subset: keep the builder modular so SemEval-derived augmentation can be plugged in later.
- Unknown dataset licenses: keep raw data handling conservative and avoid broad redistribution assumptions.
- Vietnamese neutral imbalance: compute class weights from the train split and verify them in tests.
- Short-text language detection errors: fallback to `default_lang` for very short inputs and test that branch directly.
