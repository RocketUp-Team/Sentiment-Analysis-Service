# Labeling Guideline For VI Sarcasm And Mixed-Language Eval Sets

This guideline defines how the team labels `data/eval/vi_sarcasm_eval.csv` and `data/eval/mixed_lang_eval.csv`.

## Goal

- Produce a small, high-agreement evaluation set for cross-lingual sarcasm and mixed-language sentiment behavior.
- Keep the schema simple enough for manual review and reproducible enough for DVC tracking.

## Scope

- `vi_sarcasm_eval.csv`: Vietnamese-only texts for sarcasm detection.
- `mixed_lang_eval.csv`: English-Vietnamese mixed or code-switched texts for sentiment and sarcasm checks.

## Schema

Each CSV row should contain:

- `id`: stable unique identifier, for example `vi_sarc_0001`
- `text`: raw user-facing text, preserved exactly
- `label`: target label
- `labeler_id`: annotator identifier, for example `trung`, `quan`, `long`
- `confidence`: integer in `1..3`
- `notes`: optional short rationale for edge cases
- `dominant_lang`: one of `en`, `vi`, `mixed` for `mixed_lang_eval.csv`

Recommended file shape for adjudicated exports:

- `vi_sarcasm_eval.csv`
  - `id,text,label,confidence,notes`
- `mixed_lang_eval.csv`
  - `id,text,label,confidence,dominant_lang,notes`

## Label Definitions

### VI sarcasm labels

- `sarcastic`: the surface wording is inconsistent with the intended meaning, often mocking, ironic, or backhanded
- `not_sarcastic`: literal or straightforward expression without clear ironic inversion

### Mixed-language labels

- `negative`
- `neutral`
- `positive`

If the text is sarcastic and also clearly sentiment-bearing, label the *intended* sentiment, not the literal surface wording.

## Decision Rules

- Prefer intended meaning over literal wording.
- Keep emoji, slang, repeated punctuation, and casing as evidence; do not normalize them during annotation.
- If context outside the text is required to decide, mark the example for removal unless the intent is still clear from the text alone.
- For ultra-short texts with no reliable signal, exclude the example instead of forcing a label.

## Double-label Process

- Every example must receive two independent labels before adjudication.
- Start with a pilot batch of `30` examples per eval set.
- After the pilot batch, compute agreement and refine instructions before full labeling.
- At least `50` examples per eval set must remain double-labeled in the final export so agreement is measurable on the final guideline version.

## Cohen's Kappa Gate

- Compute Cohen's kappa on the double-labeled subset.
- Threshold:
  - `kappa >= 0.50`: acceptable, continue
  - `0.40 <= kappa < 0.50`: refine instructions, relabel disputed categories, then recompute
  - `kappa < 0.40`: stop, revisit example selection and guideline wording before continuing

## Adjudication Workflow

- One reviewer resolves disagreements after both labels are submitted.
- The adjudicated label becomes the exported `label`.
- Keep the original disagreement notes outside the final eval CSV if needed, for example in a private working sheet.

## Example Heuristics

- `"Đỉnh cao dịch vụ luôn, đợi 45 phút mới có món"` -> `sarcastic`
- `"Shop này ok, giao hàng đúng hẹn"` -> `not_sarcastic`
- `"quá nice nhưng support như disappear luôn"` -> mixed-language sentiment is `negative`
- `"đồ ăn ổn, service bình thường"` -> mixed-language sentiment is `neutral`

## Export Rules

- Save files as UTF-8 CSV.
- No empty `label`.
- No duplicate `id`.
- Preserve original text exactly, including emoji and punctuation.
- Commit the final adjudicated CSVs through DVC rather than regular git if they belong under `data/eval/`.
