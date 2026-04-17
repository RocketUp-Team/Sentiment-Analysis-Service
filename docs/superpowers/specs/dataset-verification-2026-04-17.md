# Dataset Verification - 2026-04-17

This note locks the dataset facts gathered before Phase 2 training work starts.

## Verification Method

- `cardiffnlp/tweet_eval` (`irony`): read public parquet splits from Hugging Face.
- `tyqiangz/multilingual-sentiments` (`english`): read raw CSV splits from the upstream GitHub source referenced by the dataset script.
- `uitnlp/vietnamese_students_feedback`: downloaded split files from the public Google Drive URLs referenced by the dataset script and counted labels directly.
- SemEval restaurants: checked the local project path and public task metadata.

## Dataset: `cardiffnlp/tweet_eval` / `irony`

- Purpose: English sarcasm / irony training set for the `sarcasm` adapter.
- License: `unknown` in the dataset card; use with citation and keep redistribution assumptions conservative.
- Schema:
  - `text: string`
  - `label: int` where `0=non_irony`, `1=irony`
- Split sizes:
  - train: `2862`
  - validation: `955`
  - test: `784`
- Class distribution:
  - train: `non_irony=1417`, `irony=1445`
  - validation: `non_irony=499`, `irony=456`
  - test: `non_irony=473`, `irony=311`
- Assessment: close to balanced on train/validation, but the test split is skewed toward `non_irony`.

## Dataset: `tyqiangz/multilingual-sentiments` / `english`

- Purpose: English sentence-level sentiment component for the multilingual `sentiment` adapter.
- License: `apache-2.0` per dataset card.
- Schema:
  - `text: string`
  - `label: string` in `{positive, neutral, negative}`
  - `source: string`
- Split sizes:
  - train: `1839`
  - validation: `324`
  - test: `870`
- Class distribution:
  - train: `negative=613`, `neutral=613`, `positive=613`
  - validation: `negative=108`, `neutral=108`, `positive=108`
  - test: `negative=290`, `neutral=290`, `positive=290`
- Notes:
  - Every checked split is perfectly balanced.
  - The observed `source` value is only `sem_eval_2017`, so this English slice is much smaller than the original Phase 2 assumption of roughly 30k rows.

## Dataset: `uitnlp/vietnamese_students_feedback`

- Purpose: Vietnamese sentence-level sentiment component for the multilingual `sentiment` adapter.
- License: `unknown` in the dataset card and dataset script.
- Schema:
  - `sentence: string`
  - `sentiment: int` where `0=negative`, `1=neutral`, `2=positive`
  - `topic: int` is available in the original dataset but not needed for Phase 2 sentiment training
- Split sizes:
  - train: `11426`
  - validation: `1583`
  - test: `3166`
- Class distribution:
  - train: `negative=5325`, `neutral=458`, `positive=5643`
  - validation: `negative=705`, `neutral=73`, `positive=805`
  - test: `negative=1409`, `neutral=167`, `positive=1590`
- Assessment:
  - The Vietnamese data is strongly imbalanced on `neutral`.
  - Class-weighted loss is justified for the Phase 2 sentiment adapter.

## Dataset: SemEval-2014 restaurants

- Purpose: sentence-derived English sentiment augmentation and existing ABSA zero-shot evaluation context.
- License: no explicit permissive license was found; local project README already treats this as manual acquisition under the task terms.
- Local availability:
  - `data/external/semeval2014/README.md` exists
  - required XML source files are **not present** in this worktree
- Expected source schema after extraction:
  - `sentences.csv`: `sentence_id`, `text`, `split`
  - `aspects.csv`: `sentence_id`, `aspect_category`, `sentiment`
- Public task counts referenced from task metadata:
  - train v2: approximately `3041` sentences
  - test gold: `800` sentences
- Class distribution: not computed yet because the local XML files required for sentence-level derivation are absent.
- Decision:
  - Treat SemEval-derived sentiment counts as blocked until the XML files are supplied.
  - Do not claim final merged sentiment dataset cardinality until this source is available locally.

## Impact On Phase 2

- The sarcasm dataset assumption (`~2.8k`) is confirmed.
- The English multilingual sentiment slice assumption (`~30k`) is false for the selected `english` config; the observed count is `1839/324/870`.
- UIT-VSFC is the dominant sentiment training source by volume and introduces a severe `neutral` imbalance.
- SemEval-derived enrichment remains a dependency on manual data placement, so training code should support the sentiment task without it and log whether the SemEval slice is present.
