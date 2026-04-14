# Contracts Handoff

Branch: `feature/handoff-package`

Head commit: `fee1899`

## Scope

The shared handoff package is ready under `contracts/` for backend and frontend integration work:

- `contracts/__init__.py`
- `contracts/errors.py`
- `contracts/model_interface.py`
- `contracts/mock_model.py`
- `contracts/schemas.py`
- `contracts/sample_batch_input.csv`
- `contracts/sample_responses.json`
- `contracts/README.md`

## Intended Usage

Backend can import from the package surface directly:

```python
from contracts import MockModelInference, PredictRequest, PredictResponse
```

Detailed integration guidance is in `contracts/README.md`.

Concrete request/response examples are in `contracts/sample_responses.json`.

Batch upload sample input is in `contracts/sample_batch_input.csv`.

## Verification

Verified in this branch with:

```bash
/Users/trungshin/miniconda3/envs/memory_bank/bin/python -m pytest tests/contracts/ -v --cov=contracts --cov-report=term-missing
```

Result:

- `37 passed`
- `96%` total coverage for `contracts`
- clean `git status`

## Important Open Decision

Batch language handling is still unresolved.

- The current interface is `predict_batch(texts, lang="en")`.
- The batch CSV format includes a per-row `lang` column.
- Backend implementation should not assume mixed-language batch support until that contract is decided explicitly.
