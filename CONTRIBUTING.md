# Contributing Guide

This repository is maintained as a coursework project for DDM501. The team structure below matches Appendix G of the final report and should be used to coordinate ownership, review scope, and contribution tracking.

## Before You Start

- Read the current [`README.md`](./README.md) and architecture notes in [`docs/`](./docs/).
- Check [`dvc.yaml`](./dvc.yaml) and [`params.yaml`](./params.yaml) before changing data or training logic.
- Prefer small, scoped pull requests over large mixed-purpose changes.

## Branch Naming

Use short feature branches:

- `feat/<short-description>`
- `fix/<short-description>`
- `docs/<short-description>`
- `refactor/<short-description>`
- `test/<short-description>`

Examples:

- `feat/batch-status-api`
- `fix/shap-token-length`
- `docs/readme-refresh`

## Commit Conventions

Use clear, imperative commit messages.

Recommended format:

```text
type: concise summary
```

Common types:

- `feat`
- `fix`
- `docs`
- `refactor`
- `test`
- `chore`

Examples:

- `feat: add ONNX batch benchmark`
- `fix: handle empty batch rows`
- `docs: expand contributing guide`

## Code Style

- Keep Python code typed and readable.
- Preserve existing docstrings and module structure.
- Prefer small helper functions over large inline blocks.
- Avoid changing unrelated files in the same PR.
- Do not remove user-authored changes unless explicitly requested.

## Testing Expectations

Run the relevant tests before opening a PR:

- backend changes: `pytest --cov=src tests/`
- API changes: targeted FastAPI integration tests
- data pipeline changes: DVC stages and validation checks
- model changes: model validation and smoke runs when applicable

If your change affects multiple layers, run the smallest test set that still exercises the modified path.

## Documentation Expectations

Update documentation when behavior changes:

- update `README.md` for setup, API, pipeline, or deployment changes
- update `docs/` for architecture or report-level changes
- update inline docstrings when function behavior changes

## Pull Request Checklist

- describe the purpose of the change
- list the files or subsystems touched
- mention the tests you ran
- call out any follow-up work or known limitations
- include screenshots or sample payloads when UI or API output changes

## Appendix G - Team Contributions

Table 39: Project team and responsibilities

| Role | Name | Responsibilities |
|---|---|---|
| Tech Lead | Le Quang Tuyen | Technical leadership and architecture governance; system design review; technology selection and technical strategy; integration oversight; project technical direction. |
| ML Engineer | Duong Binh An | Dataset preparation and management; training pipeline development; model evaluation and benchmarking; DVC and MLflow integration; model optimization and experimentation. |
| AI Engineer | Do Quoc Trung | Inference engine development; ONNX Runtime optimization; SHAP explainability implementation; ABSA; language detection and runtime AI services. |
| Solution Engineer | Duong Hong Quan | End-to-end solution architecture; system integration design; deployment workflow design; monitoring and observability planning; cross-component coordination. |
| Backend Engineer | Pham Duc Long | FastAPI backend development; API design and implementation; request validation and error handling; batch processing workflows; evaluation and monitoring endpoints. |
| UI/UX Engineer | Nguyen Le Hong Nhi | Angular frontend development; user interface and user experience design; frontend workflow implementation; visualization and explainability interfaces; API integration and usability improvements. |

## Report and Presentation Ownership

For report writing, demo preparation, and slide narration, use the following ownership split as the default reference:

- Dương Binh An
  - ML pipeline narrative
  - dataset preparation and model evaluation sections
  - DVC, MLflow, benchmarking, and optimization write-up
  - supporting figures and technical explanation for the model section
- Dương Hồng Quân
  - system architecture narrative
  - deployment, monitoring, and observability sections
  - integration workflow explanation
  - demo flow and cross-component coordination slides

## Suggested Contribution Split

To keep ownership clear, align changes with the named responsibilities above when possible:

- model, training, export, and evaluation work should stay under `src/model/`, `src/training/`, and `src/scripts/`
- backend and infrastructure work should stay under `src/main.py`, `infra/`, and `docker-compose.yml`
- frontend and documentation work should stay under `app/sentiment-analysis-chatbot/` and `docs/`

## Review Standard

Reviewers should look for:

- correctness regressions
- schema or API compatibility issues
- missing tests
- documentation drift
- performance regressions in inference or batch processing

## Communication

If a change affects model artifacts, pipeline outputs, or API schemas, mention it explicitly in the PR description so other contributors can update dependent code and docs.
