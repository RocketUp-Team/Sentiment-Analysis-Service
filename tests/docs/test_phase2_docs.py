from pathlib import Path


def test_phase2_docs_exist_and_have_required_sections():
    required = {
        "docs/superpowers/specs/dataset-verification-2026-04-17.md": [
            "Dataset",
            "License",
            "Class distribution",
        ],
        "docs/superpowers/specs/labeling-guideline.md": [
            "Schema",
            "Double-label",
            "Cohen",
        ],
        "docs/superpowers/specs/2026-04-17-absa-sarcasm-finetuning-phase2-design.md": [
            "Delta vs spec gốc",
            "Out of scope",
            "Success criteria",
        ],
    }

    for path, tokens in required.items():
        content = Path(path).read_text(encoding="utf-8")
        for token in tokens:
            assert token in content
