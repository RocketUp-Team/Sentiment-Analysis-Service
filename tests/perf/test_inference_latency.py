import json
from pathlib import Path

import yaml


def test_perf_report_schema_has_p50_p95():
    payload = json.loads(Path("reports/perf-baseline-phase2.json").read_text())

    assert "cpu" in payload and "gpu" in payload
    assert "p50_ms" in payload["cpu"]
    assert "p95_ms" in payload["cpu"]
    assert "p50_ms" in payload["gpu"]
    assert "p95_ms" in payload["gpu"]


def test_dvc_phase2_stages_reference_training_params():
    dvc_config = yaml.safe_load(Path("dvc.yaml").read_text())
    params = yaml.safe_load(Path("params.yaml").read_text())

    required_stages = {
        "download_sarcasm",
        "download_sentiment",
        "prepare_eval",
        "finetune_sarcasm",
        "finetune_sentiment",
        "evaluate_finetuned",
    }

    assert required_stages.issubset(dvc_config["stages"])
    assert "training" in params
    assert "sarcasm" in params["training"]
    assert "sentiment" in params["training"]


def test_dvc_prepare_eval_uses_correct_script():
    """Regression test: prepare_eval must call prepare_eval.py, not evaluate_finetuned.py."""
    dvc_config = yaml.safe_load(Path("dvc.yaml").read_text())
    prepare_eval_cmd = dvc_config["stages"]["prepare_eval"]["cmd"]
    assert "prepare_eval" in prepare_eval_cmd, (
        f"prepare_eval stage cmd should reference prepare_eval.py, got: {prepare_eval_cmd!r}"
    )
    assert "evaluate_finetuned" not in prepare_eval_cmd, (
        "prepare_eval stage must NOT call evaluate_finetuned.py "
        "(it was incorrectly doing so before the DVC fix)"
    )


def test_dvc_download_stage_does_not_claim_raw_directory():
    """Regression test: 'download' stage must not own data/raw/ (overlapping output fix)."""
    dvc_config = yaml.safe_load(Path("dvc.yaml").read_text())
    download_outs = dvc_config["stages"]["download"].get("outs", [])
    assert "data/raw/" not in download_outs, (
        "The 'download' stage must not claim the entire data/raw/ directory as an output "
        "because download_sarcasm and download_sentiment write individual files there. "
        "Use explicit file paths instead."
    )
