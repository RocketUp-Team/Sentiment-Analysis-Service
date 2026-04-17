from pathlib import Path

from src.scripts.evaluate_finetuned import main, parse_args


def test_evaluate_finetuned_accepts_task_and_output_path(tmp_path):
    output_path = tmp_path / "metrics.json"

    args = parse_args(
        ["--task", "sentiment", "--output", str(output_path)]
    )

    assert args.task == "sentiment"
    assert args.output == output_path


def test_evaluate_finetuned_defaults_to_phase2_reports():
    args = parse_args(["--task", "sarcasm"])

    assert isinstance(args.output, Path)
    assert args.output == Path("reports/metrics_finetuned.json")


def test_evaluate_finetuned_main_writes_phase2_report_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    exit_code = main(["--task", "sentiment"])

    assert exit_code == 0
    assert Path("reports/metrics_finetuned.json").exists()
    assert Path("reports/per_language_f1.json").exists()
    assert Path("reports/fairness_report.json").exists()
