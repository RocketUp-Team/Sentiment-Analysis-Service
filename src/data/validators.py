"""Data quality checks for processed sentence/aspect CSVs."""

from __future__ import annotations

import importlib
import json
import logging
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.utils import load_params
from src.training.mlflow_callback import resolve_pipeline_tracking_uri

logger = logging.getLogger("data_validator")

REQUIRED_SENTENCE_COLUMNS = ("sentence_id", "text", "sentiment", "split")
REQUIRED_ASPECT_COLUMNS = ("sentence_id", "aspect_category", "sentiment")

# Serialized JSON key for rows whose ``split`` value is null/NaN (groupby-style accounting).
MISSING_SPLIT_KEY = "<missing_split>"
NULL_LABEL_PLACEHOLDER = "<null_or_na>"

# Per-split null_ratio ignores ``split`` itself so the ``<missing_split>`` bucket is not
# trivially max-null solely because the grouping key is null.
SPLIT_NULL_RATIO_COLUMNS = ("sentence_id", "text", "sentiment")


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (bool, str)) or obj is None:
        return obj
    if isinstance(obj, (int,)):
        return int(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    if hasattr(obj, "item"):
        try:
            return _json_safe(obj.item())
        except Exception:
            return str(obj)
    return obj


def _text_length_series(sentences: pd.Series) -> pd.Series:
    return sentences.fillna("").astype(str).str.len()


def _normalize_split_value(val: Any) -> str:
    try:
        if pd.isna(val):
            return MISSING_SPLIT_KEY
    except (ValueError, TypeError):
        pass
    return str(val)


def _normalize_split_series(series: pd.Series) -> pd.Series:
    return series.map(_normalize_split_value)


def _label_distribution_key(val: Any) -> str:
    try:
        if pd.isna(val):
            return NULL_LABEL_PLACEHOLDER
    except (ValueError, TypeError):
        pass
    return str(val)


def _invalid_label_examples(values: pd.Series, mask: pd.Series, *, limit: int = 10) -> list[Any]:
    if not bool(mask.any()):
        return []
    bad_vals = values.loc[mask]
    uniq = bad_vals.dropna().unique().tolist()[:limit]
    if not uniq:
        return [NULL_LABEL_PLACEHOLDER]
    return uniq


def _split_stats(s_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    split_keys = _normalize_split_series(s_df["split"])
    out: dict[str, dict[str, Any]] = {}
    for split_key in sorted(split_keys.unique(), key=str):
        part = s_df.loc[split_keys == split_key]
        null_ratios = [float(part[c].isna().mean()) for c in SPLIT_NULL_RATIO_COLUMNS]
        null_ratio = max(null_ratios) if null_ratios else 0.0
        dist = part["sentiment"].value_counts(dropna=False)
        label_distribution = {_label_distribution_key(k): int(v) for k, v in dist.items()}
        out[str(split_key)] = {
            "samples": int(len(part)),
            "null_ratio": float(null_ratio),
            "label_distribution": label_distribution,
        }
    return out


def _collect_distribution_warnings(
    *,
    splits_stats: dict[str, dict[str, Any]],
    expected_sentiments: list[str],
    aspect_distribution: dict[str, int],
    expected_aspects: list[str],
) -> list[str]:
    warnings: list[str] = []
    for split_name, info in splits_stats.items():
        samples = int(info["samples"])
        if samples <= 0:
            continue
        dist: dict[str, int] = info["label_distribution"]
        threshold = max(5, int(math.ceil(0.05 * samples)))
        for label in expected_sentiments:
            count = int(dist.get(label, 0))
            if count == 0:
                warnings.append(
                    f"split {split_name}: missing expected sentence sentiment class '{label}'"
                )
            elif count < threshold:
                warnings.append(
                    f"split {split_name}: underrepresented sentence sentiment '{label}' "
                    f"(count={count}, samples={samples})"
                )

    total_aspects = int(sum(aspect_distribution.values()))
    if total_aspects > 0:
        asp_threshold = max(5, int(math.ceil(0.05 * total_aspects)))
        for label in expected_aspects:
            count = int(aspect_distribution.get(label, 0))
            if count == 0:
                warnings.append(
                    f"dataset: missing expected aspect_category '{label}' in aspects"
                )
            elif count < asp_threshold:
                warnings.append(
                    f"dataset: underrepresented aspect_category '{label}' "
                    f"(count={count}, aspects={total_aspects})"
                )
    return warnings


class DataQualityValidator:
    def __init__(self, validation_params: dict):
        self._params = validation_params

    def validate(self, processed_dir: str) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        proc = Path(processed_dir)
        sentences_path = proc / "sentences.csv"
        aspects_path = proc / "aspects.csv"

        empty_report = {
            "total_samples": 0,
            "splits": {},
            "aspect_distribution": {},
            "text_length_stats": {},
            "checks_passed": False,
            "errors": errors,
            "warnings": warnings,
        }

        if not sentences_path.is_file():
            errors.append(f"Missing sentences file: {sentences_path}")
        if not aspects_path.is_file():
            errors.append(f"Missing aspects file: {aspects_path}")

        if errors:
            return _json_safe(empty_report)

        s_df = pd.read_csv(sentences_path)
        a_df = pd.read_csv(aspects_path)

        missing_s = [c for c in REQUIRED_SENTENCE_COLUMNS if c not in s_df.columns]
        if missing_s:
            errors.append(
                f"sentences.csv missing required columns: {missing_s} "
                f"(required: {list(REQUIRED_SENTENCE_COLUMNS)})"
            )
        missing_a = [c for c in REQUIRED_ASPECT_COLUMNS if c not in a_df.columns]
        if missing_a:
            errors.append(
                f"aspects.csv missing required columns: {missing_a} "
                f"(required: {list(REQUIRED_ASPECT_COLUMNS)})"
            )

        if errors:
            return _json_safe(
                {
                    "total_samples": int(len(s_df)),
                    "splits": {},
                    "aspect_distribution": {},
                    "text_length_stats": {},
                    "checks_passed": False,
                    "errors": errors,
                    "warnings": warnings,
                }
            )

        min_samples = int(self._params["min_samples"])
        max_null_ratio = float(self._params["max_null_ratio"])
        expected = self._params["expected_labels"]
        sentiment_labels = set(expected["sentiment"])
        aspect_labels = set(expected["aspect"])

        total_samples = int(len(s_df))

        if total_samples < min_samples:
            errors.append(
                f"total_samples {total_samples} is below min_samples {min_samples}"
            )

        if len(a_df) == 0:
            errors.append(
                "aspects.csv contains no aspect rows; expected at least one ABSA aspect record"
            )

        splits_stats = _split_stats(s_df)
        for split_name, info in splits_stats.items():
            if info["samples"] < min_samples:
                errors.append(
                    f"split {split_name}: samples {info['samples']} is below min_samples {min_samples}"
                )
            if float(info["null_ratio"]) > max_null_ratio:
                errors.append(
                    f"split {split_name}: null_ratio {info['null_ratio']:.6f} exceeds "
                    f"max_null_ratio {max_null_ratio}"
                )

        if len(a_df) > 0:
            ratios_a = [float(a_df[c].isna().mean()) for c in REQUIRED_ASPECT_COLUMNS]
            worst_a = max(ratios_a) if ratios_a else 0.0
            if worst_a > max_null_ratio:
                errors.append(
                    f"aspects.csv null ratio {worst_a:.6f} exceeds max_null_ratio {max_null_ratio}"
                )

        invalid_sent = ~s_df["sentiment"].isin(sentiment_labels)
        if invalid_sent.any():
            examples = _invalid_label_examples(s_df["sentiment"], invalid_sent)
            errors.append(f"invalid sentence sentiment labels (examples): {examples}")

        invalid_asp_cat = ~a_df["aspect_category"].isin(aspect_labels)
        if invalid_asp_cat.any():
            examples = _invalid_label_examples(a_df["aspect_category"], invalid_asp_cat)
            errors.append(f"invalid aspect_category labels (examples): {examples}")

        invalid_asp_sent = ~a_df["sentiment"].isin(sentiment_labels)
        if invalid_asp_sent.any():
            examples = _invalid_label_examples(a_df["sentiment"], invalid_asp_sent)
            errors.append(f"invalid aspect sentiment labels (examples): {examples}")

        sent_ids = set(s_df["sentence_id"].astype(str))
        asp_ids = set(a_df["sentence_id"].astype(str))
        orphans = sorted(asp_ids - sent_ids)
        if orphans:
            preview = orphans[:5]
            errors.append(
                f"orphan aspect rows: {len(orphans)} sentence_id(s) not in sentences "
                f"(examples): {preview}"
            )

        aspect_dist_series = a_df["aspect_category"].value_counts(dropna=False)
        aspect_distribution = {
            _label_distribution_key(k): int(v) for k, v in aspect_dist_series.items()
        }

        lengths = _text_length_series(s_df["text"])
        text_length_stats = {
            "mean": float(lengths.mean()) if len(lengths) else 0.0,
            "min": int(lengths.min()) if len(lengths) else 0,
            "max": int(lengths.max()) if len(lengths) else 0,
            "median": float(lengths.median()) if len(lengths) else 0.0,
            "p95": float(lengths.quantile(0.95)) if len(lengths) else 0.0,
        }

        warnings.extend(
            _collect_distribution_warnings(
                splits_stats=splits_stats,
                expected_sentiments=list(expected["sentiment"]),
                aspect_distribution=aspect_distribution,
                expected_aspects=list(expected["aspect"]),
            )
        )

        checks_passed = len(errors) == 0

        report = {
            "total_samples": total_samples,
            "splits": splits_stats,
            "aspect_distribution": aspect_distribution,
            "text_length_stats": text_length_stats,
            "checks_passed": checks_passed,
            "errors": errors,
            "warnings": warnings,
        }
        return _json_safe(report)


def save_report(report: dict[str, Any], filepath: str) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def _find_project_root(start_path: Path) -> Path:
    resolved = start_path.resolve()
    for parent in (resolved, *resolved.parents):
        if (parent / "params.yaml").is_file() or (parent / ".git").exists():
            return parent
    return Path.cwd()


def _git_short_sha(project_root: Path) -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(project_root),
        )
    except Exception as exc:
        logger.warning("MLflow logging could not resolve git SHA: %s", exc)
        return "unknown"

    if isinstance(output, bytes):
        return output.decode().strip()
    return str(output).strip()


def log_quality_report_to_mlflow(
    report: dict[str, Any],
    report_path: str | Path,
    mlflow_config: dict[str, Any],
) -> bool:
    try:
        mlflow = importlib.import_module("mlflow")
    except ModuleNotFoundError:
        logger.warning("MLflow logging skipped because the 'mlflow' package is not installed.")
        return False

    report_file = Path(report_path).resolve()
    project_root = _find_project_root(report_file)
    git_version = _git_short_sha(project_root)
    tracking_uri = resolve_pipeline_tracking_uri(mlflow_config, fallback=None)
    experiment_name = str(mlflow_config.get("experiment_name", "data_preprocessing"))
    run_params: dict[str, Any] = {
        "dataset": str(mlflow_config.get("dataset_name", "semeval2014_restaurants")),
        "git_version": git_version,
    }
    for key in ("validation_ratio", "drop_conflict"):
        if key in mlflow_config:
            run_params[key] = mlflow_config[key]

    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"data_v{git_version}"):
            mlflow.log_artifact(str(report_file))
            mlflow.log_metric("total_samples", int(report["total_samples"]))
            mlflow.log_metric(
                "passed_quality_checks",
                1 if bool(report["checks_passed"]) else 0,
            )
            for split_name, split_data in report.get("splits", {}).items():
                mlflow.log_metric(f"{split_name}_samples", int(split_data["samples"]))
                mlflow.log_metric(
                    f"{split_name}_null_ratio",
                    float(split_data["null_ratio"]),
                )
            mlflow.log_params(run_params)
    except Exception as exc:
        logger.warning("MLflow logging failed: %s", exc)
        return False

    return True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    root = Path(__file__).resolve().parents[2]
    params = load_params(str(root / "params.yaml"))
    validation = params["validation"]
    processed = root / "data" / "processed"
    validator = DataQualityValidator(validation)
    report = validator.validate(str(processed))
    out_report = root / "data" / "reports" / "quality_report.json"
    save_report(report, str(out_report))
    log_quality_report_to_mlflow(
        report,
        out_report,
        {
            **params.get("mlflow", {}),
            "dataset_name": params["data"]["dataset_name"],
            "validation_ratio": params["data"].get("validation_ratio"),
            "drop_conflict": params["preprocessing"].get("drop_conflict_labels"),
        },
    )
    if not report["checks_passed"]:
        logger.error("Data quality validation failed: %s", report["errors"])
        sys.exit(1)
    logger.info(
        "Data quality validation passed; report saved to %s",
        out_report,
    )
