"""
check_class_balance.py
======================
Phân tích phân phối nhãn (class distribution) và kiểm tra mất cân bằng dữ liệu
cho tất cả các dataset được sử dụng trong Sentiment Analysis Service.

Datasets được kiểm tra:
  1. SemEval-2014 Restaurants  – aspect-level sentiment (đọc trực tiếp từ XML local nếu có)
  2. tweet_eval/irony          – sarcasm / not-sarcasm     (HuggingFace)
  3. multilingual-sentiments   – sentiment tiếng Anh       (HuggingFace)
  4. UIT-VSFC                  – sentiment tiếng Việt      (HuggingFace)

Chạy:
    python -m src.scripts.check_class_balance [--output reports/class_balance_report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

try:
    from sklearn.metrics import classification_report
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

SEPARATOR = "=" * 72


def _imbalance_ratio(counts: dict[str, int]) -> float:
    """IR = max_count / min_count  (1.0 = perfectly balanced)."""
    if not counts or min(counts.values()) == 0:
        return float("inf")
    return max(counts.values()) / min(counts.values())


def _majority_class_pct(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    return (max(counts.values()) / total * 100) if total else 0.0


def _assess_balance(ir: float) -> str:
    if ir < 1.5:
        return "✅  Balanced"
    elif ir < 3.0:
        return "⚠️  Mildly imbalanced"
    elif ir < 10.0:
        return "🔶  Moderately imbalanced"
    else:
        return "❌  Severely imbalanced"


def _print_distribution(label: str, series: pd.Series) -> dict:
    counts = series.value_counts().to_dict()
    total = sum(counts.values())
    ir = _imbalance_ratio(counts)
    majority_pct = _majority_class_pct(counts)
    status = _assess_balance(ir)

    print(f"\n{'─'*60}")
    print(f"  Dataset : {label}")
    print(f"  Total   : {total:,} samples")
    print(f"{'─'*60}")
    print(f"  {'Class':<20} {'Count':>8}  {'%':>7}")
    print(f"  {'─'*20}  {'─'*8}  {'─'*7}")
    for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        pct = cnt / total * 100
        bar = "█" * int(pct / 5)
        print(f"  {str(cls):<20} {cnt:>8,}  {pct:>6.1f}%  {bar}")
    print(f"\n  Imbalance Ratio (IR) : {ir:.2f}x  →  {status}")
    print(f"  Majority class       : {majority_pct:.1f}% of data")

    result = {
        "total": total,
        "counts": counts,
        "imbalance_ratio": round(ir, 4),
        "majority_class_pct": round(majority_pct, 2),
        "status": status,
    }

    # Nếu sklearn có sẵn, in dummy classification_report (đánh giá phân phối)
    if _HAS_SKLEARN and total > 0:
        print(f"\n  📊 Classification Report (label distribution as 'true' vs majority-class baseline):")
        labels = sorted(counts.keys(), key=str)
        majority_cls = max(counts, key=counts.get)
        y_true = series.astype(str).tolist()
        y_pred = [str(majority_cls)] * len(y_true)
        try:
            report_str = classification_report(
                y_true, y_pred,
                labels=[str(l) for l in labels],
                zero_division=0,
            )
            # indent
            for line in report_str.splitlines():
                print(f"    {line}")
            report_dict = classification_report(
                y_true, y_pred,
                labels=[str(l) for l in labels],
                output_dict=True,
                zero_division=0,
            )
            result["majority_baseline_classification_report"] = report_dict
        except Exception as e:
            print(f"    (Không thể tạo classification report: {e})")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loaders
# ──────────────────────────────────────────────────────────────────────────────

def _analyze_semeval_xml(root_dir: Path) -> dict | None:
    """Đọc XML SemEval-2014 nếu file tồn tại, trả về phân tích."""
    external = root_dir / "data" / "external" / "semeval2014"
    xml_files = {
        "train": external / "Restaurants_Train_v2.xml",
        "test": external / "Restaurants_Test_Gold.xml",
    }

    available = {k: v for k, v in xml_files.items() if v.exists()}
    if not available:
        print(f"\n[SemEval-2014] ⚠️  Không tìm thấy XML files tại {external}")
        print("  → Bỏ qua. Hãy đặt file XML vào data/external/semeval2014/")
        return None

    results: dict[str, dict] = {}
    all_sentiments: list[str] = []

    for split, xml_path in available.items():
        tree = ET.parse(xml_path)
        root = tree.getroot()
        aspect_sentiments = []
        for sentence in root.findall("sentence"):
            aspect_cats = sentence.find("aspectCategories")
            if aspect_cats is None:
                continue
            for aspect in aspect_cats.findall("aspectCategory"):
                sentiment = aspect.get("polarity", "")
                if sentiment and sentiment.lower() not in {"none", "null", "na"}:
                    aspect_sentiments.append(sentiment.lower())
        series = pd.Series(aspect_sentiments)
        results[split] = _print_distribution(
            f"SemEval-2014 [{split}] (aspect-level sentiment)", series
        )
        all_sentiments.extend(aspect_sentiments)

    if all_sentiments:
        series_all = pd.Series(all_sentiments)
        results["all"] = _print_distribution(
            "SemEval-2014 [ALL splits] (aspect-level sentiment)", series_all
        )

    return results


def _analyze_hf_dataset(task: str) -> dict | None:
    """Download và phân tích dataset từ HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print(f"\n[{task}] ⚠️  Thư viện 'datasets' chưa cài. Chạy: pip install datasets")
        return None

    results: dict[str, dict] = {}

    if task == "sarcasm":
        print(f"\n{'─'*60}")
        print("  📥 Đang tải tweet_eval/irony từ HuggingFace...")
        try:
            ds = load_dataset("tweet_eval", "irony", trust_remote_code=True)
        except Exception as e:
            print(f"  ❌ Lỗi: {e}")
            return None

        label_map = {0: "not_sarcastic", 1: "sarcastic"}
        all_labels: list[str] = []
        for split_name, split_ds in ds.items():
            df = split_ds.to_pandas()
            series = df["label"].map(label_map).fillna("unknown")
            results[split_name] = _print_distribution(
                f"tweet_eval/irony [{split_name}] (sarcasm)", series
            )
            all_labels.extend(series.tolist())
        all_series = pd.Series(all_labels)
        results["all"] = _print_distribution(
            "tweet_eval/irony [ALL splits] (sarcasm)", all_series
        )

    elif task == "sentiment_en":
        print(f"\n{'─'*60}")
        print("  📥 Đang tải multilingual-sentiments (EN) từ HuggingFace...")
        EN_MAP = {0: "positive", 1: "neutral", 2: "negative"}
        _FILES = {
            "train": "hf://datasets/tyqiangz/multilingual-sentiments@refs/convert/parquet/english/train/*.parquet",
            "validation": "hf://datasets/tyqiangz/multilingual-sentiments@refs/convert/parquet/english/validation/*.parquet",
            "test": "hf://datasets/tyqiangz/multilingual-sentiments@refs/convert/parquet/english/test/*.parquet",
        }
        try:
            ds = load_dataset("parquet", data_files=_FILES, trust_remote_code=True)
        except Exception as e:
            print(f"  ❌ Lỗi: {e}")
            return None

        all_labels: list[str] = []
        for split_name, split_ds in ds.items():
            df = split_ds.to_pandas()
            series = df["label"].map(EN_MAP).fillna("unknown")
            results[split_name] = _print_distribution(
                f"multilingual-sentiments/EN [{split_name}]", series
            )
            all_labels.extend(series.tolist())
        all_series = pd.Series(all_labels)
        results["all"] = _print_distribution(
            "multilingual-sentiments/EN [ALL splits]", all_series
        )

    elif task == "sentiment_vi":
        print(f"\n{'─'*60}")
        print("  📥 Đang tải UIT-VSFC (VI) từ HuggingFace...")
        VI_MAP = {0: "negative", 1: "neutral", 2: "positive"}
        _FILES = {
            "train": "hf://datasets/uitnlp/vietnamese_students_feedback@refs/convert/parquet/default/train/*.parquet",
            "validation": "hf://datasets/uitnlp/vietnamese_students_feedback@refs/convert/parquet/default/validation/*.parquet",
            "test": "hf://datasets/uitnlp/vietnamese_students_feedback@refs/convert/parquet/default/test/*.parquet",
        }
        try:
            ds = load_dataset("parquet", data_files=_FILES, trust_remote_code=True)
        except Exception as e:
            print(f"  ❌ Lỗi: {e}")
            return None

        all_labels: list[str] = []
        for split_name, split_ds in ds.items():
            df = split_ds.to_pandas()
            series = df["sentiment"].map(VI_MAP).fillna("unknown")
            results[split_name] = _print_distribution(
                f"UIT-VSFC/VI [{split_name}]", series
            )
            all_labels.extend(series.tolist())
        all_series = pd.Series(all_labels)
        results["all"] = _print_distribution(
            "UIT-VSFC/VI [ALL splits]", all_series
        )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Kiểm tra class imbalance cho tất cả dataset trong pipeline."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/class_balance_report.json"),
        help="Đường dẫn lưu kết quả JSON (mặc định: reports/class_balance_report.json)",
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "semeval", "sarcasm", "sentiment_en", "sentiment_vi"],
        default="all",
        help="Dataset cần kiểm tra (mặc định: all)",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[2]
    full_report: dict[str, dict] = {}

    print(SEPARATOR)
    print("  🔍 CLASS IMBALANCE CHECK — Sentiment Analysis Service")
    print(SEPARATOR)

    # ── SemEval-2014 ──────────────────────────────────────────────
    if args.dataset in ("all", "semeval"):
        print(f"\n{SEPARATOR}")
        print("  📂 DATASET 1/4 : SemEval-2014 Restaurants (ABSA)")
        print(SEPARATOR)
        result = _analyze_semeval_xml(root)
        if result:
            full_report["semeval2014"] = result

    # ── Sarcasm ───────────────────────────────────────────────────
    if args.dataset in ("all", "sarcasm"):
        print(f"\n{SEPARATOR}")
        print("  📂 DATASET 2/4 : tweet_eval/irony (Sarcasm)")
        print(SEPARATOR)
        result = _analyze_hf_dataset("sarcasm")
        if result:
            full_report["sarcasm_tweet_eval"] = result

    # ── Sentiment EN ──────────────────────────────────────────────
    if args.dataset in ("all", "sentiment_en"):
        print(f"\n{SEPARATOR}")
        print("  📂 DATASET 3/4 : multilingual-sentiments (Tiếng Anh)")
        print(SEPARATOR)
        result = _analyze_hf_dataset("sentiment_en")
        if result:
            full_report["sentiment_en"] = result

    # ── Sentiment VI ──────────────────────────────────────────────
    if args.dataset in ("all", "sentiment_vi"):
        print(f"\n{SEPARATOR}")
        print("  📂 DATASET 4/4 : UIT-VSFC (Tiếng Việt)")
        print(SEPARATOR)
        result = _analyze_hf_dataset("sentiment_vi")
        if result:
            full_report["sentiment_vi"] = result

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{SEPARATOR}")
    print("  📋 TỔNG KẾT — Mức độ mất cân bằng")
    print(SEPARATOR)
    for dataset_key, dataset_val in full_report.items():
        # Ưu tiên lấy metrics của split "all", nếu không có thì split đầu tiên
        summary_split = dataset_val.get("all") or next(iter(dataset_val.values()), {})
        ir = summary_split.get("imbalance_ratio", "N/A")
        status = summary_split.get("status", "N/A")
        total = summary_split.get("total", 0)
        print(f"  {dataset_key:<30}  IR={ir:<7}  {status}  (n={total:,})")

    # ── Save JSON ─────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(full_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'─'*60}")
    print(f"  💾 Báo cáo JSON đã được lưu tại: {args.output}")
    print(f"{'─'*60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
