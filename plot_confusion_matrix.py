import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import ConfusionMatrixDisplay

LABELS = ["positive", "negative", "neutral"]


# ─────────────────────────────────────────────
# 1. Confusion Matrix
# ─────────────────────────────────────────────

def plot_cm(cm_data, lang, output_path):
    """Vẽ và lưu confusion matrix cho một ngôn ngữ / tổng thể."""
    cm_array = np.array(cm_data)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_array, display_labels=LABELS)
    disp.plot(ax=ax, cmap="Blues", values_format="d")

    plt.title(f"Confusion Matrix (Finetuned) — {lang}", fontsize=14, fontweight="bold", pad=12)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Đã lưu confusion matrix: {output_path}")


# ─────────────────────────────────────────────
# 2. Classification Report (precision / recall / f1 / support)
# ─────────────────────────────────────────────

def plot_classification_report(report_dict: dict, title: str, output_path: str) -> None:
    """Vẽ bảng classification report dưới dạng ảnh PNG.

    Args:
        report_dict: dict trả về từ sklearn classification_report(output_dict=True).
        title: Tiêu đề hiển thị trên ảnh.
        output_path: Đường dẫn file PNG đầu ra.
    """
    # Các hàng cần hiển thị: nhãn riêng lẻ + avg
    row_keys = LABELS + ["macro avg", "weighted avg"]
    col_keys = ["precision", "recall", "f1-score", "support"]

    # Xây dựng data matrix
    cell_data = []
    for row_key in row_keys:
        if row_key not in report_dict:
            continue
        row_stats = report_dict[row_key]
        cell_data.append([
            f"{row_stats.get('precision', 0):.4f}",
            f"{row_stats.get('recall', 0):.4f}",
            f"{row_stats.get('f1-score', 0):.4f}",
            f"{int(row_stats.get('support', 0))}",
        ])

    present_rows = [k for k in row_keys if k in report_dict]
    n_rows = len(present_rows)
    n_cols = len(col_keys)

    fig, ax = plt.subplots(figsize=(10, 0.6 * n_rows + 1.8))
    ax.axis("off")

    table = ax.table(
        cellText=cell_data,
        rowLabels=present_rows,
        colLabels=col_keys,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)

    # Tô màu header
    header_color = "#2196F3"
    for col_idx in range(n_cols):
        cell = table[0, col_idx]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold")

    # Tô màu row-label column + zebra stripes
    for row_idx, row_key in enumerate(present_rows, start=1):
        label_cell = table[row_idx, -1]  # row label cell

        # Màu nền xen kẽ
        bg = "#E3F2FD" if row_idx % 2 == 0 else "white"

        # Đường kẻ phân tách trước macro/weighted avg
        if row_key in ("macro avg", "weighted avg"):
            bg = "#FFF9C4"

        for col_idx in range(n_cols):
            table[row_idx, col_idx].set_facecolor(bg)

        # Làm đậm row label
        table[row_idx, -1].set_text_props(fontweight="bold")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Đã lưu classification report: {output_path}")


# ─────────────────────────────────────────────
# 3. Main
# ─────────────────────────────────────────────

def main():
    report_path = "fairness_report.json"
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file: {report_path}")
        return

    # ── Confusion matrices ──────────────────────────────────
    cms = report.get("confusion_matrices", {})
    if not cms:
        print("⚠️  Không tìm thấy dữ liệu confusion_matrices trong file json.")
    else:
        # Từng ngôn ngữ
        for lang, cm_data in cms.items():
            output_file = f"confusion_matrix_finetuned_{lang}.png"
            plot_cm(cm_data, lang.upper(), output_file)

        # Tổng thể (cộng gộp)
        overall_cm = np.zeros_like(np.array(list(cms.values())[0]))
        for cm_data in cms.values():
            overall_cm += np.array(cm_data)
        plot_cm(overall_cm.tolist(), "Overall", "confusion_matrix_finetuned_overall.png")

    # ── Classification report ───────────────────────────────
    clf_report = report.get("classification_report")
    if not clf_report:
        print(
            "⚠️  Không tìm thấy classification_report trong file json.\n"
            "   Hãy chạy lại: python -m src.scripts.evaluate_finetuned --task <task>\n"
            "   để tái tạo fairness_report.json với trường mới."
        )
    else:
        plot_classification_report(
            clf_report,
            title="Classification Report (Finetuned) — Overall",
            output_path="classification_report_finetuned.png",
        )


if __name__ == "__main__":
    main()
