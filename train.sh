#!/bin/bash
# ============================================================
# train.sh — Helper script để chạy training trong Docker
# ============================================================
# Cách dùng:
#   chmod +x train.sh
#   ./train.sh smoke        # Test nhanh (32 samples, 1 epoch, ~5 phút)
#   ./train.sh sentiment    # Full sentiment training
#   ./train.sh sarcasm      # Full sarcasm training
#   ./train.sh all          # Cả hai (sentiment + sarcasm)
#   ./train.sh download     # Chỉ download data, không train

set -e

MODE="${1:-smoke}"
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "======================================================"
echo "  🐳 Sentiment Analysis — Docker Training Runner"
echo "  Mode: $MODE"
echo "======================================================"

# ── Bước 1: Build training image ──────────────────────────
echo ""
echo "📦 Building training image (Dockerfile.train)..."
docker compose --profile train build trainer

# ── Bước 2: Download data nếu chưa có ────────────────────
if [ "$MODE" != "download" ]; then
    if [ ! -d "$PROJECT_ROOT/data/raw" ] || [ -z "$(ls -A $PROJECT_ROOT/data/raw 2>/dev/null)" ]; then
        echo ""
        echo "⚠️  Chưa có data trong data/raw/ — đang download..."
        echo "   (Cần kết nối internet lần đầu)"
        docker compose --profile train run --rm trainer \
            python -m src.data.downloader --task sentiment
        docker compose --profile train run --rm trainer \
            python -m src.data.downloader --task sarcasm
        echo "✅ Data downloaded."
    else
        echo ""
        echo "✅ Data đã có tại data/raw/, bỏ qua download."
    fi
fi

# ── Bước 3: Train ────────────────────────────────────────
case "$MODE" in
    "smoke")
        echo ""
        echo "🔬 Smoke test — Sentiment (32 samples, 1 epoch)..."
        docker compose --profile train run --rm trainer \
            python -m src.scripts.run_finetuning --task sentiment --smoke
        echo ""
        echo "🔬 Smoke test — Sarcasm (32 samples, 1 epoch)..."
        docker compose --profile train run --rm trainer \
            python -m src.scripts.run_finetuning --task sarcasm --smoke
        ;;

    "sentiment")
        echo ""
        echo "🚀 Full training — Sentiment (class balancing enabled)..."
        docker compose --profile train run --rm trainer \
            python -m src.scripts.run_finetuning --task sentiment
        ;;

    "sarcasm")
        echo ""
        echo "🚀 Full training — Sarcasm (class balancing enabled)..."
        docker compose --profile train run --rm trainer \
            python -m src.scripts.run_finetuning --task sarcasm
        ;;

    "all")
        echo ""
        echo "🚀 Full training — Sentiment..."
        docker compose --profile train run --rm trainer \
            python -m src.scripts.run_finetuning --task sentiment
        echo ""
        echo "🚀 Full training — Sarcasm..."
        docker compose --profile train run --rm trainer \
            python -m src.scripts.run_finetuning --task sarcasm
        ;;

    "no-balance")
        echo ""
        echo "📊 Training WITHOUT class balancing (baseline comparison)..."
        docker compose --profile train run --rm trainer \
            python -m src.scripts.run_finetuning --task sentiment --no-balance
        ;;

    "download")
        echo ""
        echo "⬇️  Downloading all datasets..."
        docker compose --profile train run --rm trainer \
            python -m src.data.downloader --task sentiment
        docker compose --profile train run --rm trainer \
            python -m src.data.downloader --task sarcasm
        echo "✅ All datasets downloaded to data/raw/"
        ;;

    *)
        echo "❌ Unknown mode: $MODE"
        echo "   Usage: ./train.sh [smoke|sentiment|sarcasm|all|no-balance|download]"
        exit 1
        ;;
esac

echo ""
echo "======================================================"
echo "  ✅ Completed: $MODE"
echo "  📁 Adapters saved to: models/adapters/"
echo "  📊 MLflow runs saved to: mlruns/"
echo "  📋 Reports saved to: reports/"
echo "======================================================"
