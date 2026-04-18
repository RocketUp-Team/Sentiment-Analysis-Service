#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# export_pdf.sh  —  Compile final_project_report.tex → PDF
# Usage:  ./docs/export_pdf.sh
# ─────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEX_FILE="$SCRIPT_DIR/final_project_report.tex"
OUT_DIR="$SCRIPT_DIR"

# ── Locate xelatex ───────────────────────────────────────────
if command -v xelatex &>/dev/null; then
  XELATEX="xelatex"
elif [ -x "/Library/TeX/texbin/xelatex" ]; then
  XELATEX="/Library/TeX/texbin/xelatex"
else
  echo "❌  xelatex not found."
  echo "    Install TeX Live: https://tug.org/texlive/"
  exit 1
fi

echo "✅  Using: $XELATEX"
echo "📄  Source: $TEX_FILE"
echo "📁  Output: $OUT_DIR/final_project_report.pdf"
echo ""

# ── Compile (2 passes for TOC / cross-refs) ──────────────────
for pass in 1 2; do
  echo "🔄  Pass $pass / 2 ..."
  "$XELATEX" \
    -interaction=nonstopmode \
    -output-directory "$OUT_DIR" \
    "$TEX_FILE" \
    | grep -E "^(Output written|! |.*Error)" || true
done

# ── Clean build artifacts ─────────────────────────────────────
echo ""
echo "🧹  Cleaning build artifacts ..."
for ext in aux log out toc fls fdb_latexmk synctex.gz; do
  rm -f "$OUT_DIR/final_project_report.$ext"
done

echo ""
echo "🎉  Done!  ➜  $OUT_DIR/final_project_report.pdf"

# ── Open PDF (macOS) ─────────────────────────────────────────
if [[ "$OSTYPE" == "darwin"* ]]; then
  open "$OUT_DIR/final_project_report.pdf"
fi
