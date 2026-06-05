#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEX_FILE="$SCRIPT_DIR/codebase_architecture_review.tex"
MD_FILE="$SCRIPT_DIR/codebase_architecture_review.md"
OUT_DIR="$SCRIPT_DIR"
JOB_NAME="codebase_architecture_review"
MODE="${1:-auto}"

if [[ "$MODE" == "--from-tex" ]]; then
  LATEX_BIN="pdflatex"
elif [[ "$MODE" == "--from-md" ]]; then
  PDF_FILE="$OUT_DIR/$JOB_NAME.pdf"
  echo "Using renderer: python3 render_markdown_pdf.py"
  echo "Source: $MD_FILE"
  python3 "$SCRIPT_DIR/render_markdown_pdf.py" "$MD_FILE" "$PDF_FILE" >/tmp/${JOB_NAME}.log 2>&1 || {
    cat "/tmp/${JOB_NAME}.log"
    exit 1
  }
elif command -v python3 >/dev/null 2>&1 && [[ -f "$MD_FILE" && -f "$SCRIPT_DIR/render_markdown_pdf.py" ]]; then
  PDF_FILE="$OUT_DIR/$JOB_NAME.pdf"
  echo "Using renderer: python3 render_markdown_pdf.py"
  echo "Source: $MD_FILE"
  python3 "$SCRIPT_DIR/render_markdown_pdf.py" "$MD_FILE" "$PDF_FILE" >/tmp/${JOB_NAME}.log 2>&1 || {
    cat "/tmp/${JOB_NAME}.log"
    exit 1
  }
elif command -v xelatex >/dev/null 2>&1; then
  LATEX_BIN="xelatex"
elif command -v lualatex >/dev/null 2>&1; then
  LATEX_BIN="lualatex"
elif command -v pdflatex >/dev/null 2>&1; then
  LATEX_BIN="pdflatex"
else
  echo "No supported PDF renderer found. Install google-chrome, xelatex, lualatex, or pdflatex."
  exit 1
fi

if [[ -n "${LATEX_BIN:-}" ]]; then
  echo "Using compiler: $LATEX_BIN"
  echo "Source: $TEX_FILE"

  for pass in 1 2; do
    echo "Pass $pass/2"
    "$LATEX_BIN" \
      -interaction=nonstopmode \
      -output-directory "$OUT_DIR" \
      "$TEX_FILE" >/tmp/${JOB_NAME}.log 2>&1 || {
        cat "/tmp/${JOB_NAME}.log"
        exit 1
      }
  done

  for ext in aux log out toc; do
    rm -f "$OUT_DIR/$JOB_NAME.$ext"
  done
fi

echo "Done: $OUT_DIR/$JOB_NAME.pdf"
