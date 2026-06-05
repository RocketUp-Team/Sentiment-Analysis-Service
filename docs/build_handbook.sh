#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-both}"

build_one() {
  local tex_file="$1"
  local out_name="$2"
  local compiler="pdflatex"

  echo "Building $out_name from $(basename "$tex_file")"
  for pass in 1 2; do
    echo "Pass $pass/2"
    "$compiler" \
      -interaction=nonstopmode \
      -output-directory "$SCRIPT_DIR" \
      "$tex_file" >/tmp/"$out_name".log 2>&1 || {
      cat /tmp/"$out_name".log
      exit 1
    }
  done

  for ext in aux log out toc; do
    rm -f "$SCRIPT_DIR/$out_name.$ext"
  done
}

case "$MODE" in
  en)
    build_one "$SCRIPT_DIR/codebase_architecture_review.tex" "codebase_architecture_review"
    ;;
  vi)
    build_one "$SCRIPT_DIR/codebase_architecture_review_vi.tex" "codebase_architecture_review_vi"
    ;;
  both)
    build_one "$SCRIPT_DIR/codebase_architecture_review.tex" "codebase_architecture_review"
    build_one "$SCRIPT_DIR/codebase_architecture_review_vi.tex" "codebase_architecture_review_vi"
    ;;
  *)
    echo "Usage: $0 [en|vi|both]"
    exit 1
    ;;
esac

echo "Done."

