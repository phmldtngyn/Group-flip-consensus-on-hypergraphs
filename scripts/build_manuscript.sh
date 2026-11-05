#!/usr/bin/env bash
# Build the Supplementary Material and the main manuscript.
# SM is built first because main.tex resolves SM labels through xr-hyper.
# Usage: bash scripts/build_manuscript.sh
set -euo pipefail
cd "$(dirname "$0")/../manuscript"
python3 ../scripts/make_numbers_tex.py > /dev/null
run() { pdflatex -interaction=nonstopmode -halt-on-error "$1" > /dev/null; }
run SM.tex; run SM.tex; run SM.tex
run main.tex
bibtex main > /dev/null
run main.tex; run main.tex
for f in SM main; do
  if grep -qE "undefined|Rerun to get" "$f.log"; then
    echo "WARNING: unresolved references or rerun request in $f.log"
    grep -E "undefined|Rerun to get" "$f.log" | head
  fi
done
echo "built: manuscript/SM.pdf ($(pdfinfo SM.pdf | awk '/Pages/{print $2}') pp), manuscript/main.pdf ($(pdfinfo main.pdf | awk '/Pages/{print $2}') pp)"
