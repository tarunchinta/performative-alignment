#!/usr/bin/env bash
# Build the ACL submission PDF from acl-v6/main.tex
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
cp main.pdf ../self-invalidating-alignment-outline-v6.pdf
echo "Wrote ../self-invalidating-alignment-outline-v6.pdf"
