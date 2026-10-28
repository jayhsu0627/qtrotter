#!/usr/bin/env bash
# Builds writeup.pdf from writeup.md with pandoc and pdflatex.
# Figures are pinned in place ([H]): pandoc typesets tables as longtables, which misplace floating figures.
set -euo pipefail
cd "$(dirname "$0")"
pandoc writeup.md -o writeup.pdf --pdf-engine=pdflatex \
  -V geometry:margin=0.8in -V geometry:vmargin=0.7in -V fontsize=10pt \
  -V colorlinks=true -V linkcolor=blue -V urlcolor=blue \
  -V 'header-includes=\usepackage{float}\floatplacement{figure}{H}' \
  --resource-path=.:..
