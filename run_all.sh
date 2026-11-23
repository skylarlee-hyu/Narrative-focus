#!/usr/bin/env bash
# Reproduce all tables and figures. Place the raw files in data/raw/ (see data/README.md) or set FINSENT_DATA.
set -euo pipefail
cd "$(dirname "$0")/scripts"
mkdir -p ../outputs/logs
for s in 00a_kospi_benchmark 00_prepare 01_table1_descriptives 02_framing_contrasts 03_strategy_returns \
         05_topic_composition 08_heterogeneity_tests 10_ordinary_vs_hype 11_valence_argmax 12_storyline_checks \
         13_content_controls 14_results_by_valence 15_figures 99_verify; do
  echo "=== $s ==="
  python "$s.py" 2>&1 | tee "../outputs/logs/$s.log"
done
