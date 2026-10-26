#!/usr/bin/env bash
# Digits logistic-regression projection ablation.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs cache
"${PYTHON:-python3}" risk_vs_rf.py \
  --dataset digits --model LR --weight_decay 0.1 --batch_size 8 \
  --T 5 --seeds 30 --min_rf 0.001 --max_rf 0.1 --n_rf 5 \
  --run vru:0.1 1 0.6 empirical \
  --run vru-proj:0.1 1 0.6 empirical \
  --cache_file cache/figure3.pkl --save_file outputs/figure3.pdf
