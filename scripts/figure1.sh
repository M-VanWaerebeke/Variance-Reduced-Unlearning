#!/usr/bin/env bash
# Digits logistic-regression utility versus forget-fraction grid.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs cache
"${PYTHON:-python3}" risk_vs_rf.py \
  --dataset digits --model LR --weight_decay 0.1 --batch_size 8 \
  --T 10 --seeds 30 --min_rf 0.001 --max_rf 0.1 --n_rf 5 \
  --run vru-proj:1 1.1 0.55 after \
  --run nft:1 0.3 0.8 before \
  --run gd 2 0.8 none \
  --run svrg 1 0.4 none \
  --run sgd 0.5 0.9 none \
  --cache_file cache/figure1.pkl --save_file outputs/figure1.pdf
