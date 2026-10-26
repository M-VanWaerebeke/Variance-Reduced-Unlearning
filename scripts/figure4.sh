#!/usr/bin/env bash
# Breast Cancer logistic-regression privacy/utility grid.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs cache
SHADOW_DIR="shadow_manifests/breast_cancer_LR_bs64"
"${PYTHON:-python3}" generate_shadow_csv.py \
  --dataset breast_cancer --model LR --weight_decay 0.1 \
  --N 7 --shadow_multiplier 2 --batch_size 64 --output_dir "$SHADOW_DIR"
"${PYTHON:-python3}" risk_vs_privacy.py \
  --dataset breast_cancer --model LR --weight_decay 0.1 --batch_size 64 \
  --T 5 --seeds 2 --rf_values 0.006 0.0244949 0.1 --n_shadows 7 \
  --run vru-proj:0.005 1 0.7 empirical \
  --run nft 0.001 0.7 none \
  --run neggrad+ 0.005 0.7 0.02 \
  --run scrub 0.005 0.8 0.01 \
  --cache_dir "$SHADOW_DIR" \
  --cache_file cache/figure4.pkl --save_file outputs/figure4.pdf --show_std
