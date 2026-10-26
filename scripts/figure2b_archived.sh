#!/usr/bin/env bash
# Digits NN privacy/utility configuration with 100 shadows and batch size 64.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs cache
SHADOW_DIR="shadow_manifests/digits_NN_bs64_archived"
"${PYTHON:-python3}" generate_shadow_csv.py \
  --dataset digits --model NN --weight_decay 0.1 \
  --N 100 --shadow_multiplier 5 --batch_size 64 --output_dir "$SHADOW_DIR"
"${PYTHON:-python3}" risk_vs_privacy.py \
  --dataset digits --model NN --weight_decay 0.1 --batch_size 64 \
  --T 10 --seeds 10 --min_rf 0.003 --max_rf 0.1 --n_rf 3 --n_shadows 100 \
  --run vru-proj:0.01 0.1 0.7 empirical \
  --run nft 0.002 0.7 none \
  --run neggrad+ 0.002 0.7 0.01 \
  --run scrub 0.001 0.7 0.01 \
  --cache_dir "$SHADOW_DIR" \
  --cache_file cache/figure2b_archived.pkl \
  --save_file outputs/figure2b_archived.pdf
