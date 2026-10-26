#!/usr/bin/env bash
# Digits LR or NN privacy/utility grid with five reference shadows.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:-LR}"
if [[ "$MODEL" != LR && "$MODEL" != NN ]]; then
  echo 'Usage: scripts/figure2.sh [LR|NN]' >&2
  exit 2
fi
mkdir -p outputs cache
SHADOW_DIR="shadow_manifests/digits_${MODEL}_bs8"
"${PYTHON:-python3}" generate_shadow_csv.py \
  --dataset digits --model "$MODEL" --weight_decay 0.1 \
  --N 5 --batch_size 8 --output_dir "$SHADOW_DIR"
"${PYTHON:-python3}" risk_vs_privacy.py \
  --dataset digits --model "$MODEL" --weight_decay 0.1 --batch_size 8 \
  --T 5 --seeds 10 --rf_values 0.003 0.02 0.1 --n_shadows 5 \
  --run vru-proj:0.1 1 0.6 empirical \
  --run nft 0.005 0.8 none \
  --run neggrad+ 0.003 0.7 0.005 \
  --run scrub 0.005 0.8 0.005 \
  --cache_dir "$SHADOW_DIR" \
  --cache_file "cache/figure2_${MODEL}.pkl" \
  --save_file "outputs/figure2_${MODEL}.pdf"
