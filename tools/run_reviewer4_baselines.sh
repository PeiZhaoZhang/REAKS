#!/usr/bin/env bash
set -Eeuo pipefail

# Reviewer #4 stronger-baseline workflow. Default scope: Mip-NeRF 360 bicycle only.
# This script only writes under tools/reviewer4_baselines and reuses existing results when present.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
UTIL="$SCRIPT_DIR/reviewer4_baselines_utils.py"
OUT_DIR="$SCRIPT_DIR/reviewer4_baselines"
MAX_STEPS=${MAX_STEPS:-30000}
SEED=${SEED:-42}
CUDA_DEVICE=${CUDA_DEVICE:-0}
CONDA_ENV=${CONDA_ENV:-gsplat}
RUN_TRAIN=${RUN_TRAIN:-1}
DRY_RUN=${DRY_RUN:-0}

mkdir -p "$OUT_DIR"

echo "[reviewer4] output: $OUT_DIR"
echo "[reviewer4] scene: /root/project/data/360_v2/bicycle"
echo "[reviewer4] generating selected image lists for Pose-FPS / Coverage-Greedy / SfM-Covisibility"
conda run -n "$CONDA_ENV" python "$UTIL" select

if [[ "$RUN_TRAIN" == "1" ]]; then
  echo "[reviewer4] training missing new-baseline results with CUDA_VISIBLE_DEVICES=$CUDA_DEVICE"
  if [[ "$DRY_RUN" == "1" ]]; then
    conda run -n "$CONDA_ENV" python "$UTIL" train-missing --max-steps "$MAX_STEPS" --seed "$SEED" --cuda "$CUDA_DEVICE" --dry-run
  else
    conda run -n "$CONDA_ENV" python "$UTIL" train-missing --max-steps "$MAX_STEPS" --seed "$SEED" --cuda "$CUDA_DEVICE"
  fi
else
  echo "[reviewer4] RUN_TRAIN=0, only collecting existing results"
fi

echo "[reviewer4] collecting metrics"
conda run -n "$CONDA_ENV" python "$UTIL" collect

echo "[reviewer4] generating 2x4 visual comparison"
conda run -n "$CONDA_ENV" python "$UTIL" figure

echo "[reviewer4] done"
echo "New 3 baseline CSV: $OUT_DIR/reviewer4_new3_metrics.csv"
echo "8-method CSV: $OUT_DIR/reviewer4_8method_metrics.csv"
echo "8-method summary: $OUT_DIR/reviewer4_8method_summary.csv"
echo "LaTeX table: $OUT_DIR/reviewer4_8method_table.tex"
echo "2x4 figure PNG: $OUT_DIR/reviewer4_8method_visual_comparison.png"
echo "2x4 figure PDF: $OUT_DIR/reviewer4_8method_visual_comparison.pdf"
echo "2x4 figure SVG: $OUT_DIR/reviewer4_8method_visual_comparison.svg"
echo "Best view metrics: $OUT_DIR/reviewer4_best_view_metrics.csv"
echo "Notes: $OUT_DIR/reviewer4_new3_notes.md"
