#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p artifacts/metrics artifacts/logs artifacts/tables

MODEL="./modelpara.pth"
TRAIN_CSV="./Species_train_annotation.csv"
TEST_CSV="./Species_test_annotation.csv"
DATASET_ROOT="${DATASET_ROOT:-}"

EXTRA_ARGS=()
if [[ -n "$DATASET_ROOT" ]]; then
  EXTRA_ARGS+=(--dataset-root "$DATASET_ROOT")
fi

# 1) Baseline
python improved_delayed_phase_encoding.py \
  --device auto \
  --model "$MODEL" \
  --train-csv "$TRAIN_CSV" \
  --test-csv "$TEST_CSV" \
  "${EXTRA_ARGS[@]}" \
  --event-threshold 0.08 \
  --save-metrics ./artifacts/metrics/exp_baseline.pkl \
  | tee ./artifacts/logs/exp_baseline.log

# 2) Event-threshold sweep (ablation)
for thr in 0.00 0.02 0.05 0.08 0.12 0.16; do
  thr_tag=$(echo "$thr" | tr -d '.')
  python improved_delayed_phase_encoding.py \
    --device auto \
    --model "$MODEL" \
    --train-csv "$TRAIN_CSV" \
    --test-csv "$TEST_CSV" \
    "${EXTRA_ARGS[@]}" \
    --event-threshold "$thr" \
    --save-metrics "./artifacts/metrics/exp_thr_${thr_tag}.pkl" \
    | tee "./artifacts/logs/exp_thr_${thr_tag}.log"
done

# 3) RF ablation
for rf in 4 8; do
  python improved_delayed_phase_encoding.py \
    --device auto \
    --model "$MODEL" \
    --train-csv "$TRAIN_CSV" \
    --test-csv "$TEST_CSV" \
    "${EXTRA_ARGS[@]}" \
    --rf-h "$rf" \
    --rf-w "$rf" \
    --event-threshold 0.08 \
    --save-metrics "./artifacts/metrics/exp_rf_${rf}x${rf}.pkl" \
    | tee "./artifacts/logs/exp_rf_${rf}x${rf}.log"
done

echo "All experiments finished."
