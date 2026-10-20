#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
ITERS="${ITERS:-500000 100000 200000 300000 400000}"
DATASETS="${DATASETS:-Set5 Set14 B100 Urban100 Manga109}"
TILE="${TILE:-0}"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-./experiments}"
BENCHMARK_DIR="${BENCHMARK_DIR:-../datasets/SRBenchmarks/benchmark}"
LOG="${LOG:-results/eval_sweep.log}"

mkdir -p results
# Fail fast when there are no checkpoints or no benchmark data.
n_ckpt=$(find "$EXPERIMENTS_ROOT" -name 'iter_*.pth' 2>/dev/null | wc -l)
[ "$n_ckpt" -gt 0 ] || { echo "nessun iter_*.pth in $EXPERIMENTS_ROOT" >&2; exit 1; }
[ -d "$BENCHMARK_DIR/Set5/HR" ] || { echo "benchmark non trovati in $BENCHMARK_DIR" >&2; exit 1; }
echo "[$(date '+%F %T')] $n_ckpt checkpoint in $EXPERIMENTS_ROOT | iters: $ITERS | dataset: $DATASETS | tile $TILE" | tee -a "$LOG"

# Evaluate every checkpoint once per iteration; skip done runs and drop warning lines from the log.
for it in $ITERS; do
  echo "[$(date '+%F %T')] === iter $it ===" | tee -a "$LOG"
  PYTHONUNBUFFERED=1 "$PYTHON" scripts/eval_all.py sweep --iters "$it" --skip_done --tile "$TILE" \
      --experiments_root "$EXPERIMENTS_ROOT" --benchmark_dir "$BENCHMARK_DIR" --datasets $DATASETS \
      2>&1 | grep -v -i "warning" | tee -a "$LOG"
done

echo "[$(date '+%F %T')] === tabelle e figure ===" | tee -a "$LOG"
# Tables and figures only for the 100k and 500k iterations, independent of ITERS.
for it in 100000 500000; do
  "$PYTHON" scripts/make_tables.py --iters "$it" --datasets $DATASETS \
      --tables_dir "results/tables_${it}" --figures_dir "results/figures_${it}" 2>&1 | grep -v -i warning | tee -a "$LOG"
done
echo "[$(date '+%F %T')] fatto: results/quality.csv ($(($(wc -l < results/quality.csv) - 1)) righe), results/tables_*/, results/figures_*/" | tee -a "$LOG"
