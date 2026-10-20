#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
SCALES="${SCALES:-2 3 4}"
RHOS="${RHOS:-0.0 0.25 0.5 0.75 1.0}"
SEEDS="${SEEDS:-0 1 2}"
FULL_SEED="${FULL_SEED:-0}"
ABL_ITERS="${ABL_ITERS:-100000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DRY="${DRY:-0}"

# Map rho to the 3-digit tag used in config names, e.g. 0.25 -> 025.
tag() { printf "%03d" "$(python -c "print(int(round($1*100)))")"; }

# Sweep scales x rhos x seeds: skip finished runs, resume from latest.pth if present.
mkdir -p logs
for s in $SCALES; do
  for r in $RHOS; do
    name="rho_$(tag "$r")_x${s}"
    cfg="configs/${name}.yml"
    [ -f "$cfg" ] || { echo "config mancante: $cfg (python scripts/make_configs.py)" >&2; exit 1; }
    for seed in $SEEDS; do
      ckpt="experiments/${name}/seed${seed}"
      if [ -f "${ckpt}/final.pth" ]; then echo "gia' fatto: ${ckpt}"; continue; fi
      # Only FULL_SEED runs the full schedule; other seeds use the shorter ABL_ITERS budget.
      iters_flag=""
      if [ "$seed" != "$FULL_SEED" ]; then iters_flag="--total_iter ${ABL_ITERS}"; fi
      resume_flag=""
      if [ -f "${ckpt}/latest.pth" ]; then resume_flag="--resume ${ckpt}/latest.pth"; fi
      cmd="${PYTHON} train.py --config ${cfg} --seed ${seed} ${iters_flag} ${resume_flag} ${EXTRA_ARGS}"
      echo "[$(date '+%F %T')] ${name} seed=${seed}: ${cmd}"
      if [ "$DRY" != "1" ]; then
        ${cmd} 2>&1 | tee "logs/${name}_seed${seed}.log"
      fi
    done
  done
done
