#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

integer_dir="${NSRL_PRODUCTION_SMOKE_OUT_DIR:-data/experiments/production-model-v1/p10m-smoke}"
out_dir="${NSRL_PRODUCTION_FLOAT_OUT_DIR:-data/experiments/production-model-v1/p10m-float-smoke}"
tokens="${NSRL_PRODUCTION_TRAIN_TOKENS:-data/processed/production-corpus-v1/train.nsrltok}"

if [[ ! -f "$integer_dir/initial.nsrlpm" ]]; then
  scripts/run-production-model-v1-smoke.sh
fi

python3 scripts/production-float-twin-v1.py \
  --model "$integer_dir/initial.nsrlpm" \
  --tokens "$tokens" \
  --out "$out_dir/trained.npz" \
  --trace "$out_dir/train.json" \
  --context-tokens 4 \
  --max-windows 8 \
  --epochs 2 \
  --learning-rate-millionths 1000
node scripts/freeze-production-float-twin-v1.mjs --run-dir "$out_dir"

echo "production-model-v1 p10m float twin smoke passed"
