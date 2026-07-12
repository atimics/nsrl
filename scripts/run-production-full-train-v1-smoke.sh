#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

integer_dir="${NSRL_PRODUCTION_SMOKE_OUT_DIR:-data/experiments/production-model-v1/p10m-smoke}"
out_dir="${NSRL_PRODUCTION_FULL_OUT_DIR:-data/experiments/production-model-v1/p10m-full-smoke}"
tokenizer="${NSRL_PRODUCTION_TOKENIZER:-data/processed/production-corpus-v1/tokenizer.nsrlbpe}"
tokens="${NSRL_PRODUCTION_TRAIN_TOKENS:-data/processed/production-corpus-v1/train.nsrltok}"

if [[ ! -f "$integer_dir/initial.nsrlpm" ]]; then
  scripts/run-production-model-v1-smoke.sh
fi

mkdir -p "$out_dir"
cargo build --release -p nsrl-train --bin nsrl-production-model
target/release/nsrl-production-model full-train-smoke \
  --tokenizer "$tokenizer" \
  --tokens "$tokens" \
  --model "$integer_dir/initial.nsrlpm" \
  --model-out "$out_dir/trained.nsrlpm" \
  --optimizer-state-out "$out_dir/optimizer.nsrlpo" \
  --trace "$out_dir/train.json" \
  --context-tokens 4 \
  --max-windows 8 \
  --epochs 2
node scripts/freeze-production-full-train-v1.mjs --run-dir "$out_dir"

echo "production-model-v1 p10m full-backward smoke passed"
