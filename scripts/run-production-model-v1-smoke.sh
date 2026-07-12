#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_dir="${NSRL_PRODUCTION_SMOKE_OUT_DIR:-data/experiments/production-model-v1/p10m-smoke}"
tokenizer="${NSRL_PRODUCTION_TOKENIZER:-data/processed/production-corpus-v1/tokenizer.nsrlbpe}"
tokens="${NSRL_PRODUCTION_TRAIN_TOKENS:-data/processed/production-corpus-v1/train.nsrltok}"

mkdir -p "$out_dir"
cargo build --release -p nsrl-train --bin nsrl-production-model
target/release/nsrl-production-model init \
  --profile p10m \
  --tokenizer "$tokenizer" \
  --model-out "$out_dir/initial.nsrlpm" \
  --trace "$out_dir/init.json" \
  --seed 7
target/release/nsrl-production-model smoke-train \
  --tokenizer "$tokenizer" \
  --tokens "$tokens" \
  --model "$out_dir/initial.nsrlpm" \
  --model-out "$out_dir/trained.nsrlpm" \
  --trace "$out_dir/train.json" \
  --context-tokens 4 \
  --max-windows 8 \
  --epochs 2
node scripts/freeze-production-model-v1.mjs --run-dir "$out_dir"

echo "production-model-v1 p10m smoke passed"
