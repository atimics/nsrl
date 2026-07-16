#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_dir="${NSRL_PRODUCTION_PROBABILITY_NORMALIZATION_OUT_DIR:-data/experiments/production-model-v1/p10m-probability-normalization-accuracy}"
checkpoint_out="${NSRL_PRODUCTION_PROBABILITY_NORMALIZATION_CHECKPOINT_OUT:-benchmarks/production-model-v1/p10m-probability-normalization-accuracy.json}"
source_model="data/experiments/production-model-v1/p10m-up-useful-update/model-3.nsrlpm"
candidate_model="data/experiments/production-model-v1/p10m-up-shift22-breakthrough/model-3.nsrlpm"
tokenizer="${NSRL_PRODUCTION_TOKENIZER:-data/processed/production-corpus-v1/tokenizer.nsrlbpe}"
dev_tokens="${NSRL_PRODUCTION_DEV_TOKENS:-data/processed/production-corpus-v1/dev.nsrltok}"
binary="target/release/nsrl-production-model"

mkdir -p "$out_dir"
cargo build --release -p nsrl-train --bin nsrl-production-model

if [[ ! -s "$out_dir/audit.json" ]]; then
  "$binary" probability-normalization-audit \
    --tokenizer "$tokenizer" --tokens "$dev_tokens" \
    --model "$source_model" --candidate-model "$candidate_model" \
    --trace "$out_dir/audit.json.tmp" \
    --context-tokens 64 --max-windows 256 --up-forward-shift 7
  mv "$out_dir/audit.json.tmp" "$out_dir/audit.json"
fi

node scripts/freeze-production-probability-normalization-accuracy-v1.mjs \
  --run-dir "$out_dir" --out "$checkpoint_out"
echo "production-model-v1 p10m probability-normalization accuracy review completed"
