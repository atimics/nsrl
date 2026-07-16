#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_dir="${NSRL_PRODUCTION_UP_FORWARD_SCALE_OUT_DIR:-data/experiments/production-model-v1/p10m-up-forward-scale-sensitivity}"
checkpoint_out="${NSRL_PRODUCTION_UP_FORWARD_SCALE_CHECKPOINT_OUT:-benchmarks/production-model-v1/p10m-up-forward-scale-sensitivity.json}"
source_model="data/experiments/production-model-v1/p10m-up-useful-update/model-3.nsrlpm"
candidate_model="data/experiments/production-model-v1/p10m-up-shift22-breakthrough/model-3.nsrlpm"
tokenizer="${NSRL_PRODUCTION_TOKENIZER:-data/processed/production-corpus-v1/tokenizer.nsrlbpe}"
dev_tokens="${NSRL_PRODUCTION_DEV_TOKENS:-data/processed/production-corpus-v1/dev.nsrltok}"
binary="target/release/nsrl-production-model"

mkdir -p "$out_dir"
cargo build --release -p nsrl-train --bin nsrl-production-model

for shift in 10 9 8 7; do
  if [[ -s "$out_dir/up-forward-shift-$shift.json" ]]; then
    echo "up forward-scale sensitivity reuse shift $shift"
    continue
  fi
  "$binary" compare-evaluate \
    --tokenizer "$tokenizer" --tokens "$dev_tokens" \
    --model "$source_model" --candidate-model "$candidate_model" \
    --trace "$out_dir/up-forward-shift-$shift.json.tmp" \
    --context-tokens 64 --max-windows 256 --up-forward-shift "$shift"
  mv "$out_dir/up-forward-shift-$shift.json.tmp" \
    "$out_dir/up-forward-shift-$shift.json"
done

node scripts/freeze-production-up-forward-scale-sensitivity-v1.mjs \
  --run-dir "$out_dir" --out "$checkpoint_out"
echo "production-model-v1 p10m up forward-scale sensitivity completed"
