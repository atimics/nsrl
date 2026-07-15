#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_dir="${NSRL_PRODUCTION_STABILIZATION_OUT_DIR:-data/experiments/production-model-v1/p10m-stabilization}"
checkpoint_out="${NSRL_PRODUCTION_STABILIZATION_CHECKPOINT_OUT:-benchmarks/production-model-v1/p10m-stabilization.json}"
tokenizer="${NSRL_PRODUCTION_TOKENIZER:-data/processed/production-corpus-v1/tokenizer.nsrlbpe}"
train_tokens="${NSRL_PRODUCTION_TRAIN_TOKENS:-data/processed/production-corpus-v1/train.nsrltok}"
dev_tokens="${NSRL_PRODUCTION_DEV_TOKENS:-data/processed/production-corpus-v1/dev.nsrltok}"
binary="target/release/nsrl-production-model"

mkdir -p "$out_dir"
cargo build --release -p nsrl-train --bin nsrl-production-model

"$binary" init \
  --profile p10m --tokenizer "$tokenizer" \
  --model-out "$out_dir/initial.nsrlpm" --trace "$out_dir/init.json" \
  --seed 7 --output-init-amplitude 1 --output-forward-shift 14

"$binary" evaluate \
  --tokenizer "$tokenizer" --tokens "$dev_tokens" \
  --model "$out_dir/initial.nsrlpm" --trace "$out_dir/dev-initial.json" \
  --context-tokens 64 --max-windows 256

"$binary" full-train-smoke \
  --tokenizer "$tokenizer" --tokens "$train_tokens" \
  --model "$out_dir/initial.nsrlpm" \
  --model-out "$out_dir/trained.nsrlpm" \
  --optimizer-state-out "$out_dir/optimizer.nsrlpo" \
  --trace "$out_dir/train.json" \
  --context-tokens 64 --max-windows 256 --evaluation-windows 64 \
  --epochs 1 --batch-windows 4 \
  --matrix-learning-rate-shift 23 \
  --q-learning-rate-shift 27 --k-learning-rate-shift 33 \
  --v-learning-rate-shift 31 --o-learning-rate-shift 23 \
  --up-learning-rate-shift 23 --gate-learning-rate-shift 23 \
  --down-learning-rate-shift 23 \
  --vector-learning-rate-shift 21 \
  --embedding-learning-rate-shift 15 \
  --output-learning-rate-shift 34 --output-backward-shift 8

"$binary" evaluate \
  --tokenizer "$tokenizer" --tokens "$dev_tokens" \
  --model "$out_dir/trained.nsrlpm" --trace "$out_dir/dev-final.json" \
  --context-tokens 64 --max-windows 256

node scripts/freeze-production-integer-stabilization-v1.mjs \
  --run-dir "$out_dir" --out "$checkpoint_out"
echo "production-model-v1 p10m integer stabilization preflight completed"
