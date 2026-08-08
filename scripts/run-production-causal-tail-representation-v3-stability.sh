#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

name="p10m-causal-tail-representation-v3-stability"
contract="benchmarks/production-model-v1/${name}-contract.json"
out_dir="data/experiments/production-model-v1/${name}"
checkpoint="benchmarks/production-model-v1/${name}.json"
binary="target/release/nsrl-production-model"
source_model="data/experiments/production-model-v1/p10m-causal-tail-full-v1/candidate.nsrlpm"
tokenizer="data/processed/production-corpus-v1/tokenizer.nsrlbpe"
train_tokens="data/processed/production-corpus-v1/train.nsrltok"
dev_tokens="data/processed/production-corpus-v1/dev.nsrltok"
manifest="benchmarks/open-generation-v1/manifest.tsv"

mkdir -p "$out_dir"
work_dir="$(mktemp -d "$out_dir/.run.XXXXXX")"
cleanup() {
  local status=$?
  if ((status == 0)); then
    rm -rf "$work_dir"
  else
    echo "representation-v3 stability run failed; retained work directory: $work_dir" >&2
  fi
}
trap cleanup EXIT

cargo build --release -p nsrl-train \
  --bin nsrl-production-model \
  --bin nsrl-production-residual-saturation-audit \
  --bin nsrl-production-parameter-delta-audit

echo "representation-v3 stability phase: train 2,048 windows"
"$binary" full-train-smoke \
  --tokenizer "$tokenizer" --tokens "$train_tokens" \
  --model "$source_model" \
  --model-out "$work_dir/candidate.nsrlpm" \
  --optimizer-state-out "$work_dir/candidate.nsrlpo" \
  --trace "$work_dir/train.json" \
  --context-tokens 64 --targets-per-window 8 --training-workers 8 \
  --spread-windows --max-windows 2048 --evaluation-windows 64 --epochs 1 \
  --batch-windows 4 --max-optimizer-steps 512 --reject-saturated-batch \
  --matrix-learning-rate-shift 59 \
  --q-learning-rate-shift 59 --k-learning-rate-shift 22 \
  --v-learning-rate-shift 26 --o-learning-rate-shift 10 \
  --up-learning-rate-shift 59 --gate-learning-rate-shift 59 \
  --down-learning-rate-shift 59 --vector-learning-rate-shift 62 \
  --final-rms-learning-rate-shift 59 \
  --embedding-learning-rate-shift 0 --embedding-learning-rate-boost-shift 2 \
  --output-learning-rate-shift 51 --output-bias-learning-rate-shift 51 \
  --output-backward-shift 9 --probability-gradient-fractional-bits 23 \
  --probability-normalization q47-newton1

echo "representation-v3 stability phase: development and health audit"
"$binary" evaluate-canonical \
  --tokenizer "$tokenizer" --tokens "$dev_tokens" \
  --model "$work_dir/candidate.nsrlpm" --trace "$work_dir/development.json" \
  --context-tokens 64 --max-windows 512
target/release/nsrl-production-residual-saturation-audit \
  --manifest "$manifest" --tokenizer "$tokenizer" \
  --model "$work_dir/candidate.nsrlpm" --trace "$work_dir/saturation.json" >/dev/null
target/release/nsrl-production-parameter-delta-audit \
  --source "$source_model" --candidate "$work_dir/candidate.nsrlpm" \
  --trace "$work_dir/delta.json" >/dev/null

for file in "$work_dir"/*; do
  mv "$file" "$out_dir/$(basename "$file")"
done

node scripts/freeze-production-representation-stability-v1.mjs \
  --contract "$contract" --run-dir "$out_dir" --out "$checkpoint"
