#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

name="p10m-causal-tail-representation-v2-atomic-saturation-guard"
contract="benchmarks/production-model-v1/${name}-contract.json"
out_dir="data/experiments/production-model-v1/${name}"
checkpoint="benchmarks/production-model-v1/${name}.json"
binary="target/release/nsrl-production-model"
source_model="data/experiments/production-model-v1/p10m-causal-tail-representation-v2-stability-localization-r3/model-step-428.nsrlpm"
source_optimizer="data/experiments/production-model-v1/p10m-causal-tail-representation-v2-stability-localization-r3/optimizer-step-428.nsrlpo"
reference_model="data/experiments/production-model-v1/p10m-causal-tail-representation-v2-stability-localization-r4/model-step-429.nsrlpm"
reference_optimizer="data/experiments/production-model-v1/p10m-causal-tail-representation-v2-stability-localization-r4/optimizer-step-429.nsrlpo"
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
    echo "atomic saturation guard replay failed; retained work directory: $work_dir" >&2
  fi
}
trap cleanup EXIT

cargo build --release -p nsrl-train \
  --bin nsrl-production-model \
  --bin nsrl-production-residual-saturation-audit

echo "atomic saturation guard phase: replay optimizer steps 429-430"
"$binary" full-train-smoke \
  --tokenizer "$tokenizer" --tokens "$train_tokens" \
  --model "$source_model" --optimizer-state "$source_optimizer" \
  --model-out "$work_dir/candidate.nsrlpm" \
  --optimizer-state-out "$work_dir/optimizer.nsrlpo" \
  --trace "$work_dir/train.json" \
  --context-tokens 64 --targets-per-window 8 --training-workers 8 \
  --spread-windows --max-windows 2048 --evaluation-windows 64 --epochs 1 \
  --batch-windows 4 --max-optimizer-steps 4 --reject-saturated-batch \
  --matrix-learning-rate-shift 59 \
  --q-learning-rate-shift 59 --k-learning-rate-shift 22 \
  --v-learning-rate-shift 26 --o-learning-rate-shift 10 \
  --up-learning-rate-shift 59 --gate-learning-rate-shift 59 \
  --down-learning-rate-shift 59 --vector-learning-rate-shift 62 \
  --final-rms-learning-rate-shift 59 \
  --embedding-learning-rate-shift 0 --embedding-learning-rate-boost-shift 2 \
  --output-learning-rate-shift 51 --output-bias-learning-rate-shift 51 \
  --output-backward-shift 8 --probability-gradient-fractional-bits 23 \
  --probability-normalization q47-newton1

cmp -s "$work_dir/candidate.nsrlpm" "$reference_model"
cmp -s "$work_dir/optimizer.nsrlpo" "$reference_optimizer"

echo "atomic saturation guard phase: audit last committed checkpoint"
"$binary" evaluate-canonical \
  --tokenizer "$tokenizer" --tokens "$dev_tokens" \
  --model "$work_dir/candidate.nsrlpm" --trace "$work_dir/development.json" \
  --context-tokens 64 --max-windows 512
target/release/nsrl-production-residual-saturation-audit \
  --manifest "$manifest" --tokenizer "$tokenizer" \
  --model "$work_dir/candidate.nsrlpm" --trace "$work_dir/saturation.json" >/dev/null

for file in "$work_dir"/*; do
  mv "$file" "$out_dir/$(basename "$file")"
done

node scripts/freeze-production-atomic-saturation-guard-v1.mjs \
  --contract "$contract" --run-dir "$out_dir" --out "$checkpoint"
