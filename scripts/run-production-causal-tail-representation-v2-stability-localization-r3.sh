#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

name="p10m-causal-tail-representation-v2-stability-localization-r3"
contract="benchmarks/production-model-v1/${name}-contract.json"
out_dir="data/experiments/production-model-v1/${name}"
checkpoint="benchmarks/production-model-v1/${name}.json"
binary="target/release/nsrl-production-model"
source_model="data/experiments/production-model-v1/p10m-causal-tail-representation-v2-stability-localization-r2/model-step-416.nsrlpm"
source_optimizer="data/experiments/production-model-v1/p10m-causal-tail-representation-v2-stability-localization-r2/optimizer-step-416.nsrlpo"
baseline_model="data/experiments/production-model-v1/p10m-causal-tail-full-v1/candidate.nsrlpm"
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
    echo "four-step stability localization failed; retained work directory: $work_dir" >&2
  fi
}
trap cleanup EXIT

cargo build --release -p nsrl-train \
  --bin nsrl-production-model \
  --bin nsrl-production-residual-saturation-audit \
  --bin nsrl-production-parameter-delta-audit

train_point() {
  local total_step="$1" additional_steps="$2"
  "$binary" full-train-smoke \
    --tokenizer "$tokenizer" --tokens "$train_tokens" \
    --model "$source_model" --optimizer-state "$source_optimizer" \
    --model-out "$work_dir/model-step-${total_step}.nsrlpm" \
    --optimizer-state-out "$work_dir/optimizer-step-${total_step}.nsrlpo" \
    --trace "$work_dir/train-step-${total_step}.json" \
    --context-tokens 64 --targets-per-window 8 --training-workers 8 \
    --spread-windows --max-windows 2048 --evaluation-windows 64 --epochs 1 \
    --batch-windows 4 --max-optimizer-steps "$additional_steps" \
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
}

declare -a pids=()
for pair in "420 4" "424 8" "428 12"; do
  read -r total additional <<< "$pair"
  echo "four-step stability localization phase: train step $total"
  train_point "$total" "$additional" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

for total in 420 424 428; do
  model="$work_dir/model-step-${total}.nsrlpm"
  echo "four-step stability localization phase: audit step $total"
  "$binary" evaluate-canonical \
    --tokenizer "$tokenizer" --tokens "$dev_tokens" --model "$model" \
    --trace "$work_dir/development-step-${total}.json" \
    --context-tokens 64 --max-windows 512
  target/release/nsrl-production-residual-saturation-audit \
    --manifest "$manifest" --tokenizer "$tokenizer" --model "$model" \
    --trace "$work_dir/saturation-step-${total}.json" >/dev/null
  target/release/nsrl-production-parameter-delta-audit \
    --source "$baseline_model" --candidate "$model" \
    --trace "$work_dir/delta-step-${total}.json" >/dev/null
done

for file in "$work_dir"/*; do
  mv "$file" "$out_dir/$(basename "$file")"
done

node scripts/freeze-production-stability-localization-v1.mjs \
  --contract "$contract" --run-dir "$out_dir" --out "$checkpoint"
