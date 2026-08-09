#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

name="p10m-causal-tail-representation-v7-signed-block-restart-replay"
contract="benchmarks/production-model-v1/${name}-contract.json"
out_dir="data/experiments/production-model-v1/${name}"
checkpoint="benchmarks/production-model-v1/${name}.json"
binary="target/release/nsrl-production-model"
source_model="data/experiments/production-model-v1/p10m-causal-tail-full-v1/candidate.nsrlpm"
tokenizer="data/processed/production-corpus-v1/tokenizer.nsrlbpe"
train_tokens="data/processed/production-corpus-v1/train.nsrltok"
reference_dir="data/experiments/production-model-v1/p10m-causal-tail-representation-v7-signed-block"

mkdir -p "$out_dir"
work_dir="$(mktemp -d "$out_dir/.run.XXXXXX")"
cleanup() {
  local status=$?
  if ((status == 0)); then
    rm -rf "$work_dir"
  else
    echo "signed-block restart replay failed; retained work directory: $work_dir" >&2
  fi
}
trap cleanup EXIT

cargo build --release -p nsrl-train --bin nsrl-production-model

common_args=(
  full-train-smoke
  --tokenizer "$tokenizer" --tokens "$train_tokens"
  --context-tokens 64 --targets-per-window 8 --training-workers 8
  --spread-windows --max-windows 2048 --evaluation-windows 64 --epochs 1
  --batch-windows 4 --reject-saturated-batch
  --matrix-learning-rate-shift 59
  --q-learning-rate-shift 59 --k-learning-rate-shift 21
  --v-learning-rate-shift 23 --o-learning-rate-shift 9
  --up-learning-rate-shift 59 --gate-learning-rate-shift 59
  --down-learning-rate-shift 59 --vector-learning-rate-shift 62
  --final-rms-learning-rate-shift 59
  --embedding-learning-rate-shift 0 --embedding-learning-rate-boost-shift 3
  --flush-batched-embedding-residuals --descent-guard-windows 64
  --descent-guard-signed-representation-blocks
  --output-learning-rate-shift 51 --output-bias-learning-rate-shift 51
  --output-backward-shift 9 --probability-gradient-fractional-bits 23
  --probability-normalization q47-newton1
)

echo "signed-block restart replay phase: stop immediately before optimizer step 487"
"$binary" "${common_args[@]}" \
  --model "$source_model" \
  --model-out "$work_dir/partial.nsrlpm" \
  --optimizer-state-out "$work_dir/partial.nsrlpo" \
  --trace "$work_dir/partial.json" \
  --max-optimizer-steps 486

echo "signed-block restart replay phase: reload disk artifacts and cross signed step 487"
"$binary" "${common_args[@]}" \
  --model "$work_dir/partial.nsrlpm" \
  --optimizer-state "$work_dir/partial.nsrlpo" \
  --model-out "$work_dir/replay.nsrlpm" \
  --optimizer-state-out "$work_dir/replay.nsrlpo" \
  --trace "$work_dir/resume.json" \
  --max-optimizer-steps 512

cmp "$reference_dir/candidate.nsrlpm" "$work_dir/replay.nsrlpm"
cmp "$reference_dir/candidate.nsrlpo" "$work_dir/replay.nsrlpo"

for file in "$work_dir"/*; do
  mv "$file" "$out_dir/$(basename "$file")"
done

node scripts/freeze-production-representation-restart-replay-v1.mjs \
  --contract "$contract" --run-dir "$out_dir" --out "$checkpoint"
