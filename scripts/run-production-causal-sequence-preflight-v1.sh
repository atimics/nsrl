#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

contract="${NSRL_CAUSAL_SEQUENCE_CONTRACT:-benchmarks/production-model-v1/p10m-causal-sequence-preflight-v1-contract.json}"
out_dir="${NSRL_CAUSAL_SEQUENCE_OUT_DIR:-data/experiments/production-model-v1/p10m-causal-sequence-preflight-v1}"
checkpoint="${NSRL_CAUSAL_SEQUENCE_CHECKPOINT:-benchmarks/production-model-v1/p10m-causal-sequence-preflight-v1.json}"
binary="target/release/nsrl-production-model"
tokenizer="data/processed/production-corpus-v1/tokenizer.nsrlbpe"
train_tokens="data/processed/production-corpus-v1/train.nsrltok"
dev_tokens="data/processed/production-corpus-v1/dev.nsrltok"
test_tokens="data/processed/production-corpus-v1/test.nsrltok"
source_model="data/experiments/production-model-v1/p10m-kv-scaling-readiness/integer-model-7.nsrlpm"

mkdir -p "$out_dir"
work_dir="$(mktemp -d "$out_dir/.run.XXXXXX")"
trap 'rm -rf "$work_dir"' EXIT

cargo build --release -p nsrl-train --bin nsrl-production-model

eval_model() {
  local tokens="$1" model="$2" trace="$3"
  "$binary" evaluate-canonical \
    --tokenizer "$tokenizer" --tokens "$tokens" --model "$model" \
    --trace "$trace" --context-tokens 8 --max-windows 512
}

train_model() {
  local model_in="$1" optimizer_in="$2" model_out="$3" optimizer_out="$4"
  local trace_out="$5" optimizer_steps="$6"
  local args=(
    full-train-smoke --tokenizer "$tokenizer" --tokens "$train_tokens"
    --model "$model_in" --model-out "$model_out"
    --optimizer-state-out "$optimizer_out" --trace "$trace_out"
    --context-tokens 8 --targets-per-window 8 --spread-windows
    --max-windows 64 --evaluation-windows 64 --epochs 1
    --batch-windows 4 --max-optimizer-steps "$optimizer_steps"
    --matrix-learning-rate-shift 23
    --q-learning-rate-shift 23 --k-learning-rate-shift 21
    --v-learning-rate-shift 24 --o-learning-rate-shift 18
    --up-learning-rate-shift 22 --gate-learning-rate-shift 23
    --down-learning-rate-shift 10 --vector-learning-rate-shift 17
    --final-rms-learning-rate-shift 15 --embedding-learning-rate-shift 5
    --output-learning-rate-shift 33 --output-backward-shift 8
    --probability-gradient-fractional-bits 23
    --probability-normalization q47-newton1
  )
  if [[ -n "$optimizer_in" ]]; then
    args+=(--optimizer-state "$optimizer_in")
  fi
  "$binary" "${args[@]}"
}

eval_model "$dev_tokens" "$source_model" "$work_dir/source-dev.json"
eval_model "$test_tokens" "$source_model" "$work_dir/source-test.json"

train_model "$source_model" "" \
  "$work_dir/midpoint.nsrlpm" "$work_dir/midpoint.nsrlpo" \
  "$work_dir/train-midpoint.json" 8
train_model "$work_dir/midpoint.nsrlpm" "$work_dir/midpoint.nsrlpo" \
  "$work_dir/candidate.nsrlpm" "$work_dir/candidate.nsrlpo" \
  "$work_dir/train-final.json" 16
train_model "$work_dir/midpoint.nsrlpm" "$work_dir/midpoint.nsrlpo" \
  "$work_dir/replay.nsrlpm" "$work_dir/replay.nsrlpo" \
  "$work_dir/train-replay.json" 16

cmp "$work_dir/candidate.nsrlpm" "$work_dir/replay.nsrlpm"
cmp "$work_dir/candidate.nsrlpo" "$work_dir/replay.nsrlpo"
cmp "$work_dir/train-final.json" "$work_dir/train-replay.json"

eval_model "$dev_tokens" "$work_dir/candidate.nsrlpm" "$work_dir/candidate-dev.json"
source_dev="$(jq -r '.evaluation.total_nll_millibits' "$work_dir/source-dev.json")"
candidate_dev="$(jq -r '.evaluation.total_nll_millibits' "$work_dir/candidate-dev.json")"
if ((candidate_dev < source_dev)); then
  eval_model "$test_tokens" "$work_dir/candidate.nsrlpm" "$work_dir/candidate-test.json"
fi

for file in "$work_dir"/*; do
  mv "$file" "$out_dir/$(basename "$file")"
done

node scripts/freeze-production-causal-sequence-preflight-v1.mjs \
  --contract "$contract" --run-dir "$out_dir" --binary "$binary" --out "$checkpoint"

echo "causal sequence preflight completed: $checkpoint"
