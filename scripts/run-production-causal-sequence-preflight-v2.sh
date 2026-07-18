#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

contract="${NSRL_CAUSAL_SEQUENCE_CONTRACT:-benchmarks/production-model-v1/p10m-causal-sequence-preflight-v2-contract.json}"
out_dir="${NSRL_CAUSAL_SEQUENCE_OUT_DIR:-data/experiments/production-model-v1/p10m-causal-sequence-preflight-v2}"
checkpoint="${NSRL_CAUSAL_SEQUENCE_CHECKPOINT:-benchmarks/production-model-v1/p10m-causal-sequence-preflight-v2.json}"
binary="target/release/nsrl-production-model"
tokenizer="data/processed/production-corpus-v1/tokenizer.nsrlbpe"
train_tokens="data/processed/production-corpus-v1/train.nsrltok"
dev_tokens="data/processed/production-corpus-v1/dev.nsrltok"
test_tokens="data/processed/production-corpus-v1/test.nsrltok"
source_model="${NSRL_CAUSAL_SEQUENCE_SOURCE_MODEL:-data/experiments/production-model-v1/p10m-kv-scaling-readiness/integer-model-7.nsrlpm}"
embedding_boost_shift="${NSRL_EMBEDDING_BOOST_SHIFT:-0}"
matrix_learning_rate_shift="${NSRL_MATRIX_LEARNING_RATE_SHIFT:-23}"
embedding_learning_rate_shift="${NSRL_EMBEDDING_LEARNING_RATE_SHIFT:-0}"
vector_learning_rate_shift="${NSRL_VECTOR_LEARNING_RATE_SHIFT:-9}"
final_rms_learning_rate_shift="${NSRL_FINAL_RMS_LEARNING_RATE_SHIFT:-9}"
q_learning_rate_shift="${NSRL_Q_LEARNING_RATE_SHIFT:-16}"
k_learning_rate_shift="${NSRL_K_LEARNING_RATE_SHIFT:-19}"
v_learning_rate_shift="${NSRL_V_LEARNING_RATE_SHIFT:-23}"
o_learning_rate_shift="${NSRL_O_LEARNING_RATE_SHIFT:-11}"
up_learning_rate_shift="${NSRL_UP_LEARNING_RATE_SHIFT:-16}"
gate_learning_rate_shift="${NSRL_GATE_LEARNING_RATE_SHIFT:-16}"
down_learning_rate_shift="${NSRL_DOWN_LEARNING_RATE_SHIFT:-4}"
output_learning_rate_shift="${NSRL_OUTPUT_LEARNING_RATE_SHIFT:-30}"
output_bias_learning_rate_shift="${NSRL_OUTPUT_BIAS_LEARNING_RATE_SHIFT:-}"
context_tokens="${NSRL_CAUSAL_SEQUENCE_CONTEXT_TOKENS:-8}"
targets_per_window="${NSRL_CAUSAL_SEQUENCE_TARGETS_PER_WINDOW:-8}"
training_workers="${NSRL_CAUSAL_SEQUENCE_TRAINING_WORKERS:-1}"
max_windows="${NSRL_CAUSAL_SEQUENCE_MAX_WINDOWS:-256}"
evaluation_windows="${NSRL_CAUSAL_SEQUENCE_TRAIN_EVALUATION_WINDOWS:-256}"
midpoint_steps="${NSRL_CAUSAL_SEQUENCE_MIDPOINT_STEPS:-32}"
optimizer_steps="${NSRL_CAUSAL_SEQUENCE_OPTIMIZER_STEPS:-64}"
parallel_replay="${NSRL_CAUSAL_SEQUENCE_PARALLEL_REPLAY:-0}"
if [[ "$parallel_replay" != "0" && "$parallel_replay" != "1" ]]; then
  echo "NSRL_CAUSAL_SEQUENCE_PARALLEL_REPLAY must be 0 or 1" >&2
  exit 2
fi

mkdir -p "$out_dir"
work_dir="$(mktemp -d "$out_dir/.run.XXXXXX")"
cleanup() {
  local status=$?
  if ((status == 0)); then
    rm -rf "$work_dir"
  else
    echo "causal sequence run failed; retained work directory: $work_dir" >&2
  fi
}
trap cleanup EXIT

cargo build --release -p nsrl-train --bin nsrl-production-model

eval_model() {
  local tokens="$1" model="$2" trace="$3"
  "$binary" evaluate-canonical \
    --tokenizer "$tokenizer" --tokens "$tokens" --model "$model" \
    --trace "$trace" --context-tokens "$context_tokens" --max-windows 512
}

train_model() {
  local model_in="$1" optimizer_in="$2" model_out="$3" optimizer_out="$4"
  local trace_out="$5" optimizer_steps="$6"
  local args=(
    full-train-smoke --tokenizer "$tokenizer" --tokens "$train_tokens"
    --model "$model_in" --model-out "$model_out"
    --optimizer-state-out "$optimizer_out" --trace "$trace_out"
    --context-tokens "$context_tokens" --targets-per-window "$targets_per_window"
    --training-workers "$training_workers" --spread-windows
    --max-windows "$max_windows" --evaluation-windows "$evaluation_windows" --epochs 1
    --batch-windows 4 --max-optimizer-steps "$optimizer_steps"
    --matrix-learning-rate-shift "$matrix_learning_rate_shift"
    --q-learning-rate-shift "$q_learning_rate_shift"
    --k-learning-rate-shift "$k_learning_rate_shift"
    --v-learning-rate-shift "$v_learning_rate_shift"
    --o-learning-rate-shift "$o_learning_rate_shift"
    --up-learning-rate-shift "$up_learning_rate_shift"
    --gate-learning-rate-shift "$gate_learning_rate_shift"
    --down-learning-rate-shift "$down_learning_rate_shift"
    --vector-learning-rate-shift "$vector_learning_rate_shift"
    --final-rms-learning-rate-shift "$final_rms_learning_rate_shift"
    --embedding-learning-rate-shift "$embedding_learning_rate_shift"
    --embedding-learning-rate-boost-shift "$embedding_boost_shift"
    --output-learning-rate-shift "$output_learning_rate_shift" --output-backward-shift 8
    --probability-gradient-fractional-bits 23
    --probability-normalization q47-newton1
  )
  if [[ -n "$optimizer_in" ]]; then
    args+=(--optimizer-state "$optimizer_in")
  fi
  if [[ -n "$output_bias_learning_rate_shift" ]]; then
    args+=(--output-bias-learning-rate-shift "$output_bias_learning_rate_shift")
  fi
  "$binary" "${args[@]}"
}

eval_model "$dev_tokens" "$source_model" "$work_dir/source-dev.json"
eval_model "$test_tokens" "$source_model" "$work_dir/source-test.json"

echo "causal sequence phase: midpoint"
train_model "$source_model" "" \
  "$work_dir/midpoint.nsrlpm" "$work_dir/midpoint.nsrlpo" \
  "$work_dir/train-midpoint.json" "$midpoint_steps"
echo "causal sequence phase: final"
if [[ "$parallel_replay" == "1" ]]; then
  train_model "$work_dir/midpoint.nsrlpm" "$work_dir/midpoint.nsrlpo" \
    "$work_dir/candidate.nsrlpm" "$work_dir/candidate.nsrlpo" \
    "$work_dir/train-final.json" "$optimizer_steps" &
  final_pid=$!
  echo "causal sequence phase: replay (parallel)"
  train_model "$work_dir/midpoint.nsrlpm" "$work_dir/midpoint.nsrlpo" \
    "$work_dir/replay.nsrlpm" "$work_dir/replay.nsrlpo" \
    "$work_dir/train-replay.json" "$optimizer_steps" &
  replay_pid=$!
  final_status=0
  replay_status=0
  wait "$final_pid" || final_status=$?
  wait "$replay_pid" || replay_status=$?
  if ((final_status != 0 || replay_status != 0)); then
    echo "parallel final/replay failed: final=$final_status replay=$replay_status" >&2
    exit 1
  fi
else
  train_model "$work_dir/midpoint.nsrlpm" "$work_dir/midpoint.nsrlpo" \
    "$work_dir/candidate.nsrlpm" "$work_dir/candidate.nsrlpo" \
    "$work_dir/train-final.json" "$optimizer_steps"
  echo "causal sequence phase: replay"
  train_model "$work_dir/midpoint.nsrlpm" "$work_dir/midpoint.nsrlpo" \
    "$work_dir/replay.nsrlpm" "$work_dir/replay.nsrlpo" \
    "$work_dir/train-replay.json" "$optimizer_steps"
fi

cmp "$work_dir/candidate.nsrlpm" "$work_dir/replay.nsrlpm"
cmp "$work_dir/candidate.nsrlpo" "$work_dir/replay.nsrlpo"
cmp "$work_dir/train-final.json" "$work_dir/train-replay.json"

echo "causal sequence phase: candidate development evaluation"
eval_model "$dev_tokens" "$work_dir/candidate.nsrlpm" "$work_dir/candidate-dev.json"
source_dev="$(jq -r '.evaluation.total_nll_millibits' "$work_dir/source-dev.json")"
candidate_dev="$(jq -r '.evaluation.total_nll_millibits' "$work_dir/candidate-dev.json")"
if ((candidate_dev < source_dev)); then
  echo "causal sequence phase: candidate test confirmation"
  eval_model "$test_tokens" "$work_dir/candidate.nsrlpm" "$work_dir/candidate-test.json"
fi

rm -f "$out_dir/candidate-test.json"
for file in "$work_dir"/*; do
  mv "$file" "$out_dir/$(basename "$file")"
done

node scripts/freeze-production-causal-sequence-preflight-v1.mjs \
  --contract "$contract" --run-dir "$out_dir" --binary "$binary" --out "$checkpoint"

echo "causal sequence preflight completed: $checkpoint"
