#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

initial_dir="${NSRL_PRODUCTION_SMOKE_OUT_DIR:-data/experiments/production-model-v1/p10m-smoke}"
out_dir="${NSRL_PRODUCTION_PILOT_OUT_DIR:-data/experiments/production-model-v1/p10m-pilot}"
checkpoint_out="${NSRL_PRODUCTION_PILOT_CHECKPOINT_OUT:-benchmarks/production-model-v1/p10m-pilot.json}"
tokenizer="${NSRL_PRODUCTION_TOKENIZER:-data/processed/production-corpus-v1/tokenizer.nsrlbpe}"
train_tokens="${NSRL_PRODUCTION_TRAIN_TOKENS:-data/processed/production-corpus-v1/train.nsrltok}"
dev_tokens="${NSRL_PRODUCTION_DEV_TOKENS:-data/processed/production-corpus-v1/dev.nsrltok}"

if [[ ! -f "$initial_dir/initial.nsrlpm" ]]; then
  scripts/run-production-model-v1-smoke.sh
fi

mkdir -p "$out_dir"
cargo build --release -p nsrl-train --bin nsrl-production-model

target/release/nsrl-production-model evaluate \
  --tokenizer "$tokenizer" --tokens "$dev_tokens" \
  --model "$initial_dir/initial.nsrlpm" --trace "$out_dir/integer-dev-initial.json" \
  --context-tokens 64 --max-windows 256

integer_chunk() {
  local lane="$1" index="$2" steps="$3" model_in="$4" optimizer_in="${5:-}"
  local args=(
    full-train-smoke --tokenizer "$tokenizer" --tokens "$train_tokens"
    --model "$model_in" --model-out "$out_dir/integer-$lane-$index.nsrlpm"
    --optimizer-state-out "$out_dir/integer-$lane-$index.nsrlpo"
    --trace "$out_dir/integer-$lane-$index.json"
    --context-tokens 64 --max-windows 1024 --evaluation-windows 64
    --epochs 1 --batch-windows 4 --max-optimizer-steps "$steps"
    --matrix-learning-rate-shift 20 --vector-learning-rate-shift 14
    --embedding-learning-rate-shift 8 --output-learning-rate-shift 28
  )
  if [[ -n "$optimizer_in" ]]; then
    args+=(--optimizer-state "$optimizer_in")
  fi
  target/release/nsrl-production-model "${args[@]}"
}

run_durable_lane() {
  local model_in="$initial_dir/initial.nsrlpm" optimizer_in=""
  for index in 0 1 2 3; do
    integer_chunk durable "$index" 64 "$model_in" "$optimizer_in"
    model_in="$out_dir/integer-durable-$index.nsrlpm"
    optimizer_in="$out_dir/integer-durable-$index.nsrlpo"
  done
}

run_midpoint_lane() {
  integer_chunk midpoint 0 128 "$initial_dir/initial.nsrlpm"
  integer_chunk midpoint 1 128 \
    "$out_dir/integer-midpoint-0.nsrlpm" "$out_dir/integer-midpoint-0.nsrlpo"
}

run_float_lane() {
  local float_in="" start_window
  for index in 0 1 2 3; do
    start_window=$((index * 256))
    args=(
      --math-contract legacy-v1
      --model "$initial_dir/initial.nsrlpm" --tokens "$train_tokens"
      --out "$out_dir/float-$index.npz" --trace "$out_dir/float-$index.json"
      --context-tokens 64 --start-window "$start_window" --max-windows 256
      --train-eval-max-windows 64 --epochs 1 --batch-windows 4
      --learning-rate-millionths 1000 --allow-partial-gates
    )
    if [[ -n "$float_in" ]]; then
      args+=(--resume "$float_in")
    fi
    if [[ "$index" == 0 || "$index" == 3 ]]; then
      args+=(--eval-tokens "$dev_tokens" --eval-max-windows 256)
    fi
    python3 scripts/production-float-twin-v1.py "${args[@]}"
    float_in="$out_dir/float-$index.npz"
  done
}

if [[ "${NSRL_PRODUCTION_PILOT_PARALLEL_LANES:-0}" == "1" ]]; then
  run_durable_lane & durable_pid=$!
  run_midpoint_lane & midpoint_pid=$!
  run_float_lane & float_pid=$!
  set +e
  wait "$durable_pid"; durable_status=$?
  wait "$midpoint_pid"; midpoint_status=$?
  wait "$float_pid"; float_status=$?
  set -e
  if ((durable_status != 0 || midpoint_status != 0 || float_status != 0)); then
    echo "pilot lane failed: durable=$durable_status midpoint=$midpoint_status float=$float_status" >&2
    exit 1
  fi
else
  run_durable_lane
  run_midpoint_lane
  run_float_lane
fi

cmp "$out_dir/integer-durable-3.nsrlpm" "$out_dir/integer-midpoint-1.nsrlpm"
cmp "$out_dir/integer-durable-3.nsrlpo" "$out_dir/integer-midpoint-1.nsrlpo"

target/release/nsrl-production-model evaluate \
  --tokenizer "$tokenizer" --tokens "$dev_tokens" \
  --model "$out_dir/integer-durable-3.nsrlpm" --trace "$out_dir/integer-dev-final.json" \
  --context-tokens 64 --max-windows 256

node scripts/freeze-production-pilot-v1.mjs --run-dir "$out_dir" --out "$checkpoint_out"
echo "production-model-v1 p10m controlled pilot completed"
