#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_dir="${NSRL_PRODUCTION_STABILIZED_PILOT_OUT_DIR:-data/experiments/production-model-v1/p10m-stabilized-pilot}"
checkpoint_out="${NSRL_PRODUCTION_STABILIZED_PILOT_CHECKPOINT_OUT:-benchmarks/production-model-v1/p10m-stabilized-pilot.json}"
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
  --model "$out_dir/initial.nsrlpm" --trace "$out_dir/integer-dev-initial.json" \
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
    --matrix-learning-rate-shift 25
    --q-learning-rate-shift 29 --k-learning-rate-shift 35
    --v-learning-rate-shift 33 --o-learning-rate-shift 25
    --up-learning-rate-shift 25 --gate-learning-rate-shift 25
    --down-learning-rate-shift 25
    --vector-learning-rate-shift 23
    --embedding-learning-rate-shift 17
    --output-learning-rate-shift 36 --output-backward-shift 8
  )
  if [[ -n "$optimizer_in" ]]; then args+=(--optimizer-state "$optimizer_in"); fi
  "$binary" "${args[@]}"
}

run_durable_lane() {
  local model_in="$out_dir/initial.nsrlpm" optimizer_in=""
  for index in 0 1 2 3; do
    integer_chunk durable "$index" 64 "$model_in" "$optimizer_in"
    "$binary" evaluate \
      --tokenizer "$tokenizer" --tokens "$dev_tokens" \
      --model "$out_dir/integer-durable-$index.nsrlpm" \
      --trace "$out_dir/integer-dev-durable-$index.json" \
      --context-tokens 64 --max-windows 256
    node scripts/check-production-stabilized-pilot-chunk-v1.mjs \
      --chunk "$index" --train "$out_dir/integer-durable-$index.json" \
      --initial-dev "$out_dir/integer-dev-initial.json" \
      --current-dev "$out_dir/integer-dev-durable-$index.json" \
      --out "$out_dir/early-stop-durable-$index.json"
    model_in="$out_dir/integer-durable-$index.nsrlpm"
    optimizer_in="$out_dir/integer-durable-$index.nsrlpo"
  done
}

run_midpoint_lane() {
  integer_chunk midpoint 0 128 "$out_dir/initial.nsrlpm"
  integer_chunk midpoint 1 128 \
    "$out_dir/integer-midpoint-0.nsrlpm" "$out_dir/integer-midpoint-0.nsrlpo"
}

run_float_lane() {
  local float_in="" start_window
  for index in 0 1 2 3; do
    start_window=$((index * 256))
    args=(
      --model "$out_dir/initial.nsrlpm" --tokens "$train_tokens"
      --out "$out_dir/float-$index.npz" --trace "$out_dir/float-$index.json"
      --context-tokens 64 --start-window "$start_window" --max-windows 256
      --train-eval-max-windows 64 --epochs 1 --batch-windows 4
      --learning-rate-millionths 1000 --allow-partial-gates
    )
    if [[ -n "$float_in" ]]; then args+=(--resume "$float_in"); fi
    if [[ "$index" == 0 || "$index" == 3 ]]; then
      args+=(--eval-tokens "$dev_tokens" --eval-max-windows 256)
    fi
    python3 scripts/production-float-twin-v1.py "${args[@]}"
    float_in="$out_dir/float-$index.npz"
  done
}

if [[ "${NSRL_PRODUCTION_STABILIZED_PILOT_PARALLEL_LANES:-0}" == "1" ]]; then
  child_pids=()
  cleanup_children() {
    if ((${#child_pids[@]} > 0)); then
      kill "${child_pids[@]}" >/dev/null 2>&1 || true
    fi
  }
  trap cleanup_children INT TERM EXIT
  run_durable_lane & durable_pid=$!; child_pids+=("$durable_pid")
  run_midpoint_lane & midpoint_pid=$!; child_pids+=("$midpoint_pid")
  run_float_lane & float_pid=$!; child_pids+=("$float_pid")
  if ! wait "$durable_pid"; then
    cleanup_children
    wait "$midpoint_pid" >/dev/null 2>&1 || true
    wait "$float_pid" >/dev/null 2>&1 || true
    echo "stabilized pilot durable lane stopped by a failed health gate" >&2
    exit 1
  fi
  durable_status=0
  set +e
  wait "$midpoint_pid"; midpoint_status=$?
  wait "$float_pid"; float_status=$?
  set -e
  child_pids=()
  trap - INT TERM EXIT
  if ((midpoint_status != 0 || float_status != 0)); then
    echo "stabilized pilot lane failed: durable=$durable_status midpoint=$midpoint_status float=$float_status" >&2
    exit 1
  fi
else
  run_durable_lane
  run_midpoint_lane
  run_float_lane
fi

cmp "$out_dir/integer-durable-3.nsrlpm" "$out_dir/integer-midpoint-1.nsrlpm"
cmp "$out_dir/integer-durable-3.nsrlpo" "$out_dir/integer-midpoint-1.nsrlpo"

node scripts/freeze-production-stabilized-pilot-v1.mjs \
  --run-dir "$out_dir" --out "$checkpoint_out"
echo "production-model-v1 p10m stabilized pilot completed"
