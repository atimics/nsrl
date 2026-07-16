#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_dir="${NSRL_PRODUCTION_KV_READINESS_OUT_DIR:-data/experiments/production-model-v1/p10m-kv-scaling-readiness}"
checkpoint_out="${NSRL_PRODUCTION_KV_READINESS_CHECKPOINT_OUT:-benchmarks/production-model-v1/p10m-kv-scaling-readiness.json}"
tokenizer="${NSRL_PRODUCTION_TOKENIZER:-data/processed/production-corpus-v1/tokenizer.nsrlbpe}"
train_tokens="${NSRL_PRODUCTION_TRAIN_TOKENS:-data/processed/production-corpus-v1/train.nsrltok}"
dev_tokens="${NSRL_PRODUCTION_DEV_TOKENS:-data/processed/production-corpus-v1/dev.nsrltok}"
binary="target/release/nsrl-production-model"

mkdir -p "$out_dir"
cargo build --release -p nsrl-train --bin nsrl-production-model

if [[ ! -s "$out_dir/initial.nsrlpm" || ! -s "$out_dir/init.json" ]]; then
  "$binary" init \
    --profile p10m --tokenizer "$tokenizer" \
    --model-out "$out_dir/initial.nsrlpm.tmp" --trace "$out_dir/init.json.tmp" \
    --seed 7 --output-init-amplitude 1 --output-forward-shift 14
  mv "$out_dir/initial.nsrlpm.tmp" "$out_dir/initial.nsrlpm"
  mv "$out_dir/init.json.tmp" "$out_dir/init.json"
fi
if [[ ! -s "$out_dir/integer-dev-initial.json" ]]; then
  "$binary" evaluate \
    --tokenizer "$tokenizer" --tokens "$dev_tokens" \
    --model "$out_dir/initial.nsrlpm" \
    --trace "$out_dir/integer-dev-initial.json.tmp" \
    --context-tokens 64 --max-windows 256
  mv "$out_dir/integer-dev-initial.json.tmp" "$out_dir/integer-dev-initial.json"
fi

train_integer_chunk() {
  local model_in="$1" optimizer_in="$2" model_out="$3" optimizer_out="$4"
  local trace_out="$5" steps="$6"
  local args=(
    full-train-smoke --tokenizer "$tokenizer" --tokens "$train_tokens"
    --model "$model_in" --model-out "$model_out.tmp"
    --optimizer-state-out "$optimizer_out.tmp" --trace "$trace_out.tmp"
    --context-tokens 64 --max-windows 2048 --evaluation-windows 64
    --epochs 1 --batch-windows 4 --max-optimizer-steps "$steps"
    --matrix-learning-rate-shift 25
    --q-learning-rate-shift 29 --k-learning-rate-shift 26
    --v-learning-rate-shift 30 --o-learning-rate-shift 25
    --up-learning-rate-shift 25 --gate-learning-rate-shift 25
    --down-learning-rate-shift 25 --vector-learning-rate-shift 23
    --embedding-learning-rate-shift 17
    --output-learning-rate-shift 34 --output-backward-shift 8
  )
  if [[ -n "$optimizer_in" ]]; then args+=(--optimizer-state "$optimizer_in"); fi
  "$binary" "${args[@]}"
  mv "$model_out.tmp" "$model_out"
  mv "$optimizer_out.tmp" "$optimizer_out"
  mv "$trace_out.tmp" "$trace_out"
}

run_integer_lane() {
  local interval model_in optimizer_in state_in status
  for interval in 0 1 2 3 4 5 6 7; do
    required=("integer-model-$interval.nsrlpm" "integer-optimizer-$interval.nsrlpo"
      "integer-train-$interval.json" "integer-dev-$interval.json"
      "integer-state-$interval.json" "integer-event-$interval.json")
    complete=1
    for file in "${required[@]}"; do
      if [[ ! -s "$out_dir/$file" ]]; then complete=0; fi
    done
    if ((complete == 1)); then
      echo "K+V readiness reuse integer chunk $interval"
      continue
    fi

    if ((interval == 0)); then
      model_in="$out_dir/initial.nsrlpm"
      optimizer_in=""
      state_in=""
    else
      model_in="$out_dir/integer-model-$((interval - 1)).nsrlpm"
      optimizer_in="$out_dir/integer-optimizer-$((interval - 1)).nsrlpo"
      state_in="$out_dir/integer-state-$((interval - 1)).json"
    fi
    train_integer_chunk "$model_in" "$optimizer_in" \
      "$out_dir/integer-model-$interval.nsrlpm" \
      "$out_dir/integer-optimizer-$interval.nsrlpo" \
      "$out_dir/integer-train-$interval.json" 64

    "$binary" evaluate \
      --tokenizer "$tokenizer" --tokens "$dev_tokens" \
      --model "$out_dir/integer-model-$interval.nsrlpm" \
      --trace "$out_dir/integer-dev-$interval.json.tmp" \
      --context-tokens 64 --max-windows 256
    mv "$out_dir/integer-dev-$interval.json.tmp" \
      "$out_dir/integer-dev-$interval.json"

    check_args=(
      --trace "$out_dir/integer-train-$interval.json" --interval "$interval"
      --state-out "$out_dir/integer-state-$interval.json.tmp"
      --event-out "$out_dir/integer-event-$interval.json.tmp"
      --dev-initial "$out_dir/integer-dev-initial.json"
      --dev-current "$out_dir/integer-dev-$interval.json"
      --output-unlock-deadline-intervals 1
      --trunk-activation-deadline-intervals 1
      --require-trunk-update-by-interval 0
      --required-trunk-group k --required-trunk-group v
    )
    if [[ -n "$state_in" ]]; then check_args+=(--state-in "$state_in"); fi
    set +e
    node scripts/check-production-training-liveness-v1.mjs "${check_args[@]}"
    status=$?
    set -e
    mv "$out_dir/integer-state-$interval.json.tmp" \
      "$out_dir/integer-state-$interval.json"
    mv "$out_dir/integer-event-$interval.json.tmp" \
      "$out_dir/integer-event-$interval.json"
    if ((status != 0)); then
      echo "K+V readiness integer lane died at chunk $interval" >&2
      return "$status"
    fi
  done

  if [[ ! -s "$out_dir/integer-residual-analysis.json" ]]; then
    node scripts/analyze-production-optimizer-residuals-v1.mjs \
      --optimizer "$out_dir/integer-optimizer-7.nsrlpo" \
      --trace "$out_dir/integer-train-7.json" \
      --out "$out_dir/integer-residual-analysis.json.tmp"
    mv "$out_dir/integer-residual-analysis.json.tmp" \
      "$out_dir/integer-residual-analysis.json"
  fi

  if [[ ! -s "$out_dir/integer-replay-final.nsrlpm" \
    || ! -s "$out_dir/integer-replay-final.nsrlpo" \
    || ! -s "$out_dir/integer-replay.json" ]]; then
    train_integer_chunk \
      "$out_dir/integer-model-3.nsrlpm" \
      "$out_dir/integer-optimizer-3.nsrlpo" \
      "$out_dir/integer-replay-final.nsrlpm" \
      "$out_dir/integer-replay-final.nsrlpo" \
      "$out_dir/integer-replay.json" 256
  fi
  cmp "$out_dir/integer-model-7.nsrlpm" "$out_dir/integer-replay-final.nsrlpm"
  cmp "$out_dir/integer-optimizer-7.nsrlpo" "$out_dir/integer-replay-final.nsrlpo"
}

run_float_lane() {
  local chunk start_window resume_path previous_trace tmp_npz status
  resume_path=""
  previous_trace=""
  for chunk in 0 1 2 3 4 5 6 7; do
    if [[ -s "$out_dir/float-$chunk.npz" \
      && -s "$out_dir/float-$chunk.json" \
      && -s "$out_dir/float-event-$chunk.json" ]]; then
      echo "K+V readiness reuse float chunk $chunk"
      resume_path="$out_dir/float-$chunk.npz"
      previous_trace="$out_dir/float-$chunk.json"
      continue
    fi
    start_window=$((chunk * 256))
    tmp_npz="$out_dir/float-$chunk.tmp.npz"
    args=(
      --math-contract legacy-v1
      --model "$out_dir/initial.nsrlpm" --tokens "$train_tokens"
      --out "$tmp_npz" --trace "$out_dir/float-$chunk.json.tmp"
      --context-tokens 64 --start-window "$start_window" --max-windows 256
      --train-eval-max-windows 64 --epochs 1 --batch-windows 4
      --learning-rate-millionths 1000 --allow-partial-gates
      --eval-tokens "$dev_tokens" --eval-max-windows 256
    )
    if [[ -n "$resume_path" ]]; then args+=(--resume "$resume_path"); fi
    python3 scripts/production-float-twin-v1.py "${args[@]}"
    mv "$tmp_npz" "$out_dir/float-$chunk.npz"
    mv "$out_dir/float-$chunk.json.tmp" "$out_dir/float-$chunk.json"

    checker_args=(
      --trace "$out_dir/float-$chunk.json"
      --baseline "$out_dir/float-0.json"
      --out "$out_dir/float-event-$chunk.json.tmp"
      --chunk "$chunk"
    )
    if [[ -n "$previous_trace" ]]; then
      checker_args+=(--previous "$previous_trace")
    fi
    set +e
    node scripts/check-production-scaling-readiness-float-chunk-v1.mjs \
      "${checker_args[@]}"
    status=$?
    set -e
    mv "$out_dir/float-event-$chunk.json.tmp" \
      "$out_dir/float-event-$chunk.json"
    if ((status != 0)); then
      echo "K+V readiness float lane died at chunk $chunk" >&2
      return "$status"
    fi
    resume_path="$out_dir/float-$chunk.npz"
    previous_trace="$out_dir/float-$chunk.json"
  done

  if [[ ! -s "$out_dir/float-replay-final.npz" \
    || ! -s "$out_dir/float-replay.json" ]]; then
    python3 scripts/production-float-twin-v1.py \
      --math-contract legacy-v1 \
      --model "$out_dir/initial.nsrlpm" --tokens "$train_tokens" \
      --resume "$out_dir/float-3.npz" \
      --out "$out_dir/float-replay-final.tmp.npz" \
      --trace "$out_dir/float-replay.json.tmp" \
      --context-tokens 64 --start-window 1024 --max-windows 1024 \
      --train-eval-max-windows 64 --epochs 1 --batch-windows 4 \
      --learning-rate-millionths 1000 --allow-partial-gates \
      --eval-tokens "$dev_tokens" --eval-max-windows 256
    mv "$out_dir/float-replay-final.tmp.npz" \
      "$out_dir/float-replay-final.npz"
    mv "$out_dir/float-replay.json.tmp" "$out_dir/float-replay.json"
  fi
  python3 scripts/check-production-float-artifact-equality-v1.py \
    --left "$out_dir/float-7.npz" \
    --right "$out_dir/float-replay-final.npz" \
    --out "$out_dir/float-replay-equality.json.tmp"
  mv "$out_dir/float-replay-equality.json.tmp" \
    "$out_dir/float-replay-equality.json"
}

if [[ "${NSRL_PRODUCTION_KV_READINESS_PARALLEL_LANES:-1}" == "1" ]]; then
  child_pids=()
  cleanup_children() {
    if ((${#child_pids[@]} > 0)); then
      kill "${child_pids[@]}" >/dev/null 2>&1 || true
    fi
  }
  trap cleanup_children INT TERM EXIT
  run_integer_lane & integer_pid=$!; child_pids+=("$integer_pid")
  run_float_lane & float_pid=$!; child_pids+=("$float_pid")
  set +e
  wait "$integer_pid"; integer_status=$?
  if ((integer_status != 0)); then cleanup_children; fi
  wait "$float_pid"; float_status=$?
  set -e
  child_pids=()
  trap - INT TERM EXIT
  if ((integer_status != 0 || float_status != 0)); then
    echo "K+V readiness lane failed: integer=$integer_status float=$float_status" >&2
    exit 1
  fi
else
  run_integer_lane
  run_float_lane
fi

node scripts/freeze-production-kv-scaling-readiness-v1.mjs \
  --run-dir "$out_dir" --out "$checkpoint_out"
echo "production-model-v1 p10m K+V scaling readiness completed"
