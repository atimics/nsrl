#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_dir="${NSRL_PRODUCTION_GATE_PREFLIGHT_OUT_DIR:-data/experiments/production-model-v1/p10m-gate-boundary-preflight}"
checkpoint_out="${NSRL_PRODUCTION_GATE_PREFLIGHT_CHECKPOINT_OUT:-benchmarks/production-model-v1/p10m-gate-boundary-preflight.json}"
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
if [[ ! -s "$out_dir/dev-initial.json" ]]; then
  "$binary" evaluate \
    --tokenizer "$tokenizer" --tokens "$dev_tokens" \
    --model "$out_dir/initial.nsrlpm" --trace "$out_dir/dev-initial.json.tmp" \
    --context-tokens 64 --max-windows 256
  mv "$out_dir/dev-initial.json.tmp" "$out_dir/dev-initial.json"
fi

train_chunk() {
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
    --up-learning-rate-shift 25 --gate-learning-rate-shift 23
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

for interval in 0 1 2 3 4 5 6 7; do
  required=("model-$interval.nsrlpm" "optimizer-$interval.nsrlpo"
    "train-$interval.json" "dev-$interval.json"
    "state-$interval.json" "event-$interval.json")
  complete=1
  for file in "${required[@]}"; do
    if [[ ! -s "$out_dir/$file" ]]; then complete=0; fi
  done
  if ((complete == 1)); then
    echo "gate boundary preflight reuse interval $interval"
    continue
  fi

  if ((interval == 0)); then
    model_in="$out_dir/initial.nsrlpm"
    optimizer_in=""
    state_in=""
  else
    model_in="$out_dir/model-$((interval - 1)).nsrlpm"
    optimizer_in="$out_dir/optimizer-$((interval - 1)).nsrlpo"
    state_in="$out_dir/state-$((interval - 1)).json"
  fi
  train_chunk "$model_in" "$optimizer_in" \
    "$out_dir/model-$interval.nsrlpm" "$out_dir/optimizer-$interval.nsrlpo" \
    "$out_dir/train-$interval.json" 64

  "$binary" evaluate \
    --tokenizer "$tokenizer" --tokens "$dev_tokens" \
    --model "$out_dir/model-$interval.nsrlpm" \
    --trace "$out_dir/dev-$interval.json.tmp" \
    --context-tokens 64 --max-windows 256
  mv "$out_dir/dev-$interval.json.tmp" "$out_dir/dev-$interval.json"

  check_args=(
    --trace "$out_dir/train-$interval.json" --interval "$interval"
    --state-out "$out_dir/state-$interval.json.tmp"
    --event-out "$out_dir/event-$interval.json.tmp"
    --dev-initial "$out_dir/dev-initial.json"
    --dev-current "$out_dir/dev-$interval.json"
    --output-unlock-deadline-intervals 1
    --trunk-activation-deadline-intervals 1
    --require-trunk-update-by-interval 7
    --required-trunk-group gate
  )
  if [[ -n "$state_in" ]]; then check_args+=(--state-in "$state_in"); fi
  set +e
  node scripts/check-production-training-liveness-v1.mjs "${check_args[@]}"
  status=$?
  set -e
  mv "$out_dir/state-$interval.json.tmp" "$out_dir/state-$interval.json"
  mv "$out_dir/event-$interval.json.tmp" "$out_dir/event-$interval.json"
  if ((status != 0)); then
    echo "gate boundary candidate died at interval $interval" >&2
    exit "$status"
  fi
done

if [[ ! -s "$out_dir/residual-analysis.json" ]]; then
  node scripts/analyze-production-optimizer-residuals-v1.mjs \
    --optimizer "$out_dir/optimizer-7.nsrlpo" \
    --trace "$out_dir/train-7.json" \
    --out "$out_dir/residual-analysis.json.tmp"
  mv "$out_dir/residual-analysis.json.tmp" "$out_dir/residual-analysis.json"
fi

if [[ ! -s "$out_dir/replay-final.nsrlpm" || ! -s "$out_dir/replay-final.nsrlpo" \
  || ! -s "$out_dir/replay.json" ]]; then
  train_chunk "$out_dir/model-3.nsrlpm" "$out_dir/optimizer-3.nsrlpo" \
    "$out_dir/replay-final.nsrlpm" "$out_dir/replay-final.nsrlpo" \
    "$out_dir/replay.json" 256
fi

cmp "$out_dir/model-7.nsrlpm" "$out_dir/replay-final.nsrlpm"
cmp "$out_dir/optimizer-7.nsrlpo" "$out_dir/replay-final.nsrlpo"

node scripts/freeze-production-gate-boundary-preflight-v1.mjs \
  --run-dir "$out_dir" --out "$checkpoint_out"
echo "production-model-v1 p10m gate boundary preflight completed"
