#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_dir="${NSRL_PRODUCTION_LIVENESS_OUT_DIR:-data/experiments/production-model-v1/p10m-liveness-audit}"
checkpoint_out="${NSRL_PRODUCTION_LIVENESS_CHECKPOINT_OUT:-benchmarks/production-model-v1/p10m-liveness-audit.json}"
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
  --model "$out_dir/initial.nsrlpm" --trace "$out_dir/dev-initial.json" \
  --context-tokens 64 --max-windows 256

integer_chunk() {
  local lane="$1" index="$2" output_shift="$3" model_in="$4" optimizer_in="${5:-}"
  local args=(
    full-train-smoke --tokenizer "$tokenizer" --tokens "$train_tokens"
    --model "$model_in" --model-out "$out_dir/$lane-$index.nsrlpm"
    --optimizer-state-out "$out_dir/$lane-$index.nsrlpo"
    --trace "$out_dir/$lane-$index.json"
    --context-tokens 64 --max-windows 256 --evaluation-windows 64
    --epochs 1 --batch-windows 4 --max-optimizer-steps 16
    --matrix-learning-rate-shift 25
    --q-learning-rate-shift 29 --k-learning-rate-shift 35
    --v-learning-rate-shift 33 --o-learning-rate-shift 25
    --up-learning-rate-shift 25 --gate-learning-rate-shift 25
    --down-learning-rate-shift 25
    --vector-learning-rate-shift 23
    --embedding-learning-rate-shift 17
    --output-learning-rate-shift "$output_shift" --output-backward-shift 8
  )
  if [[ -n "$optimizer_in" ]]; then args+=(--optimizer-state "$optimizer_in"); fi
  "$binary" "${args[@]}"
}

run_negative_control() {
  integer_chunk negative 0 36 "$out_dir/initial.nsrlpm"
  node scripts/check-production-training-liveness-v1.mjs \
    --trace "$out_dir/negative-0.json" --interval 0 \
    --state-out "$out_dir/negative-state-0.json" \
    --event-out "$out_dir/negative-event-0.json" --expect-dead
}

run_positive_control() {
  local model_in="$out_dir/initial.nsrlpm" optimizer_in="" state_in=""
  for index in 0 1 2 3; do
    integer_chunk positive "$index" 34 "$model_in" "$optimizer_in"
    "$binary" evaluate \
      --tokenizer "$tokenizer" --tokens "$dev_tokens" \
      --model "$out_dir/positive-$index.nsrlpm" \
      --trace "$out_dir/positive-dev-$index.json" \
      --context-tokens 64 --max-windows 256
    args=(
      --trace "$out_dir/positive-$index.json" --interval "$index"
      --state-out "$out_dir/positive-state-$index.json"
      --event-out "$out_dir/positive-event-$index.json"
      --dev-initial "$out_dir/dev-initial.json"
      --dev-current "$out_dir/positive-dev-$index.json"
    )
    if [[ -n "$state_in" ]]; then args+=(--state-in "$state_in"); fi
    node scripts/check-production-training-liveness-v1.mjs "${args[@]}"
    model_in="$out_dir/positive-$index.nsrlpm"
    optimizer_in="$out_dir/positive-$index.nsrlpo"
    state_in="$out_dir/positive-state-$index.json"
  done
}

run_negative_control & negative_pid=$!
run_positive_control & positive_pid=$!
set +e
wait "$negative_pid"; negative_status=$?
wait "$positive_pid"; positive_status=$?
set -e
if ((negative_status != 0 || positive_status != 0)); then
  echo "liveness audit lane failed: negative=$negative_status positive=$positive_status" >&2
  exit 1
fi

node scripts/freeze-production-liveness-audit-v1.mjs \
  --run-dir "$out_dir" --out "$checkpoint_out"
echo "production-model-v1 liveness audit completed"
