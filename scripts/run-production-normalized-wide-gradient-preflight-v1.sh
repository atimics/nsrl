#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_dir="${NSRL_PRODUCTION_NORMALIZED_WIDE_OUT_DIR:-data/experiments/production-model-v1/p10m-normalized-wide-gradient-preflight}"
checkpoint_out="${NSRL_PRODUCTION_NORMALIZED_WIDE_CHECKPOINT_OUT:-benchmarks/production-model-v1/p10m-normalized-wide-gradient-preflight.json}"
tokenizer="${NSRL_PRODUCTION_TOKENIZER:-data/processed/production-corpus-v1/tokenizer.nsrlbpe}"
train_tokens="${NSRL_PRODUCTION_TRAIN_TOKENS:-data/processed/production-corpus-v1/train.nsrltok}"
dev_tokens="${NSRL_PRODUCTION_DEV_TOKENS:-data/processed/production-corpus-v1/dev.nsrltok}"
initial_model="data/experiments/production-model-v1/p10m-up-forward-scale-training/initial.nsrlpm"
binary="target/release/nsrl-production-model"

mkdir -p "$out_dir"
cargo build --release -p nsrl-train --bin nsrl-production-model

train_lane() {
  local suffix="$1" up_shift="$2" output_shift="$3"
  if [[ -s "$out_dir/model-q23-newton${suffix}.nsrlpm" \
    && -s "$out_dir/optimizer-q23-newton${suffix}.nsrlpo" \
    && -s "$out_dir/train-q23-newton${suffix}.json" ]]; then
    return
  fi
  "$binary" full-train-smoke \
    --tokenizer "$tokenizer" --tokens "$train_tokens" \
    --model "$initial_model" \
    --model-out "$out_dir/model-q23-newton${suffix}.nsrlpm" \
    --optimizer-state-out "$out_dir/optimizer-q23-newton${suffix}.nsrlpo" \
    --trace "$out_dir/train-q23-newton${suffix}.json" \
    --context-tokens 64 --max-windows 1024 --evaluation-windows 64 \
    --epochs 1 --batch-windows 4 --max-optimizer-steps 64 \
    --matrix-learning-rate-shift 25 \
    --q-learning-rate-shift 29 --k-learning-rate-shift 26 \
    --v-learning-rate-shift 30 --o-learning-rate-shift 25 \
    --up-learning-rate-shift "$up_shift" --gate-learning-rate-shift 23 \
    --down-learning-rate-shift 25 --vector-learning-rate-shift 23 \
    --embedding-learning-rate-shift 17 \
    --output-learning-rate-shift "$output_shift" --output-backward-shift 8 \
    --probability-gradient-fractional-bits 23 \
    --probability-normalization q47-newton1
}

evaluate_lane() {
  local suffix="$1"
  if [[ ! -s "$out_dir/dev-q23-newton${suffix}.json" ]]; then
    "$binary" evaluate \
      --tokenizer "$tokenizer" --tokens "$dev_tokens" \
      --model "$out_dir/model-q23-newton${suffix}.nsrlpm" \
      --trace "$out_dir/dev-q23-newton${suffix}.json" \
      --context-tokens 64 --max-windows 256
  fi
  if [[ ! -s "$out_dir/residual-q23-newton${suffix}.json" ]]; then
    node scripts/analyze-production-optimizer-residuals-v1.mjs \
      --optimizer "$out_dir/optimizer-q23-newton${suffix}.nsrlpo" \
      --trace "$out_dir/train-q23-newton${suffix}.json" \
      --out "$out_dir/residual-q23-newton${suffix}.json"
  fi
}

train_lane "" 22 34
train_lane "-up21" 21 34
train_lane "-output33" 22 33
evaluate_lane ""
evaluate_lane "-up21"
evaluate_lane "-output33"

if [[ ! -s "$out_dir/compare-q23-newton-up21.json" ]]; then
  "$binary" compare-evaluate \
    --tokenizer "$tokenizer" --tokens "$dev_tokens" \
    --model "$out_dir/model-q23-newton.nsrlpm" \
    --candidate-model "$out_dir/model-q23-newton-up21.nsrlpm" \
    --trace "$out_dir/compare-q23-newton-up21.json" \
    --context-tokens 64 --max-windows 256
fi
if [[ ! -s "$out_dir/compare-q23-newton-output33.json" ]]; then
  "$binary" compare-evaluate \
    --tokenizer "$tokenizer" --tokens "$dev_tokens" \
    --model "$out_dir/model-q23-newton.nsrlpm" \
    --candidate-model "$out_dir/model-q23-newton-output33.nsrlpm" \
    --trace "$out_dir/compare-q23-newton-output33.json" \
    --context-tokens 64 --max-windows 256
fi

if [[ ! -s "$out_dir/replay-selected.nsrlpm" \
  || ! -s "$out_dir/replay-selected.nsrlpo" || ! -s "$out_dir/replay.json" ]]; then
  "$binary" full-train-smoke \
    --tokenizer "$tokenizer" --tokens "$train_tokens" \
    --model "$initial_model" --model-out "$out_dir/replay-selected.nsrlpm" \
    --optimizer-state-out "$out_dir/replay-selected.nsrlpo" \
    --trace "$out_dir/replay.json" \
    --context-tokens 64 --max-windows 1024 --evaluation-windows 64 \
    --epochs 1 --batch-windows 4 --max-optimizer-steps 64 \
    --matrix-learning-rate-shift 25 \
    --q-learning-rate-shift 29 --k-learning-rate-shift 26 \
    --v-learning-rate-shift 30 --o-learning-rate-shift 25 \
    --up-learning-rate-shift 22 --gate-learning-rate-shift 23 \
    --down-learning-rate-shift 25 --vector-learning-rate-shift 23 \
    --embedding-learning-rate-shift 17 \
    --output-learning-rate-shift 34 --output-backward-shift 8 \
    --probability-gradient-fractional-bits 23 \
    --probability-normalization q47-newton1
fi

cmp "$out_dir/model-q23-newton.nsrlpm" "$out_dir/replay-selected.nsrlpm"
cmp "$out_dir/optimizer-q23-newton.nsrlpo" "$out_dir/replay-selected.nsrlpo"
node scripts/freeze-production-normalized-wide-gradient-preflight-v1.mjs \
  --run-dir "$out_dir" --out "$checkpoint_out"
echo "production-model-v1 p10m normalized wide-gradient preflight completed"
