#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_dir="${NSRL_PRODUCTION_WIDE_PROBABILITY_OUT_DIR:-data/experiments/production-model-v1/p10m-wide-probability-gradient-preflight}"
checkpoint_out="${NSRL_PRODUCTION_WIDE_PROBABILITY_CHECKPOINT_OUT:-benchmarks/production-model-v1/p10m-wide-probability-gradient-preflight.json}"
tokenizer="${NSRL_PRODUCTION_TOKENIZER:-data/processed/production-corpus-v1/tokenizer.nsrlbpe}"
train_tokens="${NSRL_PRODUCTION_TRAIN_TOKENS:-data/processed/production-corpus-v1/train.nsrltok}"
dev_tokens="${NSRL_PRODUCTION_DEV_TOKENS:-data/processed/production-corpus-v1/dev.nsrltok}"
initial_model="data/experiments/production-model-v1/p10m-up-forward-scale-training/initial.nsrlpm"
binary="target/release/nsrl-production-model"

mkdir -p "$out_dir"
cargo build --release -p nsrl-train --bin nsrl-production-model

train_candidate() {
  local bits="$1" model_out="$2" optimizer_out="$3" trace_out="$4"
  "$binary" full-train-smoke \
    --tokenizer "$tokenizer" --tokens "$train_tokens" \
    --model "$initial_model" --model-out "$model_out.tmp" \
    --optimizer-state-out "$optimizer_out.tmp" --trace "$trace_out.tmp" \
    --context-tokens 64 --max-windows 1024 --evaluation-windows 64 \
    --epochs 1 --batch-windows 4 --max-optimizer-steps 64 \
    --matrix-learning-rate-shift 25 \
    --q-learning-rate-shift 29 --k-learning-rate-shift 26 \
    --v-learning-rate-shift 30 --o-learning-rate-shift 25 \
    --up-learning-rate-shift 22 --gate-learning-rate-shift 23 \
    --down-learning-rate-shift 25 --vector-learning-rate-shift 23 \
    --embedding-learning-rate-shift 17 \
    --output-learning-rate-shift 34 --output-backward-shift 8 \
    --probability-gradient-fractional-bits "$bits"
  mv "$model_out.tmp" "$model_out"
  mv "$optimizer_out.tmp" "$optimizer_out"
  mv "$trace_out.tmp" "$trace_out"
}

for bits in 19 23; do
  if [[ ! -s "$out_dir/model-q$bits.nsrlpm" \
    || ! -s "$out_dir/optimizer-q$bits.nsrlpo" \
    || ! -s "$out_dir/train-q$bits.json" ]]; then
    train_candidate "$bits" "$out_dir/model-q$bits.nsrlpm" \
      "$out_dir/optimizer-q$bits.nsrlpo" "$out_dir/train-q$bits.json"
  fi
  if [[ ! -s "$out_dir/dev-q$bits.json" ]]; then
    "$binary" evaluate \
      --tokenizer "$tokenizer" --tokens "$dev_tokens" \
      --model "$out_dir/model-q$bits.nsrlpm" \
      --trace "$out_dir/dev-q$bits.json.tmp" \
      --context-tokens 64 --max-windows 256
    mv "$out_dir/dev-q$bits.json.tmp" "$out_dir/dev-q$bits.json"
  fi
  if [[ ! -s "$out_dir/residual-q$bits.json" ]]; then
    node scripts/analyze-production-optimizer-residuals-v1.mjs \
      --optimizer "$out_dir/optimizer-q$bits.nsrlpo" \
      --trace "$out_dir/train-q$bits.json" \
      --out "$out_dir/residual-q$bits.json"
  fi
done

selected_bits="$(node -e '
  const fs = require("fs");
  const dir = process.argv[1];
  const rows = [19, 23].map((bits) => ({
    bits,
    total: JSON.parse(fs.readFileSync(`${dir}/dev-q${bits}.json`, "utf8"))
      .evaluation.total_millibits,
  }));
  rows.sort((left, right) => left.total - right.total || left.bits - right.bits);
  process.stdout.write(String(rows[0].bits));
' "$out_dir")"
echo "wide probability-gradient preflight selected Q$selected_bits"

if [[ ! -s "$out_dir/replay-selected.nsrlpm" \
  || ! -s "$out_dir/replay-selected.nsrlpo" || ! -s "$out_dir/replay.json" ]]; then
  train_candidate "$selected_bits" "$out_dir/replay-selected.nsrlpm" \
    "$out_dir/replay-selected.nsrlpo" "$out_dir/replay.json"
fi
cmp "$out_dir/model-q$selected_bits.nsrlpm" "$out_dir/replay-selected.nsrlpm"
cmp "$out_dir/optimizer-q$selected_bits.nsrlpo" "$out_dir/replay-selected.nsrlpo"

node scripts/freeze-production-wide-probability-gradient-preflight-v1.mjs \
  --run-dir "$out_dir" --out "$checkpoint_out"
echo "production-model-v1 p10m wide probability-gradient preflight completed"
