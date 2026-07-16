#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_dir="${1:-data/experiments/integer-reachable-capacity-v1}"
model="${NSRL_REACHABLE_MODEL:-data/local-runs/literary-rms-adam-8k-shift8/model.nsrlmt}"
tokens="${NSRL_REACHABLE_TOKENS:-data/local-runs/literary-scale-8k-seq32/tokens.u8}"
max_windows="${NSRL_REACHABLE_MAX_WINDOWS:-256}"
batch_windows="${NSRL_REACHABLE_BATCH_WINDOWS:-32}"
learning_rate="${NSRL_REACHABLE_LEARNING_RATE:-16384}"

mkdir -p "$out_dir"
cargo build --release -p nsrl-train --bin nsrl-mini-transformer-low-rank-expert

for rank in 8 16 32; do
  for shift in 0 1 2 3 4; do
    for carry in off on; do
      id="rank${rank}-shift${shift}-carry${carry}"
      train_args=(
        train
        --tokens "$tokens"
        --model "$model"
        --out "$out_dir/$id.nsrlle"
        --trace "$out_dir/$id.train.json"
        --rank "$rank"
        --epochs 1
        --stride 1
        --max-windows "$max_windows"
        --batch-windows "$batch_windows"
        --learning-rate "$learning_rate"
        --learning-rate-shift "$shift"
      )
      if [[ "$carry" == "off" ]]; then
        train_args+=(--no-error-feedback)
      fi
      target/release/nsrl-mini-transformer-low-rank-expert "${train_args[@]}"
    done
  done
done

node scripts/summarize-integer-reachable-capacity.mjs \
  --input-dir "$out_dir" \
  --model "$model" \
  --tokens "$tokens" \
  --ranks 8,16,32 \
  --shifts 0,1,2,3,4 \
  --out "$out_dir/report.json"
node scripts/summarize-integer-reachable-capacity.mjs \
  --input-dir "$out_dir" \
  --model "$model" \
  --tokens "$tokens" \
  --ranks 8,16,32 \
  --shifts 0,1,2,3,4 \
  --out benchmarks/integer-reachable-capacity-v1/matrix.json

echo "integer reachable-capacity matrix written to $out_dir/report.json"
