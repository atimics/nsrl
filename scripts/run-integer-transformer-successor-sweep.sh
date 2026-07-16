#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_dir="${1:-data/experiments/integer-transformer-proof-v1/successor-sweep}"
target_dir="${NSRL_PROOF_TARGET_DIR:-target/integer-transformer-proof-h8}"
train_tokens="benchmarks/integer-transformer-proof-v1/train.txt"
eval_tokens="benchmarks/integer-transformer-proof-v1/eval.txt"
context=64

mkdir -p "$out_dir"
CARGO_TARGET_DIR="$target_dir" cargo build --release -p nsrl-train \
  --bin nsrl-train --bin nsrl-mini-transformer-eval \
  --features mini-heads-8,mini-calibrated

trainable=$(($(wc -c < "$train_tokens") - context))

run_variant() {
  local name="$1"
  local step_shift="$2"
  local epochs="$3"
  local batch_windows="$4"
  local margin_q15="$5"
  local frequency_cap="$6"
  local max_windows="$7"
  local attention="$8"
  local position="$9"
  local stride=$(((trainable + max_windows - 1) / max_windows))
  local full_model="$out_dir/$name.full.nsrlmt"
  local model="$out_dir/$name.nsrlmt"
  local trace="$out_dir/$name.train.jsonl"
  local evaluation="$out_dir/$name.eval.json"

  if [[ -f "$evaluation" && -f "$model" ]]; then
    echo "successor sweep reuse: $name"
    return
  fi

  echo "successor sweep train: $name"
  "$target_dir/release/nsrl-train" \
    --mode mini-transformer-adam \
    --tokens "$train_tokens" \
    --model-out "$full_model" \
    --rms-norm-initial-gamma-q15 16384 \
    --adam-step-shift "$step_shift" \
    --argmax-margin-weight-q15 "$margin_q15" \
    --target-frequency-cap "$frequency_cap" \
    --target-frequency-min-weight-q15 4096 \
    --epochs "$epochs" \
    --seq-len "$context" \
    --stride "$stride" \
    --max-windows "$max_windows" \
    --batch-windows "$batch_windows" \
    --tokenizer identity \
    --mini-transformer-attention "$attention" \
    --mini-transformer-position "$position" \
    --mini-transformer-batch-mode map-reduce \
    --mini-transformer-map-reduce-workers 4 \
    --trace "$trace"

  if [[ "$position" == "nope" ]]; then
    "$target_dir/release/nsrl-mini-transformer-eval" \
      --tokens "$eval_tokens" \
      --model "$full_model" \
      --stride 1 \
      --attention "$attention" \
      --position "$position" \
      --ablation transformer-only \
      --ablated-model-out "$model" \
      --out "$evaluation"
  else
    mv "$full_model" "$model"
    "$target_dir/release/nsrl-mini-transformer-eval" \
      --tokens "$eval_tokens" \
      --model "$model" \
      --stride 1 \
      --attention "$attention" \
      --position "$position" \
      --out "$evaluation"
  fi
}

# The matrix varies update scale, duration, batch geometry, class balancing,
# training-window coverage, attention, and position policy. The frozen task and
# evaluation geometry never change.
run_variant s4-e1 4 1 16 1024 0 512 linear nope
run_variant s5-e1 5 1 16 1024 0 512 linear nope
run_variant s6-e1 6 1 16 1024 0 512 linear nope
run_variant s7-e1 7 1 16 1024 0 512 linear nope
run_variant s5-e2 5 2 16 1024 0 512 linear nope
run_variant s5-e4 5 4 16 1024 0 512 linear nope
run_variant s5-b8 5 1 8 1024 0 512 linear nope
run_variant s5-margin0 5 1 16 0 0 512 linear nope
run_variant s5-margin4096 5 1 16 4096 0 512 linear nope
run_variant s5-margin16384 5 1 16 16384 0 512 linear nope
run_variant s5-frequency16 5 1 16 1024 16 512 linear nope
run_variant s5-frequency32 5 1 16 1024 32 512 linear nope
run_variant s5-w1024 5 1 16 1024 0 1024 linear nope
run_variant s5-w2048 5 1 16 1024 0 2048 linear nope
run_variant s5-softmax 5 1 16 1024 0 512 base2-softmax nope
run_variant s5-learned-position 5 1 16 1024 0 512 linear learned-absolute

node scripts/summarize-integer-transformer-successor-sweep.mjs \
  --dir "$out_dir" \
  --out benchmarks/integer-transformer-proof-v1/transformer-successor-sweep.json
