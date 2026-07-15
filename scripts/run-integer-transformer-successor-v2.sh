#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_dir="${1:-data/experiments/integer-transformer-successor-v2/latest}"
target_dir="${NSRL_SUCCESSOR_TARGET_DIR:-target/integer-transformer-successor-v2}"
manifest="benchmarks/integer-transformer-successor-v2/manifest.tsv"
candidate="data/experiments/integer-transformer-proof-v1/candidate-default/candidate.nsrlmt"
train_tokens="benchmarks/integer-transformer-proof-v1/train.txt"
eval_tokens="benchmarks/integer-transformer-proof-v1/eval.txt"
frozen_float_model="benchmarks/integer-transformer-successor-v2/float-transformer.model"
float_runner_hash="0xd0b37c9eb3275c5b"

mkdir -p "$out_dir"
CARGO_TARGET_DIR="$target_dir" cargo build --release -p nsrl-train \
  --bin nsrl-mini-transformer-eval --features mini-heads-8,mini-calibrated
CARGO_TARGET_DIR="$target_dir" cargo build --release -p nsrl-eval --bin nsrl-eval

"$target_dir/release/nsrl-eval" successor-manifest --manifest "$manifest" \
  > "$out_dir/manifest.json"

"$target_dir/release/nsrl-mini-transformer-eval" \
  --tokens "$eval_tokens" \
  --model "$candidate" \
  --stride 1 \
  --attention linear \
  --position nope \
  --ablation transformer-only \
  --ablated-model-out "$out_dir/candidate.nsrlmt" \
  --logits-out "$out_dir/candidate.logits.i32le" \
  --out "$out_dir/candidate.eval.json"

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python3 scripts/float-transformer-successor-v2.py \
  --train "$train_tokens" \
  --eval "$eval_tokens" \
  --model-out "$out_dir/float-transformer.model" \
  --logits-out "$out_dir/float-transformer.logits.i32le" \
  --trace-out "$out_dir/float-transformer.eval.json" \
  --runner-hash "$float_runner_hash"
cmp "$frozen_float_model" "$out_dir/float-transformer.model"

node scripts/build-integer-transformer-successor-v2-results.mjs \
  --manifest "$manifest" \
  --candidate-trace "$out_dir/candidate.eval.json" \
  --candidate-logits "$out_dir/candidate.logits.i32le" \
  --float-trace "$out_dir/float-transformer.eval.json" \
  --float-logits "$out_dir/float-transformer.logits.i32le" \
  --out "$out_dir/results.tsv"

set +e
"$target_dir/release/nsrl-eval" successor-check \
  --manifest "$manifest" \
  --results "$out_dir/results.tsv" | tee "$out_dir/check.json"
check_status="${PIPESTATUS[0]}"
set -e

if [[ "$check_status" -eq 0 ]]; then
  echo "integer-transformer-successor-v2 passed: $out_dir"
  exit 0
fi
if [[ "$check_status" -eq 1 ]]; then
  echo "integer-transformer-successor-v2 validly falsified: $out_dir" >&2
  exit 1
fi
echo "integer-transformer-successor-v2 artifact validation failed: $out_dir" >&2
exit 2
