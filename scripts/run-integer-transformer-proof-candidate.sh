#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_dir="${1:-}"
if [[ -z "$out_dir" ]]; then
  echo "usage: scripts/run-integer-transformer-proof-candidate.sh OUT_DIR" >&2
  exit 2
fi

manifest="benchmarks/integer-transformer-proof-v1/manifest.tsv"
baselines="benchmarks/integer-transformer-proof-v1/baselines.tsv"
train_tokens="benchmarks/integer-transformer-proof-v1/train.txt"
eval_tokens="benchmarks/integer-transformer-proof-v1/eval.txt"
context=64
max_windows="${NSRL_PROOF_MAX_WINDOWS:-2048}"
epochs="${NSRL_PROOF_EPOCHS:-1}"
batch_windows="${NSRL_PROOF_BATCH_WINDOWS:-4}"
workers="${NSRL_PROOF_WORKERS:-4}"

for value in "$max_windows" "$epochs" "$batch_windows" "$workers"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "proof training settings must be positive integers: $value" >&2
    exit 2
  fi
done

mkdir -p "$out_dir"
cargo run -q -p nsrl-eval -- manifest --manifest "$manifest" > "$out_dir/manifest.json"
node scripts/run-integer-transformer-proof-baselines.mjs \
  --manifest "$manifest" \
  --out "$out_dir/baselines.tsv"
cmp "$baselines" "$out_dir/baselines.tsv"

cargo build --release -p nsrl-train -p nsrl-eval \
  --bin nsrl-train --bin nsrl-mini-transformer-eval --bin nsrl-eval

train_bytes="$(wc -c < "$train_tokens" | tr -d ' ')"
trainable=$((train_bytes > context ? train_bytes - context : 1))
train_stride=$(((trainable + max_windows - 1) / max_windows))
((train_stride > 0)) || train_stride=1

target/release/nsrl-train \
  --mode mini-transformer-mlp \
  --tokens "$train_tokens" \
  --model-out "$out_dir/candidate.nsrlmt" \
  --epochs "$epochs" \
  --seq-len "$context" \
  --stride "$train_stride" \
  --max-windows "$max_windows" \
  --batch-windows "$batch_windows" \
  --tokenizer identity \
  --mini-transformer-attention linear \
  --mini-transformer-position nope \
  --mini-transformer-batch-mode map-reduce \
  --mini-transformer-map-reduce-workers "$workers" \
  --mini-transformer-trace-detail summary \
  --trace "$out_dir/train.trace.jsonl" \
  --progress-out "$out_dir/train.progress.jsonl" \
  --progress-interval-batches 64

target/release/nsrl-mini-transformer-eval \
  --tokens "$eval_tokens" \
  --model "$out_dir/candidate.nsrlmt" \
  --stride 1 \
  --attention linear \
  --position nope \
  --out "$out_dir/candidate.eval.json"

node scripts/build-integer-transformer-proof-results.mjs \
  --manifest "$manifest" \
  --baselines "$baselines" \
  --candidate-trace "$out_dir/candidate.eval.json" \
  --out "$out_dir/proof-results.tsv"

set +e
target/release/nsrl-eval check \
  --manifest "$manifest" \
  --results "$out_dir/proof-results.tsv" | tee "$out_dir/proof-check.json"
proof_status="${PIPESTATUS[0]}"
set -e

if [[ "$proof_status" -eq 0 ]]; then
  echo "integer-transformer-proof-v1 passed: $out_dir"
elif [[ "$proof_status" -eq 1 ]]; then
  echo "integer-transformer-proof-v1 measured but did not pass: $out_dir" >&2
else
  echo "integer-transformer-proof-v1 artifact validation failed: $out_dir" >&2
fi
exit "$proof_status"
