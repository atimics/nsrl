#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_dir="${NSRL_OPEN_GENERATION_OUT_DIR:-data/experiments/open-generation-v1/p10m-kv-scaling-baseline}"
manifest="${NSRL_OPEN_GENERATION_MANIFEST:-benchmarks/open-generation-v1/manifest.tsv}"
tokenizer="${NSRL_OPEN_GENERATION_TOKENIZER:-data/processed/production-corpus-v1/tokenizer.nsrlbpe}"
model="${NSRL_OPEN_GENERATION_MODEL:-data/experiments/production-model-v1/p10m-kv-scaling-readiness/integer-model-7.nsrlpm}"
top_k="${NSRL_OPEN_GENERATION_TOP_K:-40}"
checkpoint_out="${NSRL_OPEN_GENERATION_CHECKPOINT_OUT:-benchmarks/open-generation-v1/p10m-kv-scaling-baseline.json}"

mkdir -p "$out_dir"
work_dir="$(mktemp -d "$out_dir/.run.XXXXXX")"
trap 'rm -rf "$work_dir"' EXIT

cargo build --release -p nsrl-train \
  --bin nsrl-open-generation-run \
  --bin nsrl-open-generation-modeling-run

target/release/nsrl-open-generation-run \
  --manifest "$manifest" \
  --tokenizer "$tokenizer" \
  --model "$model" \
  --samples-out "$work_dir/samples.jsonl" \
  --decoder-traces-out "$work_dir/decoder-traces.jsonl" \
  --trace "$work_dir/run.json" \
  --top-k "$top_k"

target/release/nsrl-open-generation-modeling-run \
  --manifest "$manifest" \
  --tokenizer "$tokenizer" \
  --model "$model" \
  --trace "$work_dir/modeling.json"

mv "$work_dir/samples.jsonl" "$out_dir/samples.jsonl"
mv "$work_dir/decoder-traces.jsonl" "$out_dir/decoder-traces.jsonl"
mv "$work_dir/run.json" "$out_dir/run.json"
mv "$work_dir/modeling.json" "$out_dir/modeling.json"

node scripts/evaluate-open-generation-development-v1.mjs \
  --manifest "$manifest" \
  --run "$out_dir/run.json" \
  --samples "$out_dir/samples.jsonl" \
  --decoder-traces "$out_dir/decoder-traces.jsonl" \
  --modeling "$out_dir/modeling.json" \
  --runner-binary target/release/nsrl-open-generation-run \
  --modeling-runner-binary target/release/nsrl-open-generation-modeling-run \
  --candidate-model "$model" \
  --candidate-tokenizer "$tokenizer" \
  --out "$out_dir/result.json"

node scripts/evaluate-open-generation-development-v1.mjs \
  --manifest "$manifest" \
  --run "$out_dir/run.json" \
  --samples "$out_dir/samples.jsonl" \
  --decoder-traces "$out_dir/decoder-traces.jsonl" \
  --modeling "$out_dir/modeling.json" \
  --runner-binary target/release/nsrl-open-generation-run \
  --modeling-runner-binary target/release/nsrl-open-generation-modeling-run \
  --candidate-model "$model" \
  --candidate-tokenizer "$tokenizer" \
  --check "$out_dir/result.json"

node scripts/freeze-open-generation-development-baseline-v1.mjs \
  --run-dir "$out_dir" \
  --out "$checkpoint_out"

echo "open-generation-v1 development baseline completed: $out_dir/result.json"
