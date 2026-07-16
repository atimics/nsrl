#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/nsrl-open-generation-v1.XXXXXX")"
trap 'rm -rf "$work_dir"' EXIT

cargo run -q -p nsrl-corpus --bin nsrl-subword -- train \
  --corpus benchmarks/integer-transformer-proof-v1/train.txt \
  --tokenizer-out "$work_dir/dev-tokenizer.nsrlbpe" \
  --trace "$work_dir/dev-tokenizer.trace.json" \
  --vocab-size 8192 \
  --min-pair-frequency 2

cmp benchmarks/open-generation-v1/dev-tokenizer.nsrlbpe "$work_dir/dev-tokenizer.nsrlbpe"
cmp benchmarks/open-generation-v1/dev-tokenizer.trace.json "$work_dir/dev-tokenizer.trace.json"

cargo run -q -p nsrl-corpus --bin nsrl-subword -- encode \
  --corpus benchmarks/integer-transformer-proof-v1/eval.txt \
  --tokenizer benchmarks/open-generation-v1/dev-tokenizer.nsrlbpe \
  --tokens-out "$work_dir/eval.nsrltok" \
  --trace "$work_dir/eval.trace.json"

cargo run -q -p nsrl-corpus --bin nsrl-subword -- decode \
  --tokens "$work_dir/eval.nsrltok" \
  --tokenizer benchmarks/open-generation-v1/dev-tokenizer.nsrlbpe \
  --out "$work_dir/eval.roundtrip.txt"

cmp benchmarks/integer-transformer-proof-v1/eval.txt "$work_dir/eval.roundtrip.txt"
node scripts/freeze-open-generation-v1.mjs --check
node --check scripts/evaluate-open-generation-development-v1.mjs
node --check scripts/freeze-open-generation-development-baseline-v1.mjs
node scripts/check-open-generation-development-checkpoint-v1.mjs
cargo run -q -p nsrl-eval -- open-generation-manifest \
  --manifest benchmarks/open-generation-v1/manifest.tsv > "$work_dir/manifest.json"

baseline_dir="data/experiments/open-generation-v1/p10m-kv-scaling-baseline"
if [[ -s "$baseline_dir/result.json" ]]; then
  node scripts/evaluate-open-generation-development-v1.mjs \
    --manifest benchmarks/open-generation-v1/manifest.tsv \
    --run "$baseline_dir/run.json" \
    --samples "$baseline_dir/samples.jsonl" \
    --decoder-traces "$baseline_dir/decoder-traces.jsonl" \
    --modeling "$baseline_dir/modeling.json" \
    --runner-binary target/release/nsrl-open-generation-run \
    --modeling-runner-binary target/release/nsrl-open-generation-modeling-run \
    --candidate-model data/experiments/production-model-v1/p10m-kv-scaling-readiness/integer-model-7.nsrlpm \
    --candidate-tokenizer data/processed/production-corpus-v1/tokenizer.nsrlbpe \
    --check "$baseline_dir/result.json"
  node scripts/freeze-open-generation-development-baseline-v1.mjs --check
fi

echo "open-generation-v1 tokenizer and contract checkpoint passed"
