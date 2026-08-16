#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

name="p10m-causal-tail-representation-v9-quality-postflight"
contract="benchmarks/production-model-v1/${name}-contract.json"
out_dir="data/experiments/production-model-v1/${name}"
checkpoint="benchmarks/production-model-v1/${name}.json"
open_generation_dir="benchmarks/open-generation-v1"
binary="target/release/nsrl-production-model"
model="data/experiments/production-model-v1/p10m-causal-tail-representation-v9-health-scale/chunk-3/model.nsrlpm"
tokenizer="data/processed/production-corpus-v1/tokenizer.nsrlbpe"
dev_tokens="data/processed/production-corpus-v1/dev.nsrltok"
test_tokens="data/processed/production-corpus-v1/test.nsrltok"
manifest="benchmarks/open-generation-v1/manifest.tsv"

mkdir -p "$out_dir" "$open_generation_dir"
work_dir="$(mktemp -d "$out_dir/.postflight.XXXXXX")"
cleanup() {
  local status=$?
  if ((status == 0)); then
    rm -rf "$work_dir"
  else
    echo "v9 quality postflight failed; retained work directory: $work_dir" >&2
  fi
}
trap cleanup EXIT

cargo build --release -p nsrl-train \
  --bin nsrl-production-model \
  --bin nsrl-production-rollout-divergence-audit \
  --bin nsrl-production-context-sensitivity-audit \
  --bin nsrl-production-residual-saturation-audit

echo "v9 quality postflight phase: public test confirmation"
"$binary" evaluate-canonical \
  --tokenizer "$tokenizer" --tokens "$test_tokens" \
  --model "$model" --trace "$work_dir/test.json" \
  --context-tokens 64 --max-windows 512

echo "v9 quality postflight phase: rollout, context, and numeric-health diagnostics"
target/release/nsrl-production-rollout-divergence-audit \
  --tokenizer "$tokenizer" --tokens "$dev_tokens" --model "$model" \
  --trace "$work_dir/rollout-divergence.json" \
  --context-tokens 64 --rollout-tokens 16 --max-windows 8
target/release/nsrl-production-context-sensitivity-audit \
  --manifest "$manifest" --tokenizer "$tokenizer" --model "$model" \
  --trace "$work_dir/context-sensitivity.json" --top-k 8
target/release/nsrl-production-residual-saturation-audit \
  --manifest "$manifest" --tokenizer "$tokenizer" --model "$model" \
  --trace "$work_dir/residual-saturation.json" >/dev/null

mv "$work_dir/test.json" "$out_dir/test.json"
mv "$work_dir/rollout-divergence.json" \
  "$open_generation_dir/${name}-rollout-divergence.json"
mv "$work_dir/context-sensitivity.json" \
  "$open_generation_dir/${name}-context-sensitivity.json"
mv "$work_dir/residual-saturation.json" \
  "$open_generation_dir/${name}-residual-saturation.json"

node scripts/freeze-production-representation-quality-postflight-v1.mjs \
  --contract "$contract" --run-dir "$out_dir" \
  --open-generation-dir "$open_generation_dir" --out "$checkpoint"
