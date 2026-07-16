#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_dir="${1:-data/experiments/integer-transformer-proof-v1/component-ablation}"
model="${NSRL_PROOF_ABLATION_MODEL:-data/experiments/integer-transformer-proof-v1/candidate-default/candidate.nsrlmt}"
tokens="${NSRL_PROOF_ABLATION_TOKENS:-benchmarks/integer-transformer-proof-v1/eval.txt}"
target_dir="${NSRL_PROOF_ABLATION_TARGET_DIR:-target/integer-transformer-proof-h8}"

mkdir -p "$out_dir"
CARGO_TARGET_DIR="$target_dir" cargo build --release -p nsrl-train \
  --bin nsrl-mini-transformer-eval --features mini-heads-8,mini-calibrated

for mode in combined transformer-only suffix-memory-only; do
  "$target_dir/release/nsrl-mini-transformer-eval" \
    --tokens "$tokens" \
    --model "$model" \
    --stride 1 \
    --attention linear \
    --position nope \
    --ablation "$mode" \
    --out "$out_dir/$mode.eval.json"
done

node scripts/summarize-integer-transformer-proof-ablation.mjs \
  --combined "$out_dir/combined.eval.json" \
  --transformer-only "$out_dir/transformer-only.eval.json" \
  --suffix-memory-only "$out_dir/suffix-memory-only.eval.json" \
  --out "$out_dir/report.json"
node scripts/summarize-integer-transformer-proof-ablation.mjs \
  --combined "$out_dir/combined.eval.json" \
  --transformer-only "$out_dir/transformer-only.eval.json" \
  --suffix-memory-only "$out_dir/suffix-memory-only.eval.json" \
  --out benchmarks/integer-transformer-proof-v1/component-ablation.json

echo "integer-transformer component ablation written to $out_dir/report.json"
