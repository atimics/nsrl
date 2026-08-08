#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

name="p10m-causal-tail-representation-v3-late-stochastic-trigger-audit"
contract="benchmarks/production-model-v1/${name}-contract.json"
out_dir="data/experiments/production-model-v1/${name}"
audit="$out_dir/audit.json"
checkpoint="benchmarks/production-model-v1/${name}.json"
tokenizer="data/processed/production-corpus-v1/tokenizer.nsrlbpe"
train_tokens="data/processed/production-corpus-v1/train.nsrltok"
model="data/experiments/production-model-v1/p10m-causal-tail-representation-v2-stability-localization-r4/model-step-429.nsrlpm"
optimizer="data/experiments/production-model-v1/p10m-causal-tail-representation-v2-stability-localization-r4/optimizer-step-429.nsrlpo"

mkdir -p "$out_dir"
cargo build --release -p nsrl-train --bin nsrl-production-saturation-backoff-audit

target/release/nsrl-production-saturation-backoff-audit \
  --tokenizer "$tokenizer" --tokens "$train_tokens" \
  --model "$model" --optimizer-state "$optimizer" \
  --output-backward-shifts 9 \
  --backward-quantization late-stochastic \
  --backward-stochastic-seed 29 \
  --trace "$audit" >/dev/null

node scripts/freeze-production-backward-quantization-trigger-audit-v1.mjs \
  --contract "$contract" --audit "$audit" --out "$checkpoint"
