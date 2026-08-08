#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

name="p10m-causal-tail-representation-v5-group-composition-audit"
contract="benchmarks/production-model-v1/${name}-contract.json"
out_dir="data/experiments/production-model-v1/${name}"
audit="$out_dir/audit.json"
checkpoint="benchmarks/production-model-v1/${name}.json"
source="data/experiments/production-model-v1/p10m-causal-tail-full-v1/candidate.nsrlpm"
candidate="data/experiments/production-model-v1/p10m-causal-tail-representation-v5-stability/candidate.nsrlpm"
tokenizer="data/processed/production-corpus-v1/tokenizer.nsrlbpe"
dev_tokens="data/processed/production-corpus-v1/dev.nsrltok"

mkdir -p "$out_dir"
cargo build --release -p nsrl-train --bin nsrl-production-group-composition-audit

target/release/nsrl-production-group-composition-audit \
  --tokenizer "$tokenizer" --tokens "$dev_tokens" \
  --source "$source" --candidate "$candidate" \
  --context-tokens 64 --max-windows 512 \
  --trace "$audit" >/dev/null

node scripts/freeze-production-group-composition-audit-v1.mjs \
  --contract "$contract" --audit "$audit" --out "$checkpoint"
