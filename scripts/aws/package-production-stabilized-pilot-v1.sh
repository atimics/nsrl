#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

out="${1:-/tmp/nsrl-production-stabilized-pilot-v1.tar.gz}"
mkdir -p "$(dirname "$out")"
tar -czf "$out" \
  Cargo.toml Cargo.lock crates \
  scripts/production-float-twin-v1.py \
  scripts/check-production-stabilized-pilot-chunk-v1.mjs \
  scripts/run-production-stabilized-pilot-v1.sh \
  scripts/freeze-production-stabilized-pilot-v1.mjs \
  scripts/aws/run-production-stabilized-pilot-v1-graviton.sh \
  benchmarks/production-model-v1/p10m-stabilization.json \
  benchmarks/production-model-v1/p10m-stabilized-pilot-attempt-1.json \
  benchmarks/production-model-v1/p10m-stabilized-pilot-contract-v2.json \
  data/processed/production-corpus-v1/tokenizer.nsrlbpe \
  data/processed/production-corpus-v1/train.nsrltok \
  data/processed/production-corpus-v1/dev.nsrltok
echo "$out"
