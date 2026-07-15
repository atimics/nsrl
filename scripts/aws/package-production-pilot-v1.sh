#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

out="${1:-/tmp/nsrl-production-pilot-v1.tar.gz}"
mkdir -p "$(dirname "$out")"
tar -czf "$out" \
  Cargo.toml Cargo.lock crates \
  scripts/production-float-twin-v1.py \
  scripts/run-production-pilot-v1.sh \
  scripts/freeze-production-pilot-v1.mjs \
  scripts/aws/run-production-pilot-v1-graviton.sh \
  benchmarks/production-model-v1/p10m-pilot-contract.json \
  data/processed/production-corpus-v1/tokenizer.nsrlbpe \
  data/processed/production-corpus-v1/train.nsrltok \
  data/processed/production-corpus-v1/dev.nsrltok \
  data/experiments/production-model-v1/p10m-smoke/initial.nsrlpm
echo "$out"
