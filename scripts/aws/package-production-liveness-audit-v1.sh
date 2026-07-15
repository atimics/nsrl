#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
out="${1:-/tmp/nsrl-production-liveness-audit-v1.tar.gz}"
mkdir -p "$(dirname "$out")"
tar -czf "$out" \
  Cargo.toml Cargo.lock crates \
  scripts/check-production-training-liveness-v1.mjs \
  scripts/run-production-liveness-audit-v1.sh \
  scripts/freeze-production-liveness-audit-v1.mjs \
  scripts/aws/run-production-liveness-audit-v1-graviton.sh \
  data/processed/production-corpus-v1/tokenizer.nsrlbpe \
  data/processed/production-corpus-v1/train.nsrltok \
  data/processed/production-corpus-v1/dev.nsrltok
echo "$out"
