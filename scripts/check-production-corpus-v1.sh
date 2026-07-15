#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

node scripts/check-production-corpus-v1-self-test.mjs
node scripts/freeze-production-corpus-v1.mjs --check
cargo test -q -p nsrl-corpus

echo "production-corpus-v1 checkpoint passed"
