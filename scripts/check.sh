#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

./scripts/check-no-floats.sh
node scripts/check-integer-transformer-proof-self-test.mjs
node scripts/check-integer-transformer-candidate-health-self-test.mjs
node scripts/freeze-integer-transformer-proof-candidate.mjs --check
scripts/check-open-generation-v1.sh
scripts/check-production-corpus-v1.sh
node scripts/check-production-model-v1.mjs
cargo fmt --all --check
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets
git diff --check
git diff --cached --check
