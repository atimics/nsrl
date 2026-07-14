#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

./scripts/check-no-floats.sh
node scripts/check-model-launch-v1.mjs
node scripts/build-model-launch-site.mjs --check
node scripts/check-model-localnet-v1.mjs
node scripts/build-model-localnet-site.mjs --check
node scripts/check-integer-transformer-proof-self-test.mjs
node scripts/check-integer-transformer-candidate-health-self-test.mjs
node scripts/freeze-integer-transformer-proof-candidate.mjs --check
scripts/check-open-generation-v1.sh
cargo fmt --all --check
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets
git diff --check
git diff --cached --check
