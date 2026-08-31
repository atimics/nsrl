#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

./scripts/check-no-floats.sh
node scripts/check-model-launch-v1.mjs
node scripts/build-model-launch-site.mjs --check
node scripts/check-model-localnet-v1.mjs
node scripts/build-model-localnet-site.mjs --check
node scripts/check-model-market-v1.mjs
node scripts/build-model-market-site.mjs --check
node scripts/check-bounty-automation-v1.mjs
node scripts/build-bounty-automation-site.mjs --check
node scripts/check-integer-transformer-proof-self-test.mjs
node scripts/check-integer-transformer-candidate-health-self-test.mjs
node scripts/check-harmonic-structure-theory-v1.mjs
node scripts/check-document-ising-theory-v1.mjs
node scripts/check-production-atomic-structure-v1.mjs
node scripts/check-production-atomic-ising-v1.mjs
node scripts/check-production-document-ising-proposal-v1.mjs
node scripts/check-production-atomic-structure-v1.mjs \
  benchmarks/production-model-v1/p10m-atomic-structure-confirmation-v1-contract.json \
  benchmarks/production-model-v1/p10m-atomic-structure-confirmation-v1.json
node scripts/check-production-atomic-ising-confirmation-execution-v1.mjs
node scripts/check-production-atomic-ising-confirmation-v1.mjs
node scripts/analyze-production-atomic-harmonics-v1.mjs \
  benchmarks/production-model-v1/p10m-atomic-structure-proposal-v1.json \
  /tmp/nsrl-p10m-atomic-harmonics-proposal-v1-check.json
cmp benchmarks/production-model-v1/p10m-atomic-harmonics-proposal-v1.json \
  /tmp/nsrl-p10m-atomic-harmonics-proposal-v1-check.json
node scripts/analyze-production-conditional-exchange-v1.mjs \
  benchmarks/production-model-v1/p10m-atomic-structure-proposal-v1.json \
  benchmarks/production-model-v1/p10m-atomic-structure-confirmation-v1.json \
  benchmarks/production-model-v1/p10m-atomic-ising-confirmation-v1.json \
  /tmp/nsrl-p10m-atomic-conditional-exchange-confirmation-v1-check.json
cmp benchmarks/production-model-v1/p10m-atomic-conditional-exchange-confirmation-v1.json \
  /tmp/nsrl-p10m-atomic-conditional-exchange-confirmation-v1-check.json
node scripts/check-production-training-liveness-self-test.mjs
node scripts/check-production-optimizer-residual-analysis-self-test.mjs
node scripts/freeze-integer-transformer-proof-candidate.mjs --check
scripts/check-open-generation-v1.sh
node scripts/run-q22-solomon-prospective.mjs --check-contract
node scripts/check-q22-solomon-evidence.mjs
node scripts/run-q22-compositional-solomon-prospective.mjs --check-contract
scripts/check-production-corpus-v1.sh
node scripts/freeze-production-model-v1.mjs --check
node scripts/freeze-production-full-train-v1.mjs --check
node scripts/freeze-production-float-twin-v1.mjs --check
node scripts/freeze-production-integer-stabilization-v1.mjs --check
node scripts/freeze-production-stabilized-pilot-v1.mjs --check
node scripts/freeze-production-liveness-audit-v1.mjs --check
node scripts/freeze-production-trunk-unlock-preflight-v1.mjs --check
node scripts/check-production-model-v1.mjs
node scripts/check-production-optimization-v1.mjs
cargo fmt --all --check
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets
git diff --check
git diff --cached --check
