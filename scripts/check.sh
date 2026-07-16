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
node scripts/run-integer-transformer-successor-v2.mjs --check
node scripts/check-integer-research-evidence-v1.mjs
node scripts/check-boolean-jet-theory-v1.mjs
node scripts/check-boolean-jet-stability-theory-v1.mjs
node scripts/check-objective-boundary-phase-theory-v1.mjs
node scripts/check-discrete-structure-theory-v1.mjs
node scripts/check-harmonic-structure-theory-v1.mjs
node scripts/check-document-ising-theory-v1.mjs
node scripts/check-conformal-exchange-theory-v1.mjs
node scripts/check-production-boolean-jet-confirmation-v1.mjs
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
node scripts/analyze-production-conformal-exchange-v1.mjs \
  benchmarks/production-model-v1/p10m-atomic-structure-proposal-v1.json \
  benchmarks/production-model-v1/p10m-atomic-structure-confirmation-v1.json \
  /tmp/nsrl-p10m-atomic-conformal-exchange-retrospective-v1-check.json
cmp benchmarks/production-model-v1/p10m-atomic-conformal-exchange-retrospective-v1.json \
  /tmp/nsrl-p10m-atomic-conformal-exchange-retrospective-v1-check.json
node scripts/analyze-production-cross-source-exchange-v1.mjs \
  benchmarks/production-model-v1/p10m-cross-source-exchange-v1-contract.json \
  benchmarks/production-model-v1/p10m-cross-source-exchange-v1-calibration-evaluation-structure.json \
  /tmp/nsrl-p10m-cross-source-exchange-v1-result-check.json
cmp benchmarks/production-model-v1/p10m-cross-source-exchange-v1-result.json \
  /tmp/nsrl-p10m-cross-source-exchange-v1-result-check.json
node scripts/check-production-cross-source-exchange-v1.mjs
node scripts/publish-production-cross-source-exchange-v1.mjs \
  benchmarks/production-model-v1/p10m-cross-source-exchange-v1-result.json \
  benchmarks/production-model-v1/p10m-cross-source-exchange-v1-contract.json \
  /tmp/nsrl-p10m-cross-source-exchange-v1-publication-check.json
cmp benchmarks/production-model-v1/p10m-cross-source-exchange-v1-publication.json \
  /tmp/nsrl-p10m-cross-source-exchange-v1-publication-check.json
node scripts/check-production-cross-source-exchange-publication-v1.mjs
node scripts/check-production-multisource-atomic-structure-v1.mjs \
  benchmarks/production-model-v1/p10m-multifamily-exchange-v1-fitting-structure-contract.json \
  benchmarks/production-model-v1/p10m-multifamily-exchange-v1-fitting-structure.json
for passage in 0 1 2 3; do
  for shard in 0 1; do
    node scripts/check-production-multisource-atomic-structure-v1.mjs \
      "benchmarks/production-model-v1/p10m-multifamily-exchange-v1-confirmation-passage-${passage}-shard-${shard}-structure-contract.json" \
      "benchmarks/production-model-v1/p10m-multifamily-exchange-v1-confirmation-passage-${passage}-shard-${shard}-structure.json"
  done
done
node scripts/analyze-production-multifamily-exchange-v1.mjs \
  benchmarks/production-model-v1/p10m-multifamily-exchange-v1-contract.json \
  /tmp/nsrl-p10m-multifamily-exchange-v1-result-check.json
cmp benchmarks/production-model-v1/p10m-multifamily-exchange-v1-result.json \
  /tmp/nsrl-p10m-multifamily-exchange-v1-result-check.json
node scripts/check-production-multifamily-exchange-v1.mjs
node scripts/check-research-harness-v1.mjs
node scripts/check-production-training-liveness-self-test.mjs
node scripts/check-production-optimizer-residual-analysis-self-test.mjs
node scripts/freeze-integer-transformer-proof-candidate.mjs --check
scripts/check-open-generation-v1.sh
scripts/check-production-corpus-v1.sh
node scripts/freeze-production-model-v1.mjs --check
node scripts/freeze-production-full-train-v1.mjs --check
node scripts/freeze-production-float-twin-v1.mjs --check
node scripts/freeze-production-integer-stabilization-v1.mjs --check
node scripts/freeze-production-stabilized-pilot-v1.mjs --check
node scripts/freeze-production-liveness-audit-v1.mjs --check
node scripts/freeze-production-trunk-unlock-preflight-v1.mjs --check
node scripts/freeze-production-k-stabilization-preflight-v1.mjs --check
node scripts/freeze-production-kv-boundary-pilot-v1.mjs --check
node scripts/freeze-production-kv-scaling-readiness-v1.mjs --check
node scripts/freeze-production-gate-boundary-preflight-v1.mjs --check
node scripts/freeze-production-up-useful-update-v1.mjs --check
node scripts/freeze-production-up-shift22-breakthrough-v1.mjs --check
node scripts/freeze-production-up-functional-comparison-v1.mjs --check
node scripts/freeze-production-up-forward-scale-sensitivity-v1.mjs --check
node scripts/freeze-production-up-forward-scale-training-v1.mjs --check
node scripts/freeze-production-target-probability-resolution-v1.mjs --check
node scripts/freeze-production-wide-probability-gradient-preflight-v1.mjs --check
node scripts/freeze-production-probability-normalization-accuracy-v1.mjs --check
node scripts/freeze-production-probability-normalization-signal-attribution-v1.mjs --check
node scripts/freeze-production-normalized-wide-gradient-preflight-v1.mjs --check
node scripts/check-production-model-v1.mjs
node scripts/check-production-optimization-v1.mjs
cargo fmt --all --check
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets
git diff --check
git diff --cached --check
