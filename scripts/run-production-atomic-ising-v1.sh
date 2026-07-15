#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

source_result="${NSRL_PRODUCTION_ATOMIC_STRUCTURE_TRACE:-benchmarks/production-model-v1/p10m-atomic-structure-proposal-v1.json}"
source_contract="${NSRL_PRODUCTION_ATOMIC_STRUCTURE_CONTRACT:-benchmarks/production-model-v1/p10m-atomic-structure-proposal-v1-contract.json}"
audit_contract="${NSRL_PRODUCTION_ATOMIC_ISING_CONTRACT:-benchmarks/production-model-v1/p10m-atomic-ising-audit-v1-contract.json}"
audit_result="${NSRL_PRODUCTION_ATOMIC_ISING_TRACE:-benchmarks/production-model-v1/p10m-atomic-ising-audit-v1.json}"

node scripts/analyze-production-atomic-ising-v1.mjs \
  "$source_result" "$audit_contract" "$audit_result"
node scripts/check-production-atomic-ising-v1.mjs \
  "$source_result" "$source_contract" "$audit_contract" "$audit_result"

echo "production p10m deterministic Ising audit replayed: $audit_result"
