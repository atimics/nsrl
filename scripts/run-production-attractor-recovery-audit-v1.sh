#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

verify_binding() {
  local path="$1"
  local expected_bytes="$2"
  local expected_sha256="$3"
  local actual_bytes
  local actual_sha256
  actual_bytes="$(wc -c < "$path" | tr -d ' ')"
  actual_sha256="$(shasum -a 256 "$path" | awk '{print $1}')"
  if [[ "$actual_bytes" != "$expected_bytes" || "$actual_sha256" != "$expected_sha256" ]]; then
    echo "binding mismatch: $path" >&2
    exit 1
  fi
}

verify_binding \
  data/experiments/production-model-v1/p10m-causal-tail-representation-v9-health-scale/chunk-3/model.nsrlpm \
  13539906 14f568de85931696dfd2c7b4cb35883d7b8c88430e5395b0c9c7f9f2660d5c22
verify_binding \
  data/experiments/production-model-v1/p10m-causal-tail-output-calibration-v10/candidate.nsrlpm \
  13539906 d1e323aa94170b3d0d049c1c70be2d5bebdd1cc7f57e5fc549da13ecdbb5f1be
verify_binding data/processed/production-corpus-v1/tokenizer.nsrlbpe \
  63496 9a9f96e4b7114726966ce0c2f5a0969939900e28f50860749fc1d1ebc31a25ce
verify_binding data/processed/production-corpus-v1/dev.nsrltok \
  660608 8b02253e619f047cb3cb30bf42069fa511f3367d57ed3bb697783fc3257e37b8
verify_binding benchmarks/open-generation-v1/manifest.tsv \
  658 74f176202c7483ddfa7330325cf34833358e5ef8393ebdd8bb3fa0444f0fd948

cargo build --release -p nsrl-train --bin nsrl-production-attractor-recovery-audit

target/release/nsrl-production-attractor-recovery-audit \
  --manifest benchmarks/open-generation-v1/manifest.tsv \
  --tokenizer data/processed/production-corpus-v1/tokenizer.nsrlbpe \
  --tokens data/processed/production-corpus-v1/dev.nsrltok \
  --source-model \
    data/experiments/production-model-v1/p10m-causal-tail-representation-v9-health-scale/chunk-3/model.nsrlpm \
  --candidate-model \
    data/experiments/production-model-v1/p10m-causal-tail-output-calibration-v10/candidate.nsrlpm \
  --trace benchmarks/production-model-v1/p10m-attractor-recovery-audit-v1.json \
  --context-tokens 64 \
  --rollout-tokens 16 \
  --max-windows 8
