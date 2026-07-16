#!/usr/bin/env bash
set -euo pipefail

export NSRL_CAUSAL_SEQUENCE_CONTRACT="benchmarks/production-model-v1/p10m-causal-sequence-preflight-v3-contract.json"
export NSRL_CAUSAL_SEQUENCE_OUT_DIR="data/experiments/production-model-v1/p10m-causal-sequence-preflight-v3"
export NSRL_CAUSAL_SEQUENCE_CHECKPOINT="benchmarks/production-model-v1/p10m-causal-sequence-preflight-v3.json"
export NSRL_EMBEDDING_BOOST_SHIFT=1

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run-production-causal-sequence-preflight-v2.sh"
