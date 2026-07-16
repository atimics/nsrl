#!/usr/bin/env bash
set -euo pipefail

export NSRL_CAUSAL_SEQUENCE_CONTRACT="benchmarks/production-model-v1/p10m-causal-sequence-scale-v1-contract.json"
export NSRL_CAUSAL_SEQUENCE_OUT_DIR="data/experiments/production-model-v1/p10m-causal-sequence-scale-v1"
export NSRL_CAUSAL_SEQUENCE_CHECKPOINT="benchmarks/production-model-v1/p10m-causal-sequence-scale-v1.json"
export NSRL_CAUSAL_SEQUENCE_SOURCE_MODEL="data/experiments/production-model-v1/p10m-causal-sequence-preflight-v3/candidate.nsrlpm"
export NSRL_EMBEDDING_BOOST_SHIFT=1
export NSRL_CAUSAL_SEQUENCE_CONTEXT_TOKENS=64
export NSRL_CAUSAL_SEQUENCE_TARGETS_PER_WINDOW=64
export NSRL_CAUSAL_SEQUENCE_MAX_WINDOWS=512
export NSRL_CAUSAL_SEQUENCE_TRAIN_EVALUATION_WINDOWS=64
export NSRL_CAUSAL_SEQUENCE_MIDPOINT_STEPS=64
export NSRL_CAUSAL_SEQUENCE_OPTIMIZER_STEPS=128

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run-production-causal-sequence-preflight-v2.sh"
