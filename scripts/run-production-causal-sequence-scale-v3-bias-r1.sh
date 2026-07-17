#!/usr/bin/env bash
set -euo pipefail

export NSRL_CAUSAL_SEQUENCE_CONTRACT="benchmarks/production-model-v1/p10m-causal-sequence-scale-v3-bias-r1-contract.json"
export NSRL_CAUSAL_SEQUENCE_OUT_DIR="data/experiments/production-model-v1/p10m-causal-sequence-scale-v3-bias-r1"
export NSRL_CAUSAL_SEQUENCE_CHECKPOINT="benchmarks/production-model-v1/p10m-causal-sequence-scale-v3-bias-r1.json"
export NSRL_CAUSAL_SEQUENCE_SOURCE_MODEL="data/experiments/production-model-v1/p10m-causal-sequence-scale-v2/candidate.nsrlpm"
export NSRL_EMBEDDING_BOOST_SHIFT=2
export NSRL_K_LEARNING_RATE_SHIFT=21
export NSRL_UP_LEARNING_RATE_SHIFT=15
export NSRL_GATE_LEARNING_RATE_SHIFT=15
export NSRL_OUTPUT_BIAS_LEARNING_RATE_SHIFT=13
export NSRL_CAUSAL_SEQUENCE_CONTEXT_TOKENS=64
export NSRL_CAUSAL_SEQUENCE_TARGETS_PER_WINDOW=64
export NSRL_CAUSAL_SEQUENCE_TRAINING_WORKERS=8
export NSRL_CAUSAL_SEQUENCE_MAX_WINDOWS=2048
export NSRL_CAUSAL_SEQUENCE_TRAIN_EVALUATION_WINDOWS=64
export NSRL_CAUSAL_SEQUENCE_MIDPOINT_STEPS=256
export NSRL_CAUSAL_SEQUENCE_OPTIMIZER_STEPS=512

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run-production-causal-sequence-preflight-v2.sh"
