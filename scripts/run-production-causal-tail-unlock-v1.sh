#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

name="p10m-causal-tail-unlock-v1"
export NSRL_CAUSAL_SEQUENCE_CONTRACT="benchmarks/production-model-v1/${name}-contract.json"
export NSRL_CAUSAL_SEQUENCE_OUT_DIR="data/experiments/production-model-v1/${name}"
export NSRL_CAUSAL_SEQUENCE_CHECKPOINT="benchmarks/production-model-v1/${name}.json"
export NSRL_CAUSAL_SEQUENCE_SOURCE_MODEL="data/experiments/production-model-v1/p10m-causal-sequence-scale-v2/candidate.nsrlpm"
export NSRL_EMBEDDING_BOOST_SHIFT=2
export NSRL_EMBEDDING_LEARNING_RATE_SHIFT=4
export NSRL_VECTOR_LEARNING_RATE_SHIFT=12
export NSRL_FINAL_RMS_LEARNING_RATE_SHIFT=11
export NSRL_Q_LEARNING_RATE_SHIFT=19
export NSRL_K_LEARNING_RATE_SHIFT=25
export NSRL_V_LEARNING_RATE_SHIFT=27
export NSRL_O_LEARNING_RATE_SHIFT=16
export NSRL_UP_LEARNING_RATE_SHIFT=18
export NSRL_GATE_LEARNING_RATE_SHIFT=18
export NSRL_DOWN_LEARNING_RATE_SHIFT=7
export NSRL_OUTPUT_LEARNING_RATE_SHIFT=33
export NSRL_OUTPUT_BIAS_LEARNING_RATE_SHIFT=13
export NSRL_CAUSAL_SEQUENCE_CONTEXT_TOKENS=64
export NSRL_CAUSAL_SEQUENCE_TARGETS_PER_WINDOW=8
export NSRL_CAUSAL_SEQUENCE_TRAINING_WORKERS=8
export NSRL_CAUSAL_SEQUENCE_MAX_WINDOWS=2048
export NSRL_CAUSAL_SEQUENCE_TRAIN_EVALUATION_WINDOWS=64
export NSRL_CAUSAL_SEQUENCE_MIDPOINT_STEPS=256
export NSRL_CAUSAL_SEQUENCE_OPTIMIZER_STEPS=512
export NSRL_CAUSAL_SEQUENCE_PARALLEL_REPLAY=1

scripts/run-production-causal-sequence-preflight-v2.sh

cargo build --release -p nsrl-train \
  --bin nsrl-production-rollout-divergence-audit \
  --bin nsrl-production-context-sensitivity-audit \
  --bin nsrl-production-residual-saturation-audit

model="data/experiments/production-model-v1/${name}/candidate.nsrlpm"
tokenizer="data/processed/production-corpus-v1/tokenizer.nsrlbpe"
rollout="benchmarks/open-generation-v1/${name}-rollout-divergence.json"
context="benchmarks/open-generation-v1/${name}-context-sensitivity.json"
saturation="benchmarks/open-generation-v1/${name}-residual-saturation.json"
target/release/nsrl-production-rollout-divergence-audit \
  --tokenizer "$tokenizer" \
  --tokens data/processed/production-corpus-v1/dev.nsrltok \
  --model "$model" --trace "$rollout" \
  --context-tokens 64 --rollout-tokens 16 --max-windows 8
target/release/nsrl-production-context-sensitivity-audit \
  --manifest benchmarks/open-generation-v1/manifest.tsv \
  --tokenizer "$tokenizer" --model "$model" --trace "$context" --top-k 8
target/release/nsrl-production-residual-saturation-audit \
  --manifest benchmarks/open-generation-v1/manifest.tsv \
  --tokenizer "$tokenizer" --model "$model" --trace "$saturation"

node scripts/freeze-production-causal-tail-context-v1.mjs \
  --contract "benchmarks/production-model-v1/${name}-contract.json" \
  --preflight "benchmarks/production-model-v1/${name}.json" \
  --rollout "$rollout" --context "$context" --saturation "$saturation" \
  --out "benchmarks/production-model-v1/${name}-quality-gate.json"
