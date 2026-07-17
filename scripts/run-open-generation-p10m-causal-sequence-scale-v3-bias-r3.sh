#!/usr/bin/env bash
set -euo pipefail

export NSRL_OPEN_GENERATION_OUT_DIR="data/experiments/open-generation-v1/p10m-causal-sequence-scale-v3-bias-r3"
export NSRL_OPEN_GENERATION_MODEL="data/experiments/production-model-v1/p10m-causal-sequence-scale-v3-bias-r3/candidate.nsrlpm"
export NSRL_OPEN_GENERATION_CHECKPOINT_OUT="benchmarks/open-generation-v1/p10m-causal-sequence-scale-v3-bias-r3.json"

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run-open-generation-development-v1.sh"
