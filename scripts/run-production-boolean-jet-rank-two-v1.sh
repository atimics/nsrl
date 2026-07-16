#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

model="data/experiments/production-model-v1/p10m-normalized-wide-gradient-preflight/model-q23-newton-output33.nsrlpm"
tokenizer="data/processed/production-corpus-v1/tokenizer.nsrlbpe"
tokens="data/processed/production-corpus-v1/dev.nsrltok"
trace="${NSRL_PRODUCTION_BOOLEAN_JET_TRACE:-benchmarks/production-model-v1/p10m-boolean-jet-rank-two-v1.json}"

cargo build --release -p nsrl-train --bin nsrl-production-model
target/release/nsrl-production-model boolean-jet-rank-two-audit \
  --tokenizer "$tokenizer" --tokens "$tokens" --model "$model" --trace "$trace" \
  --context-tokens 64 --max-windows 8 --transfer-windows 8 \
  --documents-per-surface 4 --rescue-stratified-sampling \
  --include-mass-corrected-no-rescue \
  --coordinates-per-group 1 --seed 43 \
  --expected-trunk-moves 4 --expected-head-moves 2 \
  --expected-move-fingerprint 0xc11353911a5130fb \
  --matrix-learning-rate-shift 25 \
  --q-learning-rate-shift 29 --k-learning-rate-shift 26 \
  --v-learning-rate-shift 30 --o-learning-rate-shift 25 \
  --up-learning-rate-shift 22 --gate-learning-rate-shift 23 \
  --down-learning-rate-shift 25 --vector-learning-rate-shift 23 \
  --embedding-learning-rate-shift 17 \
  --output-learning-rate-shift 33 --output-backward-shift 8 \
  --probability-gradient-fractional-bits 23 \
  --probability-normalization q47-newton1

echo "production p10m rank-two Boolean-jet audit completed: $trace"
