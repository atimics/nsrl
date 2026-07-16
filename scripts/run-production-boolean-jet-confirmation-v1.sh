#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

model="data/experiments/production-model-v1/p10m-normalized-wide-gradient-preflight/model-q23-newton-output33.nsrlpm"
tokenizer="data/processed/production-corpus-v1/tokenizer.nsrlbpe"
tokens="data/processed/production-corpus-v1/dev.nsrltok"
trace="${NSRL_PRODUCTION_BOOLEAN_JET_CONFIRMATION_TRACE:-benchmarks/production-model-v1/p10m-boolean-jet-confirmation-v1.json}"

cargo build --release -p nsrl-train --bin nsrl-production-model
target/release/nsrl-production-model boolean-jet-confirmation-audit \
  --tokenizer "$tokenizer" --tokens "$tokens" --model "$model" --trace "$trace" \
  --context-tokens 64 \
  --expected-base-model-hash 0xb10996e0707ab342 \
  --expected-tokenizer-hash 0xf4fe71d93c438c1a \
  --expected-token-stream-hash 0xda195778ceb603ab \
  --expected-move-fingerprint 0xc11353911a5130fb \
  --expected-manifest-hash 0x263f6984eeccfa84 \
  --trunk-move 3:40:-1 --trunk-move 3:68:-1 \
  --trunk-move 3:206:-1 --trunk-move 3:218:-1 \
  --head-move 11:1424062:1 --head-move 12:7866:-1 \
  --proposal-document-start 8 --proposal-documents 64 \
  --transfer-document-start 72 --transfer-documents 64 \
  --windows-per-document 2 --minimum-documents 32

echo "production p10m Boolean-jet confirmation completed: $trace"
