#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

model="data/experiments/production-model-v1/p10m-normalized-wide-gradient-preflight/model-q23-newton-output33.nsrlpm"
tokenizer="data/processed/production-corpus-v1/tokenizer.nsrlbpe"
tokens="data/processed/production-corpus-v1/dev.nsrltok"
source_index="data/processed/production-corpus-v1/dev.index.tsv"
contract="${NSRL_PRODUCTION_ATOMIC_STRUCTURE_CONTRACT:-benchmarks/production-model-v1/p10m-atomic-structure-proposal-v1-contract.json}"
trace="${NSRL_PRODUCTION_ATOMIC_STRUCTURE_TRACE:-benchmarks/production-model-v1/p10m-atomic-structure-proposal-v1.json}"

cargo build --release -p nsrl-train --bin nsrl-production-model
bindings="$(target/release/nsrl-production-model boolean-jet-protocol-bindings)"
source_fnv64="$(node -e 'const value=JSON.parse(process.argv[1]); process.stdout.write(value.source_fnv64)' "$bindings")"
binary_fnv64="$(node -e 'const value=JSON.parse(process.argv[1]); process.stdout.write(value.binary_fnv64)' "$bindings")"

common=(
  --tokenizer "$tokenizer" --tokens "$tokens" --source-index "$source_index" --model "$model"
  --context-tokens 64
  --expected-base-model-hash 0xb10996e0707ab342
  --expected-tokenizer-hash 0xf4fe71d93c438c1a
  --expected-token-stream-hash 0xda195778ceb603ab
  --expected-source-fnv64 "$source_fnv64"
  --expected-binary-fnv64 "$binary_fnv64"
  --trunk-move 3:40:-1 --trunk-move 3:68:-1
  --trunk-move 3:206:-1 --trunk-move 3:218:-1
  --head-move 11:1424062:1 --head-move 12:7866:-1
)

target/release/nsrl-production-model boolean-jet-atomic-structure-contract \
  "${common[@]}" --trace "$contract"
manifest_hash="$(node -e 'const fs=require("node:fs"); const value=JSON.parse(fs.readFileSync(process.argv[1],"utf8")); process.stdout.write(value.manifest_hash)' "$contract")"

target/release/nsrl-production-model boolean-jet-atomic-structure-audit \
  "${common[@]}" --expected-manifest-hash "$manifest_hash" --trace "$trace"

node scripts/check-production-atomic-structure-v1.mjs "$contract" "$trace"
echo "production p10m proposal-only atomic structure audit completed: $trace"
