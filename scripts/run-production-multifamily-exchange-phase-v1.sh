#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 4 || "$#" -gt 5 ]]; then
  echo "usage: $0 CORPUS INDEX PHASE_NAME OUTPUT_STEM [full|contract|audit]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

corpus="$1"
index="$2"
phase="$3"
output_stem="$4"
mode="${5:-full}"
if [[ "$mode" != "full" && "$mode" != "contract" && "$mode" != "audit" ]]; then
  echo "invalid mode: $mode" >&2
  exit 2
fi
tokenizer="data/processed/production-corpus-v1/tokenizer.nsrlbpe"
model="data/experiments/production-model-v1/p10m-normalized-wide-gradient-preflight/model-q23-newton-output33.nsrlpm"
processed="data/processed/production-multifamily-exchange-v1"
tokens="$processed/${phase}.nsrltok"
token_trace="$processed/${phase}.tokens.json"
contract="${output_stem}-structure-contract.json"
trace="${output_stem}-structure.json"

if [[ "$mode" != "audit" ]]; then
  target/release/nsrl-subword encode-indexed \
    --corpus "$corpus" --index "$index" --tokenizer "$tokenizer" \
    --tokens-out "$tokens" --trace "$token_trace"
fi

bindings="$(target/release/nsrl-production-model boolean-jet-protocol-bindings)"
source_fnv64="$(node -e 'const value=JSON.parse(process.argv[1]); process.stdout.write(value.source_fnv64)' "$bindings")"
binary_fnv64="$(node -e 'const value=JSON.parse(process.argv[1]); process.stdout.write(value.binary_fnv64)' "$bindings")"
token_hash="$(node -e 'const fs=require("node:fs"); const value=JSON.parse(fs.readFileSync(process.argv[1],"utf8")); process.stdout.write(value.token_hash)' "$token_trace")"

common=(
  --tokenizer "$tokenizer" --tokens "$tokens" --source-index "$index" --model "$model"
  --context-tokens 64
  --expected-base-model-hash 0xb10996e0707ab342
  --expected-tokenizer-hash 0xf4fe71d93c438c1a
  --expected-token-stream-hash "$token_hash"
  --expected-source-fnv64 "$source_fnv64"
  --expected-binary-fnv64 "$binary_fnv64"
  --trunk-move 3:40:-1 --trunk-move 3:68:-1
  --trunk-move 3:206:-1 --trunk-move 3:218:-1
  --head-move 11:1424062:1 --head-move 12:7866:-1
)

if [[ "$mode" != "audit" ]]; then
  target/release/nsrl-production-model boolean-jet-atomic-structure-contract \
    "${common[@]}" --trace "$contract"
fi
manifest_hash="$(node -e 'const fs=require("node:fs"); const value=JSON.parse(fs.readFileSync(process.argv[1],"utf8")); process.stdout.write(value.manifest_hash)' "$contract")"
if [[ "$mode" == "contract" ]]; then
  echo "froze multifamily exchange phase contract ${phase}: ${contract}"
  exit 0
fi
target/release/nsrl-production-model boolean-jet-atomic-structure-audit \
  "${common[@]}" --expected-manifest-hash "$manifest_hash" --trace "$trace"
node scripts/check-production-multisource-atomic-structure-v1.mjs "$contract" "$trace"
echo "completed multifamily exchange phase ${phase}: ${trace}"
