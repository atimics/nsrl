#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

name="p10m-direct-head-cross-document-stability-v1"
contract="benchmarks/production-model-v1/${name}-contract.json"
gate="benchmarks/production-model-v1/${name}-gate.json"
out_dir="data/experiments/production-model-v1/${name}"
artifact_root="${NSRL_ARTIFACT_ROOT:-$repo_root}"
source_model="$artifact_root/data/experiments/production-model-v1/p10m-causal-tail-representation-v9-health-scale/chunk-3/model.nsrlpm"
tokenizer="$artifact_root/data/processed/production-corpus-v1/tokenizer.nsrlbpe"
train_tokens="$artifact_root/data/processed/production-corpus-v1/train.nsrltok"
binary="target/release/nsrl-production-model"

verify_sha256() {
  local file="$1"
  local expected="$2"
  local observed
  observed="$(/usr/bin/shasum -a 256 "$file" | awk '{print $1}')"
  if [[ "$observed" != "$expected" ]]; then
    echo "SHA-256 mismatch: $file" >&2
    return 1
  fi
}

verify_sha256 "$source_model" "14f568de85931696dfd2c7b4cb35883d7b8c88430e5395b0c9c7f9f2660d5c22"
verify_sha256 "$tokenizer" "9a9f96e4b7114726966ce0c2f5a0969939900e28f50860749fc1d1ebc31a25ce"
verify_sha256 "$train_tokens" "08b759945cfbbbcd15e65a2538d7a34040c8a5e7346cb19f995be05b06ad24b8"

mkdir -p "$out_dir"
cargo build --release -p nsrl-train --bin nsrl-production-model

run_audit() {
  local trace="$1"
  "$binary" direct-head-cross-document-audit \
    --tokenizer "$tokenizer" --tokens "$train_tokens" \
    --model "$source_model" --trace "$trace" \
    --context-tokens 64 \
    --direct-head-document-start 2 \
    --direct-head-documents 8 \
    --direct-head-windows-per-document 32 \
    --direct-head-coordinate-direction 8303:1 \
    --direct-head-coordinate-direction 8310:-1 \
    --direct-head-coordinate-direction 8445:1 \
    --direct-head-coordinate-direction 8335:-1 \
    --direct-head-coordinate-direction 8428:1 \
    --direct-head-coordinate-direction 8377:1 \
    --direct-head-coordinate-direction 8263:1 \
    --direct-head-coordinate-direction 8431:-1
}

echo "cross-document stability: exact frozen-direction audit"
run_audit "$out_dir/audit.json"
echo "cross-document stability: exact full rerun replay"
run_audit "$out_dir/replay-audit.json"
echo "cross-document stability: prospective stop/go gate"
node scripts/check-production-direct-head-cross-document-stability-v1.mjs \
  --contract "$contract" \
  --audit "$out_dir/audit.json" \
  --replay-audit "$out_dir/replay-audit.json" \
  --out "$gate"
