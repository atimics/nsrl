#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

raw="data/raw/p10m-adaptive-composition-v1"
processed="data/processed/p10m-adaptive-composition-v1"
experiment="data/experiments/production-model-v1/p10m-adaptive-composition-v1"
manifest="$experiment/manifest"
execution="$experiment/execution"
replay="/tmp/nsrl-adaptive-composition-v1-replay"
prereg="protocol/examples/p10m-adaptive-composition-v1-preregistration.json"
frame="benchmarks/production-model-v1/p10m-adaptive-composition-v1-source-frame.json"
contract="benchmarks/production-model-v1/p10m-adaptive-composition-v1-contract.json"
precontract="benchmarks/production-model-v1/p10m-adaptive-composition-v1-precalibration-contract.json"
result="benchmarks/production-model-v1/p10m-adaptive-composition-v1-result.json"
model="data/experiments/production-model-v1/p10m-normalized-wide-gradient-preflight/model-q23-newton-output33.nsrlpm"
tokenizer="data/processed/production-corpus-v1/tokenizer.nsrlbpe"
exclusions="benchmarks/production-model-v1/p10m-multifamily-exchange-v1-source-frame.json,benchmarks/production-model-v1/p10m-solomonic-judgment-v1-source-frame.json"

node scripts/acquire-production-multifamily-exchange-v1.mjs \
  "$raw" 152 federal_register,rfc,science "$exclusions" \
  nsrl-m5-adaptive-composition-acquisition-2026-07-15-v1 whole_publication
node scripts/prepare-adaptive-composition-v1.mjs \
  "$raw/acquisition.json" "$processed" "$frame" "$exclusions"

cargo build --release -p nsrl-corpus --bin nsrl-subword
cargo build --release -p nsrl-train --bin nsrl-adaptive-composition

for role in fitting calibration adaptive endpoint; do
  target/release/nsrl-subword encode-indexed \
    --corpus "$processed/$role.txt" \
    --index "$processed/$role.index.tsv" \
    --tokenizer "$tokenizer" \
    --tokens-out "$processed/$role.nsrltok" \
    --trace "$processed/$role.tokens.json"
done

mkdir -p "$manifest" "$execution" "$replay"
target/release/nsrl-adaptive-composition fit-actions \
  --model "$model" \
  --fitting-tokens "$processed/fitting.nsrltok" \
  --fitting-panels "$processed/fitting.panels.tsv" \
  --out-dir "$manifest" \
  --trace "$manifest/action-manifest.json"

node scripts/freeze-adaptive-composition-v1.mjs \
  "$prereg" "$frame" "$manifest" "$processed" \
  target/release/nsrl-adaptive-composition "$precontract"

calibrate=(
  target/release/nsrl-adaptive-composition calibrate
  --manifest-dir "$manifest"
  --calibration-tokens "$processed/calibration.nsrltok"
  --calibration-panels "$processed/calibration.panels.tsv"
)
"${calibrate[@]}" --out-dir "$execution" --trace "$execution/calibration-manifest.json"
node scripts/seal-adaptive-composition-calibration-v1.mjs \
  "$precontract" "$execution" "$contract"

evaluate=(
  target/release/nsrl-adaptive-composition evaluate
  --manifest-dir "$manifest"
  --calibration-dir "$execution"
  --adaptive-tokens "$processed/adaptive.nsrltok"
  --adaptive-panels "$processed/adaptive.panels.tsv"
  --endpoint-tokens "$processed/endpoint.nsrltok"
  --endpoint-panels "$processed/endpoint.panels.tsv"
)
"${evaluate[@]}" --out-dir "$execution" --trace "$result"
"${calibrate[@]}" --out-dir "$replay" --trace "$replay/calibration-manifest.json"
evaluate[5]="$replay"
"${evaluate[@]}" --out-dir "$replay" --trace "$replay/result.json"

node scripts/check-adaptive-composition-execution-v1.mjs \
  "$contract" "$result" "$execution" "$replay"
