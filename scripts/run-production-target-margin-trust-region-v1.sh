#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

name="p10m-target-margin-trust-region-v1"
contract="benchmarks/production-model-v1/${name}-contract.json"
preflight="benchmarks/production-model-v1/${name}-preflight.json"
development_gate="benchmarks/production-model-v1/${name}-development-gate.json"
confirmation_gate="benchmarks/production-model-v1/${name}-confirmation-gate.json"
quality_gate="benchmarks/production-model-v1/${name}-quality-gate.json"
out_dir="data/experiments/production-model-v1/${name}"
open_generation_dir="benchmarks/open-generation-v1"
artifact_root="${NSRL_ARTIFACT_ROOT:-$repo_root}"
source_model="$artifact_root/data/experiments/production-model-v1/p10m-causal-tail-representation-v9-health-scale/chunk-3/model.nsrlpm"
tokenizer="$artifact_root/data/processed/production-corpus-v1/tokenizer.nsrlbpe"
train_tokens="$artifact_root/data/processed/production-corpus-v1/train.nsrltok"
dev_tokens="$artifact_root/data/processed/production-corpus-v1/dev.nsrltok"
test_tokens="$artifact_root/data/processed/production-corpus-v1/test.nsrltok"
manifest="benchmarks/open-generation-v1/manifest.tsv"
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
verify_sha256 "$dev_tokens" "8b02253e619f047cb3cb30bf42069fa511f3367d57ed3bb697783fc3257e37b8"
verify_sha256 "$test_tokens" "dc6c350fd02269a61b6c11e7f8c94c8ce7c0e015a337e9bf52ad9e92a0f1d1ce"

mkdir -p "$out_dir" "$open_generation_dir"
cargo build --release -p nsrl-train \
  --bin nsrl-production-model \
  --bin nsrl-production-rollout-divergence-audit \
  --bin nsrl-production-context-sensitivity-audit \
  --bin nsrl-production-residual-saturation-audit

echo "target-margin trust region: frozen feature-shift preflights"
for feature_shift in 13 14 15; do
  candidate_dir="$out_dir/preflight-shift-$feature_shift"
  mkdir -p "$candidate_dir"
  "$binary" target-margin-train \
    --tokenizer "$tokenizer" --tokens "$train_tokens" \
    --model "$source_model" \
    --model-out "$candidate_dir/candidate.nsrlpm" \
    --optimizer-state-out "$candidate_dir/candidate.nsrlmt" \
    --trace "$candidate_dir/train.json" \
    --context-tokens 64 --targets-per-window 8 --training-workers 8 \
    --spread-windows --max-windows 64 --window-schedule-windows 2048 \
    --evaluation-windows 64 --descent-guard-windows 32 \
    --epochs 1 --feature-shift "$feature_shift" --margin-q8 8 \
    --batch-windows 4 --max-optimizer-steps 16
done

node scripts/select-production-target-margin-trust-region-preflight-v1.mjs \
  --contract "$contract" \
  --trace "13:$out_dir/preflight-shift-13/train.json" \
  --trace "14:$out_dir/preflight-shift-14/train.json" \
  --trace "15:$out_dir/preflight-shift-15/train.json" \
  --out "$preflight"
selected_shift="$(node -e 'const fs=require("fs"); const value=JSON.parse(fs.readFileSync(process.argv[1])); process.stdout.write(String(value.selected_feature_shift));' "$preflight")"
echo "target-margin trust region selected feature shift: $selected_shift"

echo "target-margin trust region: uninterrupted full pilot"
"$binary" target-margin-train \
  --tokenizer "$tokenizer" --tokens "$train_tokens" \
  --model "$source_model" \
  --model-out "$out_dir/candidate.nsrlpm" \
  --optimizer-state-out "$out_dir/candidate.nsrlmt" \
  --trace "$out_dir/train-final.json" \
  --context-tokens 64 --targets-per-window 8 --training-workers 8 \
  --spread-windows --max-windows 2048 --window-schedule-windows 2048 \
  --evaluation-windows 2048 --descent-guard-windows 32 \
  --epochs 1 --feature-shift "$selected_shift" --margin-q8 8 \
  --batch-windows 4 --max-optimizer-steps 512

echo "target-margin trust region: exact midpoint restart replay"
"$binary" target-margin-train \
  --tokenizer "$tokenizer" --tokens "$train_tokens" \
  --model "$source_model" \
  --model-out "$out_dir/midpoint.nsrlpm" \
  --optimizer-state-out "$out_dir/midpoint.nsrlmt" \
  --trace "$out_dir/train-midpoint.json" \
  --context-tokens 64 --targets-per-window 8 --training-workers 8 \
  --spread-windows --max-windows 2048 --window-schedule-windows 2048 \
  --evaluation-windows 2048 --descent-guard-windows 32 \
  --epochs 1 --feature-shift "$selected_shift" --margin-q8 8 \
  --batch-windows 4 --max-optimizer-steps 256
"$binary" target-margin-train \
  --tokenizer "$tokenizer" --tokens "$train_tokens" \
  --model "$out_dir/midpoint.nsrlpm" \
  --optimizer-state "$out_dir/midpoint.nsrlmt" \
  --model-out "$out_dir/replay-candidate.nsrlpm" \
  --optimizer-state-out "$out_dir/replay-candidate.nsrlmt" \
  --trace "$out_dir/train-replay.json" \
  --context-tokens 64 --targets-per-window 8 --training-workers 8 \
  --spread-windows --max-windows 2048 --window-schedule-windows 2048 \
  --evaluation-windows 2048 --descent-guard-windows 32 \
  --epochs 1 --feature-shift "$selected_shift" --margin-q8 8 \
  --batch-windows 4 --max-optimizer-steps 512
cmp "$out_dir/candidate.nsrlpm" "$out_dir/replay-candidate.nsrlpm"
cmp "$out_dir/candidate.nsrlmt" "$out_dir/replay-candidate.nsrlmt"

echo "target-margin trust region: public development stop/go gate"
"$binary" evaluate-canonical \
  --tokenizer "$tokenizer" --tokens "$dev_tokens" --model "$source_model" \
  --trace "$out_dir/source-dev.json" --context-tokens 64 --max-windows 512
"$binary" evaluate-canonical \
  --tokenizer "$tokenizer" --tokens "$dev_tokens" --model "$out_dir/candidate.nsrlpm" \
  --trace "$out_dir/candidate-dev.json" --context-tokens 64 --max-windows 512
for role in source candidate; do
  if [[ "$role" == "source" ]]; then
    model="$source_model"
  else
    model="$out_dir/candidate.nsrlpm"
  fi
  target/release/nsrl-production-rollout-divergence-audit \
    --tokenizer "$tokenizer" --tokens "$dev_tokens" --model "$model" \
    --trace "$out_dir/${role}-dev-rollout.json" \
    --context-tokens 64 --rollout-tokens 16 --max-windows 64
done
node scripts/check-production-target-margin-trust-region-development-v1.mjs \
  --contract "$contract" --selection "$preflight" \
  --train "$out_dir/train-final.json" \
  --candidate "$out_dir/candidate.nsrlpm" \
  --replay "$out_dir/replay-candidate.nsrlpm" \
  --optimizer "$out_dir/candidate.nsrlmt" \
  --replay-optimizer "$out_dir/replay-candidate.nsrlmt" \
  --source-dev "$out_dir/source-dev.json" \
  --candidate-dev "$out_dir/candidate-dev.json" \
  --source-rollout "$out_dir/source-dev-rollout.json" \
  --candidate-rollout "$out_dir/candidate-dev-rollout.json" \
  --out "$development_gate"

echo "target-margin trust region: public test confirmation gate"
"$binary" evaluate-canonical \
  --tokenizer "$tokenizer" --tokens "$test_tokens" --model "$source_model" \
  --trace "$out_dir/source-test.json" --context-tokens 64 --max-windows 512
"$binary" evaluate-canonical \
  --tokenizer "$tokenizer" --tokens "$test_tokens" --model "$out_dir/candidate.nsrlpm" \
  --trace "$out_dir/candidate-test.json" --context-tokens 64 --max-windows 512
for role in source candidate; do
  if [[ "$role" == "source" ]]; then
    model="$source_model"
  else
    model="$out_dir/candidate.nsrlpm"
  fi
  target/release/nsrl-production-rollout-divergence-audit \
    --tokenizer "$tokenizer" --tokens "$test_tokens" --model "$model" \
    --trace "$out_dir/${role}-test-rollout.json" \
    --context-tokens 64 --rollout-tokens 16 --max-windows 64
done
node scripts/check-production-target-margin-trust-region-confirmation-v1.mjs \
  --contract "$contract" --development "$development_gate" \
  --source-test "$out_dir/source-test.json" \
  --candidate-test "$out_dir/candidate-test.json" \
  --source-rollout "$out_dir/source-test-rollout.json" \
  --candidate-rollout "$out_dir/candidate-test-rollout.json" \
  --out "$confirmation_gate"

echo "target-margin trust region: authorized open-generation checks"
for role in source candidate; do
  if [[ "$role" == "source" ]]; then
    model="$source_model"
  else
    model="$out_dir/candidate.nsrlpm"
  fi
  target/release/nsrl-production-context-sensitivity-audit \
    --manifest "$manifest" --tokenizer "$tokenizer" --model "$model" \
    --trace "$open_generation_dir/${name}-${role}-context-sensitivity.json" --top-k 8
  target/release/nsrl-production-residual-saturation-audit \
    --manifest "$manifest" --tokenizer "$tokenizer" --model "$model" \
    --trace "$open_generation_dir/${name}-${role}-residual-saturation.json" >/dev/null
done
node scripts/freeze-production-target-margin-trust-region-quality-v1.mjs \
  --contract "$contract" --confirmation "$confirmation_gate" \
  --source-context "$open_generation_dir/${name}-source-context-sensitivity.json" \
  --candidate-context "$open_generation_dir/${name}-candidate-context-sensitivity.json" \
  --source-saturation "$open_generation_dir/${name}-source-residual-saturation.json" \
  --candidate-saturation "$open_generation_dir/${name}-candidate-residual-saturation.json" \
  --out "$quality_gate"
