#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

name="p10m-direct-head-nll-safe-set-v1"
contract="benchmarks/production-model-v1/${name}-contract.json"
training_gate="benchmarks/production-model-v1/${name}-training-gate.json"
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

mkdir -p "$out_dir" "$open_generation_dir"
cargo build --release -p nsrl-train \
  --bin nsrl-production-model \
  --bin nsrl-production-rollout-divergence-audit \
  --bin nsrl-production-context-sensitivity-audit \
  --bin nsrl-production-residual-saturation-audit

echo "direct-head NLL safe set: exact bounded run"
"$binary" direct-head-train \
  --tokenizer "$tokenizer" --tokens "$train_tokens" \
  --model "$source_model" \
  --model-out "$out_dir/candidate.nsrlpm" \
  --trace "$out_dir/train.json" \
  --context-tokens 64 --max-windows 64 --evaluation-windows 32 \
  --coordinates-per-group 8 --max-optimizer-steps 8 \
  --probability-gradient-fractional-bits 23 \
  --probability-normalization q47-newton1 \
  --direct-head-exact-safe-set --seed 0

echo "direct-head NLL safe set: exact full rerun replay"
"$binary" direct-head-train \
  --tokenizer "$tokenizer" --tokens "$train_tokens" \
  --model "$source_model" \
  --model-out "$out_dir/replay-candidate.nsrlpm" \
  --trace "$out_dir/replay-train.json" \
  --context-tokens 64 --max-windows 64 --evaluation-windows 32 \
  --coordinates-per-group 8 --max-optimizer-steps 8 \
  --probability-gradient-fractional-bits 23 \
  --probability-normalization q47-newton1 \
  --direct-head-exact-safe-set --seed 0

echo "direct-head NLL safe set: private training stop/go gate"
node scripts/check-production-direct-head-nll-safe-set-training-v1.mjs \
  --contract "$contract" \
  --train "$out_dir/train.json" \
  --replay-train "$out_dir/replay-train.json" \
  --candidate "$out_dir/candidate.nsrlpm" \
  --replay "$out_dir/replay-candidate.nsrlpm" \
  --out "$training_gate"

echo "direct-head NLL safe set: public development stop/go gate"
verify_sha256 "$dev_tokens" "8b02253e619f047cb3cb30bf42069fa511f3367d57ed3bb697783fc3257e37b8"
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
node scripts/check-production-direct-head-nll-safe-set-development-v1.mjs \
  --contract "$contract" --training "$training_gate" \
  --source-dev "$out_dir/source-dev.json" \
  --candidate-dev "$out_dir/candidate-dev.json" \
  --source-rollout "$out_dir/source-dev-rollout.json" \
  --candidate-rollout "$out_dir/candidate-dev-rollout.json" \
  --out "$development_gate"

echo "direct-head NLL safe set: public test confirmation gate"
verify_sha256 "$test_tokens" "dc6c350fd02269a61b6c11e7f8c94c8ce7c0e015a337e9bf52ad9e92a0f1d1ce"
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
node scripts/check-production-direct-head-nll-safe-set-confirmation-v1.mjs \
  --contract "$contract" --development "$development_gate" \
  --source-test "$out_dir/source-test.json" \
  --candidate-test "$out_dir/candidate-test.json" \
  --source-rollout "$out_dir/source-test-rollout.json" \
  --candidate-rollout "$out_dir/candidate-test-rollout.json" \
  --out "$confirmation_gate"

echo "direct-head NLL safe set: authorized open-generation checks"
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
node scripts/freeze-production-direct-head-nll-safe-set-quality-v1.mjs \
  --contract "$contract" --confirmation "$confirmation_gate" \
  --source-context "$open_generation_dir/${name}-source-context-sensitivity.json" \
  --candidate-context "$open_generation_dir/${name}-candidate-context-sensitivity.json" \
  --source-saturation "$open_generation_dir/${name}-source-residual-saturation.json" \
  --candidate-saturation "$open_generation_dir/${name}-candidate-residual-saturation.json" \
  --out "$quality_gate"
