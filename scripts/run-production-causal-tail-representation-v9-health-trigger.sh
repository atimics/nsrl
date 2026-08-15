#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

name="p10m-causal-tail-representation-v9-health-trigger"
contract="benchmarks/production-model-v1/${name}-contract.json"
out_dir="data/experiments/production-model-v1/${name}"
checkpoint="benchmarks/production-model-v1/${name}.json"
binary="target/release/nsrl-production-model"
source_model="data/experiments/production-model-v1/p10m-causal-tail-full-v1/candidate.nsrlpm"
tokenizer="data/processed/production-corpus-v1/tokenizer.nsrlbpe"
train_tokens="data/processed/production-corpus-v1/train.nsrltok"
dev_tokens="data/processed/production-corpus-v1/dev.nsrltok"
manifest="benchmarks/open-generation-v1/manifest.tsv"

mkdir -p "$out_dir"
cargo build --release -p nsrl-train \
  --bin nsrl-production-model \
  --bin nsrl-production-residual-saturation-audit \
  --bin nsrl-production-parameter-delta-audit

common_args=(
  full-train-smoke
  --tokenizer "$tokenizer" --tokens "$train_tokens"
  --context-tokens 64 --targets-per-window 8 --training-workers 8
  --spread-windows --max-windows 8192 --evaluation-windows 64 --epochs 1
  --batch-windows 4 --reject-saturated-batch
  --matrix-learning-rate-shift 59
  --q-learning-rate-shift 59 --k-learning-rate-shift 21
  --v-learning-rate-shift 23 --o-learning-rate-shift 9
  --up-learning-rate-shift 59 --gate-learning-rate-shift 59
  --down-learning-rate-shift 59 --vector-learning-rate-shift 62
  --final-rms-learning-rate-shift 59
  --embedding-learning-rate-shift 0 --embedding-learning-rate-boost-shift 3
  --flush-batched-embedding-residuals --descent-guard-windows 64
  --descent-guard-signed-representation-blocks
  --descent-guard-signed-representation-zero-saturation
  --output-learning-rate-shift 51 --output-bias-learning-rate-shift 51
  --output-backward-shift 9 --probability-gradient-fractional-bits 23
  --probability-normalization q47-newton1
)

run_phase() {
  local phase="$1"
  local model_in="$2"
  local optimizer_in="$3"
  local max_steps="$4"
  local phase_dir="$out_dir/$phase"
  if [[ -d "$phase_dir" ]]; then
    for required in model.nsrlpm optimizer.nsrlpo train.json; do
      if [[ ! -f "$phase_dir/$required" ]]; then
        echo "health trigger phase is incomplete: $phase_dir" >&2
        exit 1
      fi
    done
    echo "health trigger reuse durable phase: $phase"
    return
  fi

  local work_dir
  work_dir="$(mktemp -d "$out_dir/.${phase}.XXXXXX")"
  local args=(
    "${common_args[@]}"
    --max-optimizer-steps "$max_steps"
    --model "$model_in"
    --model-out "$work_dir/model.nsrlpm"
    --optimizer-state-out "$work_dir/optimizer.nsrlpo"
    --trace "$work_dir/train.json"
  )
  if [[ -n "$optimizer_in" ]]; then
    args+=(--optimizer-state "$optimizer_in")
  fi
  echo "health trigger phase: $phase"
  if ! "$binary" "${args[@]}"; then
    echo "health trigger phase failed; retained work directory: $work_dir" >&2
    exit 1
  fi
  mv "$work_dir" "$phase_dir"
}

run_phase prefix "$source_model" "" 407
run_phase trigger "$out_dir/prefix/model.nsrlpm" "$out_dir/prefix/optimizer.nsrlpo" 1

# This preflight is deliberately before any development or manifest read.
node scripts/freeze-production-representation-health-trigger-v1.mjs \
  --contract "$contract" --run-dir "$out_dir" --out "$checkpoint" --trigger-only

audit_dir="$(mktemp -d "$out_dir/.audit.XXXXXX")"
echo "health trigger phase: preselected development and numeric-health confirmation"
"$binary" evaluate-canonical \
  --tokenizer "$tokenizer" --tokens "$dev_tokens" \
  --model "$out_dir/trigger/model.nsrlpm" --trace "$audit_dir/development.json" \
  --context-tokens 64 --max-windows 512
target/release/nsrl-production-residual-saturation-audit \
  --manifest "$manifest" --tokenizer "$tokenizer" \
  --model "$out_dir/trigger/model.nsrlpm" --trace "$audit_dir/saturation.json" >/dev/null
target/release/nsrl-production-parameter-delta-audit \
  --source "$source_model" --candidate "$out_dir/trigger/model.nsrlpm" \
  --trace "$audit_dir/delta.json" >/dev/null

for file in "$audit_dir"/*; do
  mv "$file" "$out_dir/$(basename "$file")"
done
rmdir "$audit_dir"

node scripts/freeze-production-representation-health-trigger-v1.mjs \
  --contract "$contract" --run-dir "$out_dir" --out "$checkpoint"
