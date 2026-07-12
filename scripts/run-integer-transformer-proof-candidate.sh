#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_dir="${1:-}"
if [[ -z "$out_dir" ]]; then
  echo "usage: scripts/run-integer-transformer-proof-candidate.sh OUT_DIR" >&2
  exit 2
fi

manifest="benchmarks/integer-transformer-proof-v1/manifest.tsv"
baselines="benchmarks/integer-transformer-proof-v1/baselines.tsv"
train_tokens="benchmarks/integer-transformer-proof-v1/train.txt"
eval_tokens="benchmarks/integer-transformer-proof-v1/eval.txt"
context=64
requested_max_windows="${NSRL_PROOF_MAX_WINDOWS:-512}"
epochs="${NSRL_PROOF_EPOCHS:-1}"
batch_windows="${NSRL_PROOF_BATCH_WINDOWS:-16}"
workers="${NSRL_PROOF_WORKERS:-4}"
profile="${NSRL_PROOF_PROFILE:-h8}"
adam_step_shift="${NSRL_PROOF_ADAM_STEP_SHIFT:-5}"
argmax_margin_weight_q15="${NSRL_PROOF_ARGMAX_MARGIN_WEIGHT_Q15:-1024}"
target_frequency_cap="${NSRL_PROOF_TARGET_FREQUENCY_CAP:-0}"
target_frequency_min_weight_q15="${NSRL_PROOF_TARGET_FREQUENCY_MIN_WEIGHT_Q15:-4096}"
rms_norm_initial_gamma_q15="${NSRL_PROOF_RMS_NORM_INITIAL_GAMMA_Q15:-16384}"
attention_kind="${NSRL_PROOF_ATTENTION:-linear}"
position_policy="${NSRL_PROOF_POSITION:-nope}"
max_rejected_batches="${NSRL_PROOF_MAX_REJECTED_BATCHES:-0}"
max_mlp_saturations="${NSRL_PROOF_MAX_MLP_SATURATIONS:-auto}"
max_attention_saturations="${NSRL_PROOF_MAX_ATTENTION_SATURATIONS:-auto}"
max_residual_saturations="${NSRL_PROOF_MAX_RESIDUAL_SATURATIONS:-auto}"
require_attention_deltas="${NSRL_PROOF_REQUIRE_ATTENTION_DELTAS:-1}"
require_rms_norm="${NSRL_PROOF_REQUIRE_RMS_NORM:-1}"
require_loss_improvement="${NSRL_PROOF_REQUIRE_LOSS_IMPROVEMENT:-1}"
min_unique_predictions="${NSRL_PROOF_MIN_UNIQUE_PREDICTIONS:-8}"
max_prediction_share_per_mille="${NSRL_PROOF_MAX_PREDICTION_SHARE_PER_MILLE:-900}"
calibrated_profile="${NSRL_PROOF_CALIBRATED_PROFILE:-1}"

for value in "$epochs" "$batch_windows" "$workers"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "proof training settings must be positive integers: $value" >&2
    exit 2
  fi
done
for value in "$adam_step_shift" "$argmax_margin_weight_q15" "$target_frequency_cap" "$target_frequency_min_weight_q15" "$rms_norm_initial_gamma_q15" "$max_rejected_batches" "$min_unique_predictions" "$max_prediction_share_per_mille"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "proof optimizer and health settings must be nonnegative integers: $value" >&2
    exit 2
  fi
done
if ((min_unique_predictions > 256)); then
  echo "NSRL_PROOF_MIN_UNIQUE_PREDICTIONS must be <= 256: $min_unique_predictions" >&2
  exit 2
fi
if ((max_prediction_share_per_mille > 1000)); then
  echo "NSRL_PROOF_MAX_PREDICTION_SHARE_PER_MILLE must be <= 1000: $max_prediction_share_per_mille" >&2
  exit 2
fi
if ((argmax_margin_weight_q15 > 32767)); then
  echo "NSRL_PROOF_ARGMAX_MARGIN_WEIGHT_Q15 must fit Q15: $argmax_margin_weight_q15" >&2
  exit 2
fi
if ((target_frequency_min_weight_q15 == 0 || target_frequency_min_weight_q15 > 32767)); then
  echo "NSRL_PROOF_TARGET_FREQUENCY_MIN_WEIGHT_Q15 must be in 1..32767: $target_frequency_min_weight_q15" >&2
  exit 2
fi
if ((rms_norm_initial_gamma_q15 == 0 || rms_norm_initial_gamma_q15 > 32767)); then
  echo "NSRL_PROOF_RMS_NORM_INITIAL_GAMMA_Q15 must be in 1..32767: $rms_norm_initial_gamma_q15" >&2
  exit 2
fi
for value in "$require_attention_deltas" "$require_rms_norm" "$require_loss_improvement" "$calibrated_profile"; do
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    echo "proof boolean settings must be 0 or 1: $value" >&2
    exit 2
  fi
done
case "$attention_kind" in
  linear|base2-softmax) ;;
  *)
    echo "NSRL_PROOF_ATTENTION must be linear or base2-softmax: $attention_kind" >&2
    exit 2
    ;;
esac
case "$position_policy" in
  nope|learned-absolute) ;;
  *)
    echo "NSRL_PROOF_POSITION must be nope or learned-absolute: $position_policy" >&2
    exit 2
    ;;
esac

case "$profile" in
  h8|small-h8-d128-ff256)
    profile="h8"
    expected_profile="small-h8-d128-ff256"
    ;;
  h2|small-h2-d128-ff256)
    profile="h2"
    expected_profile="small-h2-d128-ff256"
    ;;
  *)
    echo "NSRL_PROOF_PROFILE must be h2 or h8: $profile" >&2
    exit 2
    ;;
esac

mkdir -p "$out_dir"
target_dir="${NSRL_PROOF_TARGET_DIR:-target/integer-transformer-proof-$profile}"
if [[ "$profile" == "h8" && "$calibrated_profile" == "1" ]]; then
  CARGO_TARGET_DIR="$target_dir" cargo build --release -p nsrl-train \
    --bin nsrl-train --bin nsrl-mini-transformer-eval \
    --features mini-heads-8,mini-calibrated
elif [[ "$profile" == "h8" ]]; then
  CARGO_TARGET_DIR="$target_dir" cargo build --release -p nsrl-train \
    --bin nsrl-train --bin nsrl-mini-transformer-eval --features mini-heads-8
elif [[ "$calibrated_profile" == "1" ]]; then
  CARGO_TARGET_DIR="$target_dir" cargo build --release -p nsrl-train \
    --bin nsrl-train --bin nsrl-mini-transformer-eval --features mini-calibrated
else
  CARGO_TARGET_DIR="$target_dir" cargo build --release -p nsrl-train \
    --bin nsrl-train --bin nsrl-mini-transformer-eval
fi
cargo build --release -p nsrl-eval --bin nsrl-eval

target/release/nsrl-eval manifest --manifest "$manifest" > "$out_dir/manifest.json"
node scripts/run-integer-transformer-proof-baselines.mjs \
  --manifest "$manifest" \
  --out "$out_dir/baselines.tsv"
cmp "$baselines" "$out_dir/baselines.tsv"

train_bytes="$(wc -c < "$train_tokens" | tr -d ' ')"
trainable=$((train_bytes > context ? train_bytes - context : 1))
if [[ "$requested_max_windows" == "all" ]]; then
  max_windows="$trainable"
elif [[ "$requested_max_windows" =~ ^[1-9][0-9]*$ ]]; then
  max_windows="$requested_max_windows"
  if ((max_windows > trainable)); then
    max_windows="$trainable"
  fi
else
  echo "NSRL_PROOF_MAX_WINDOWS must be a positive integer or all: $requested_max_windows" >&2
  exit 2
fi
train_stride=$(((trainable + max_windows - 1) / max_windows))
((train_stride > 0)) || train_stride=1
examined_window_budget=$((max_windows * epochs))
if [[ "$max_mlp_saturations" == "auto" ]]; then
  max_mlp_saturations=$((examined_window_budget * 512))
elif [[ ! "$max_mlp_saturations" =~ ^[0-9]+$ ]]; then
  echo "NSRL_PROOF_MAX_MLP_SATURATIONS must be a nonnegative integer or auto" >&2
  exit 2
fi
if [[ "$max_attention_saturations" == "auto" ]]; then
  max_attention_saturations=$((examined_window_budget * 512))
elif [[ ! "$max_attention_saturations" =~ ^[0-9]+$ ]]; then
  echo "NSRL_PROOF_MAX_ATTENTION_SATURATIONS must be a nonnegative integer or auto" >&2
  exit 2
fi
if [[ "$max_residual_saturations" == "auto" ]]; then
  max_residual_saturations=$((examined_window_budget * 32768))
elif [[ ! "$max_residual_saturations" =~ ^[0-9]+$ ]]; then
  echo "NSRL_PROOF_MAX_RESIDUAL_SATURATIONS must be a nonnegative integer or auto" >&2
  exit 2
fi

"$target_dir/release/nsrl-train" \
  --mode mini-transformer-adam \
  --tokens "$train_tokens" \
  --model-out "$out_dir/candidate.nsrlmt" \
  --optimizer-state-out "$out_dir/candidate.nsrlad" \
  --rms-norm-initial-gamma-q15 "$rms_norm_initial_gamma_q15" \
  --adam-step-shift "$adam_step_shift" \
  --argmax-margin-weight-q15 "$argmax_margin_weight_q15" \
  --target-frequency-cap "$target_frequency_cap" \
  --target-frequency-min-weight-q15 "$target_frequency_min_weight_q15" \
  --epochs "$epochs" \
  --seq-len "$context" \
  --stride "$train_stride" \
  --max-windows "$max_windows" \
  --batch-windows "$batch_windows" \
  --tokenizer identity \
  --mini-transformer-attention "$attention_kind" \
  --mini-transformer-position "$position_policy" \
  --mini-transformer-batch-mode map-reduce \
  --mini-transformer-map-reduce-workers "$workers" \
  --trace "$out_dir/train.trace.jsonl"

"$target_dir/release/nsrl-mini-transformer-eval" \
  --tokens "$eval_tokens" \
  --model "$out_dir/candidate.nsrlmt" \
  --stride 1 \
  --attention "$attention_kind" \
  --position "$position_policy" \
  --out "$out_dir/candidate.eval.json"

health_args=(
  --trace "$out_dir/train.trace.jsonl"
  --eval "$out_dir/candidate.eval.json"
  --out "$out_dir/candidate-health.json"
  --expected-profile "$expected_profile"
  --min-transformer-layers 2
  --max-rejected-batches "$max_rejected_batches"
  --max-mlp-saturations "$max_mlp_saturations"
  --max-attention-saturations "$max_attention_saturations"
  --max-residual-saturations "$max_residual_saturations"
  --min-unique-predictions "$min_unique_predictions"
  --max-prediction-share-per-mille "$max_prediction_share_per_mille"
)
if [[ "$calibrated_profile" == "1" ]]; then
  health_args+=(--expected-quantization-profile calibrated-v2-suffix-memory)
fi
if [[ "$require_attention_deltas" == "0" ]]; then
  health_args+=(--allow-dead-attention)
fi
if [[ "$require_rms_norm" == "0" ]]; then
  health_args+=(--allow-no-rms-norm)
fi
if [[ "$require_loss_improvement" == "0" ]]; then
  health_args+=(--allow-loss-regression)
fi
set +e
node scripts/check-integer-transformer-candidate-health.mjs "${health_args[@]}"
health_status="$?"
set -e

node scripts/build-integer-transformer-proof-results.mjs \
  --manifest "$manifest" \
  --baselines "$baselines" \
  --candidate-trace "$out_dir/candidate.eval.json" \
  --out "$out_dir/proof-results.tsv"

set +e
target/release/nsrl-eval check \
  --manifest "$manifest" \
  --results "$out_dir/proof-results.tsv" | tee "$out_dir/proof-check.json"
proof_status="${PIPESTATUS[0]}"
set -e

if [[ "$health_status" -eq 2 || "$proof_status" -eq 2 ]]; then
  echo "integer-transformer-proof-v1 artifact validation failed: $out_dir" >&2
  exit 2
fi
if [[ "$health_status" -ne 0 ]]; then
  echo "integer-transformer-proof-v1 candidate failed mechanical health gates: $out_dir/candidate-health.json" >&2
fi
if [[ "$proof_status" -eq 0 && "$health_status" -eq 0 ]]; then
  echo "integer-transformer-proof-v1 passed: $out_dir"
  exit 0
elif [[ "$proof_status" -eq 1 ]]; then
  echo "integer-transformer-proof-v1 measured but did not pass: $out_dir" >&2
fi
exit 1
