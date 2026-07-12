#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_dir="${1:-}"
if [[ -z "$out_dir" ]]; then
  echo "usage: scripts/run-mt6-local-overfit-gate.sh OUT_DIR" >&2
  exit 2
fi

seq_len="${NSRL_MT6_SEQ_LEN:-64}"
windows="${NSRL_MT6_OVERFIT_WINDOWS:-256}"
epochs="${NSRL_MT6_OVERFIT_EPOCHS:-64}"
min_accuracy="${NSRL_MT6_MIN_OVERFIT_ACCURACY_PER_MILLE:-900}"
max_residual="${NSRL_MT6_MAX_RESIDUAL_SATURATIONS_PER_WINDOW:-4096}"

for value in "$seq_len" "$windows" "$epochs" "$min_accuracy" "$max_residual"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "MT6 gate settings must be positive integers: $value" >&2
    exit 2
  fi
done
if ((seq_len < 4 || min_accuracy > 1000)); then
  echo "MT6 gate requires seq_len >= 4 and accuracy <= 1000" >&2
  exit 2
fi

mkdir -p "$out_dir"
cargo build --release -p nsrl-train --bin nsrl-mt6-overfit
target/release/nsrl-mt6-overfit \
  --tokens benchmarks/integer-transformer-proof-v1/train.txt \
  --model-out "$out_dir/candidate.nsrlmt6" \
  --trace-out "$out_dir/overfit.trace.jsonl" \
  --seq-len "$seq_len" \
  --windows "$windows" \
  --epochs "$epochs" \
  --min-accuracy-per-mille "$min_accuracy" \
  --max-residual-saturations-per-window "$max_residual"

echo "NSRLMT6 local overfit gate passed: $out_dir"
