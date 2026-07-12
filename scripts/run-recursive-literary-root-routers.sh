#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

experiment_dir="${NSRL_RECURSIVE_SWARM_OUT_DIR:-data/experiments/literary-recursive-swarm-v1}"
oracle_dir="$experiment_dir/root-oracles"

if [[ ! -s "$oracle_dir/root-oracle-report.json" ]]; then
  echo "root oracle data is missing; run scripts/run-recursive-literary-root-oracles.sh" >&2
  exit 1
fi

cargo build --release -p nsrl-train --bin nsrl-router

view_index=0
for view in semantic structural full; do
  view_index=$((view_index + 1))
  case "$view" in
    semantic) features="0-23" ;;
    structural) features="24-40" ;;
    full) features="0-40" ;;
  esac
  router_dir="$experiment_dir/routers/root-router-$view"
  mkdir -p "$router_dir"
  target/release/nsrl-router train \
    --train "$oracle_dir/root-router-train.router.tsv" \
    --calibration "$oracle_dir/root-router-calibration.router.tsv" \
    --features "$features" \
    --epochs 64 \
    --seed "$((900 + view_index))" \
    --model-out "$router_dir/router.nsrlrt" \
    --trace "$router_dir/train.trace.jsonl" \
    --predictions-out "$router_dir/calibration.predictions.tsv"
  target/release/nsrl-router eval \
    --data "$oracle_dir/root-final-test.router.tsv" \
    --model "$router_dir/router.nsrlrt" \
    --trace "$router_dir/final.eval.jsonl" \
    --predictions-out "$router_dir/final.predictions.tsv"
done

node scripts/summarize-recursive-literary-root-routers.mjs \
  --experiment-dir "$experiment_dir" \
  --out "$experiment_dir/root-router-report.json"

echo "trained root router artifacts: $(find "$experiment_dir/routers" -path '*/root-router-*/router.nsrlrt' | wc -l | tr -d ' ')"
