#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

experiment_dir="${NSRL_RECURSIVE_SWARM_OUT_DIR:-data/experiments/literary-recursive-swarm-v1}"
oracle_dir="$experiment_dir/router-oracles"

if [[ ! -s "$oracle_dir/oracle-report.json" ]]; then
  echo "local oracle data is missing; run scripts/run-recursive-literary-router-oracles.sh" >&2
  exit 1
fi

cargo build --release -p nsrl-train --bin nsrl-router

author_index=0
for author in crowley shakespeare blake; do
  author_index=$((author_index + 1))
  view_index=0
  for view in semantic structural full; do
    view_index=$((view_index + 1))
    case "$view" in
      semantic) features="0-23" ;;
      structural) features="24-40" ;;
      full) features="0-40" ;;
    esac
    seed=$((author_index * 100 + view_index))
    router_dir="$experiment_dir/routers/$author-router-$view"
    mkdir -p "$router_dir"
    target/release/nsrl-router train \
      --train "$oracle_dir/$author-router-train.router.tsv" \
      --calibration "$oracle_dir/$author-router-calibration.router.tsv" \
      --features "$features" \
      --epochs 64 \
      --seed "$seed" \
      --model-out "$router_dir/router.nsrlrt" \
      --trace "$router_dir/train.trace.jsonl" \
      --predictions-out "$router_dir/calibration.predictions.tsv"
    target/release/nsrl-router eval \
      --data "$oracle_dir/$author-final-test.router.tsv" \
      --model "$router_dir/router.nsrlrt" \
      --trace "$router_dir/final.eval.jsonl" \
      --predictions-out "$router_dir/final.predictions.tsv"
  done
done

echo "trained local router artifacts: $(find "$experiment_dir/routers" -name router.nsrlrt | wc -l | tr -d ' ')"
