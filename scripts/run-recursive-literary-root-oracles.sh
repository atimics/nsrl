#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

experiment_dir="${NSRL_RECURSIVE_SWARM_OUT_DIR:-data/experiments/literary-recursive-swarm-v1}"

if [[ "$(find "$experiment_dir/routers" -name router.nsrlrt | wc -l | tr -d ' ')" != "9" ]]; then
  echo "all nine local router artifacts are required" >&2
  exit 1
fi

cargo build --release -p nsrl-train \
  --bin nsrl-mini-transformer-oracle-score --bin nsrl-router
node scripts/build-recursive-literary-root-oracles.mjs \
  --experiment-dir "$experiment_dir" \
  --scorer target/release/nsrl-mini-transformer-oracle-score \
  --router target/release/nsrl-router
