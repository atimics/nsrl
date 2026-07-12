#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

experiment_dir="${NSRL_RECURSIVE_SWARM_OUT_DIR:-data/experiments/literary-recursive-swarm-v1}"

if [[ "$(find "$experiment_dir/leaves" -name model.nsrlmt | wc -l | tr -d ' ')" != "9" ]]; then
  echo "all nine leaf checkpoints are required before oracle scoring" >&2
  exit 1
fi

cargo build --release -p nsrl-train --bin nsrl-mini-transformer-oracle-score
node scripts/fill-recursive-literary-router-oracles.mjs \
  --experiment-dir "$experiment_dir" \
  --scorer target/release/nsrl-mini-transformer-oracle-score
