#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

s3_uri="${1:?usage: fetch-production-pilot-v1.sh s3://bucket/prefix}"
out_dir="${NSRL_PRODUCTION_PILOT_OUT_DIR:-data/experiments/production-model-v1/p10m-pilot}"
aws s3 sync "$s3_uri" "$out_dir" --only-show-errors
node scripts/freeze-production-pilot-v1.mjs --run-dir "$out_dir"
