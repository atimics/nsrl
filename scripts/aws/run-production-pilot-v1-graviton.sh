#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

out_dir="${NSRL_PRODUCTION_PILOT_OUT_DIR:-/mnt/nsrl/production-model-v1/p10m-pilot}"
s3_uri="${NSRL_PRODUCTION_PILOT_S3_URI:?NSRL_PRODUCTION_PILOT_S3_URI is required}"
sync_seconds="${NSRL_PRODUCTION_PILOT_SYNC_SECONDS:-30}"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "aarch64" ]]; then
  echo "production pilot requires a Linux ARM64/Graviton runner" >&2
  exit 2
fi
if [[ "$s3_uri" != s3://* ]]; then
  echo "NSRL_PRODUCTION_PILOT_S3_URI must start with s3://" >&2
  exit 2
fi

mkdir -p "$out_dir"
sync_artifacts() {
  aws s3 sync "$out_dir" "$s3_uri" --only-show-errors
}

sync_loop() {
  while true; do
    sync_artifacts || true
    sleep "$sync_seconds"
  done
}

sync_loop & sync_pid=$!
cleanup() {
  kill "$sync_pid" >/dev/null 2>&1 || true
  wait "$sync_pid" >/dev/null 2>&1 || true
  sync_artifacts || true
}
trap cleanup EXIT

export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NSRL_PRODUCTION_PILOT_PARALLEL_LANES=1
export NSRL_PRODUCTION_PILOT_OUT_DIR="$out_dir"
export NSRL_PRODUCTION_PILOT_CHECKPOINT_OUT="$out_dir/checkpoint.json"

scripts/run-production-pilot-v1.sh

INSTANCE_ID="${NSRL_EC2_INSTANCE_ID:-}" INSTANCE_TYPE="${NSRL_EC2_INSTANCE_TYPE:-}" \
RUN_S3_URI="$s3_uri" OUT_PATH="$out_dir/runner.json" node --input-type=module - <<'NODE'
import fs from "node:fs";
const value = {
  schema: "nsrl.production_pilot_runner.v1",
  finished_at: new Date().toISOString(),
  architecture: process.arch,
  platform: process.platform,
  parallel_lanes: true,
  instance_id: process.env.INSTANCE_ID || "",
  instance_type: process.env.INSTANCE_TYPE || "",
  s3_uri: process.env.RUN_S3_URI,
};
fs.writeFileSync(process.env.OUT_PATH, `${JSON.stringify(value, null, 2)}\n`);
NODE
sync_artifacts
