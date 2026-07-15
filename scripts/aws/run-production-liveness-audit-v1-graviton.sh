#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
out_dir="${NSRL_PRODUCTION_LIVENESS_OUT_DIR:-/mnt/nsrl/production-model-v1/p10m-liveness-audit}"
s3_uri="${NSRL_PRODUCTION_LIVENESS_S3_URI:?NSRL_PRODUCTION_LIVENESS_S3_URI is required}"
sync_seconds="${NSRL_PRODUCTION_LIVENESS_SYNC_SECONDS:-15}"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "aarch64" ]]; then
  echo "production liveness audit requires Linux ARM64/Graviton" >&2
  exit 2
fi
mkdir -p "$out_dir"
sync_artifacts() { aws s3 sync "$out_dir" "$s3_uri" --only-show-errors; }
sync_loop() { while true; do sync_artifacts || true; sleep "$sync_seconds"; done; }
sync_loop & sync_pid=$!
cleanup() {
  kill "$sync_pid" >/dev/null 2>&1 || true
  wait "$sync_pid" >/dev/null 2>&1 || true
  sync_artifacts || true
}
trap cleanup EXIT

export RUSTFLAGS="${RUSTFLAGS:--C target-cpu=native}"
export NSRL_PRODUCTION_LIVENESS_OUT_DIR="$out_dir"
export NSRL_PRODUCTION_LIVENESS_CHECKPOINT_OUT="$out_dir/checkpoint.json"
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"; started_epoch="$(date +%s)"
set +e
scripts/run-production-liveness-audit-v1.sh
status=$?
set -e
INSTANCE_ID="${NSRL_EC2_INSTANCE_ID:-}" INSTANCE_TYPE="${NSRL_EC2_INSTANCE_TYPE:-}" \
RUN_S3_URI="$s3_uri" OUT_PATH="$out_dir/runner.json" RUN_STATUS="$status" \
STARTED_AT="$started_at" FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
ELAPSED_SECONDS="$(($(date +%s) - started_epoch))" node --input-type=module - <<'NODE'
import fs from "node:fs";
fs.writeFileSync(process.env.OUT_PATH, `${JSON.stringify({
  schema: "nsrl.production_training_liveness_runner.v1",
  started_at: process.env.STARTED_AT,
  finished_at: process.env.FINISHED_AT,
  elapsed_seconds: Number(process.env.ELAPSED_SECONDS),
  exit_status: Number(process.env.RUN_STATUS),
  architecture: process.arch,
  instance_id: process.env.INSTANCE_ID || "",
  instance_type: process.env.INSTANCE_TYPE || "",
  s3_uri: process.env.RUN_S3_URI,
}, null, 2)}\n`);
NODE
sync_artifacts
exit "$status"
