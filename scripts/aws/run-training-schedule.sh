#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" || $# -lt 1 ]]; then
  cat <<'USAGE'
Run multiple AWS training jobs from a simple schedule file.

Schedule format:
  # run name followed by KEY=VALUE overrides
  smoke-a NSRL_MAX_WINDOWS=8192 NSRL_SEQ_LEN=4 NSRL_STRIDE=36965
  hero-a  NSRL_MAX_WINDOWS=250000 NSRL_SEQ_LEN=4 NSRL_STRIDE=1211

Required environment:
  NSRL_S3_URI=s3://bucket/prefix

Optional:
  NSRL_MAX_PARALLEL=4
USAGE
  exit 0
fi

schedule_file="$1"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

if [[ ! -f "$schedule_file" ]]; then
  echo "missing schedule file: $schedule_file" >&2
  exit 2
fi

max_parallel="${NSRL_MAX_PARALLEL:-4}"
if [[ "$max_parallel" -lt 1 ]]; then
  echo "NSRL_MAX_PARALLEL must be >= 1" >&2
  exit 2
fi

schedule_name="${NSRL_SCHEDULE_NAME:-$(basename "$schedule_file" | tr -c 'A-Za-z0-9_.-' '-')}"
if [[ -n "${NSRL_S3_URI:-}" ]]; then
  aws s3 cp "$schedule_file" "${NSRL_S3_URI%/}/schedules/${schedule_name}" --only-show-errors
fi

running=0
pids=()

wait_one() {
  local pid="${pids[0]}"
  wait "$pid"
  pids=("${pids[@]:1}")
  running=$((running - 1))
}

while read -r run_name rest; do
  [[ -z "${run_name:-}" || "$run_name" == \#* ]] && continue
  while [[ "$running" -ge "$max_parallel" ]]; do
    wait_one
  done
  echo "starting $run_name $rest"
  (
    # shellcheck disable=SC2086
    env NSRL_RUN_NAME="$run_name" $rest scripts/aws/run-mini-transformer-training.sh
  ) &
  pids+=("$!")
  running=$((running + 1))
done < "$schedule_file"

while [[ "$running" -gt 0 ]]; do
  wait_one
done
