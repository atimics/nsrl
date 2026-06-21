#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
Run an NSRL mini-transformer training job and publish a static S3 dashboard.

Required:
  NSRL_S3_URI=s3://bucket/prefix

Common knobs:
  NSRL_RUN_NAME=hero-001
  NSRL_TOKENS=data/processed/wiki-bard-corpus.tokens.u8
  NSRL_TOKENS_S3_URI=s3://bucket/path/wiki-bard-corpus.tokens.u8
  NSRL_MODEL=data/processed/resume.nsrlmt
  NSRL_MODEL_S3_URI=s3://bucket/path/resume.nsrlmt
  NSRL_RESUME_CHECKPOINT=wiki-bard-golden
  NSRL_PUBLISH_CHECKPOINT=wiki-bard-golden
  NSRL_MAX_WINDOWS=32768
  NSRL_SEQ_LEN=4
  NSRL_STRIDE=1
  NSRL_BATCH_WINDOWS=2
  NSRL_ATTENTION=linear
  NSRL_POSITION=nope
  NSRL_ADAPTIVE_RULE_SHIFTS=1
  NSRL_ADAPTIVE_HOLOGRAPHIC_SHIFTS=0
  NSRL_SYNC_SECONDS=60

Artifacts:
  Local: data/aws-runs/<run-name>/
  S3:    $NSRL_S3_URI/runs/<run-name>/
  UI:    $NSRL_S3_URI/dashboard/index.html
USAGE
  exit 0
fi

if [[ -z "${NSRL_S3_URI:-}" ]]; then
  echo "NSRL_S3_URI is required, e.g. s3://my-bucket/nsrl" >&2
  exit 2
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI is required" >&2
  exit 2
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
run_name="${NSRL_RUN_NAME:-mini-transformer-${timestamp}}"
run_root="${NSRL_RUN_ROOT:-data/aws-runs}"
run_dir="${run_root}/${run_name}"
dashboard_dir="${run_root}/dashboard"
mkdir -p "$run_dir" "$dashboard_dir" "$(dirname "${NSRL_TOKENS:-data/processed/wiki-bard-corpus.tokens.u8}")"

s3_uri="${NSRL_S3_URI%/}"
tokens="${NSRL_TOKENS:-data/processed/wiki-bard-corpus.tokens.u8}"
model_in="${NSRL_MODEL:-}"
resume_model_s3_uri="${NSRL_MODEL_S3_URI:-${NSRL_RESUME_FROM_S3_URI:-}}"
if [[ -z "$resume_model_s3_uri" && -n "${NSRL_RESUME_CHECKPOINT:-}" ]]; then
  resume_model_s3_uri="${s3_uri}/checkpoints/${NSRL_RESUME_CHECKPOINT}/latest.nsrlmt"
fi
if [[ -z "$model_in" && -n "$resume_model_s3_uri" ]]; then
  model_in="${run_dir}/resume.nsrlmt"
fi
model_out="${run_dir}/${run_name}.nsrlmt"
trace_out="${run_dir}/${run_name}.trace.jsonl"
progress_out="${run_dir}/${run_name}.progress.jsonl"
log_out="${run_dir}/train.log"
command_out="${run_dir}/command.txt"
repo_rev="$(git rev-parse HEAD 2>/dev/null || true)"
sync_seconds="${NSRL_SYNC_SECONDS:-60}"

if [[ ! -f "$tokens" ]]; then
  if [[ -n "${NSRL_TOKENS_S3_URI:-}" ]]; then
    mkdir -p "$(dirname "$tokens")"
    aws s3 cp "$NSRL_TOKENS_S3_URI" "$tokens"
  else
    echo "Token file not found: $tokens" >&2
    echo "Set NSRL_TOKENS_S3_URI to download it on the instance." >&2
    exit 2
  fi
fi

if [[ -n "$model_in" && ! -f "$model_in" ]]; then
  if [[ -n "$resume_model_s3_uri" ]]; then
    mkdir -p "$(dirname "$model_in")"
    aws s3 cp "$resume_model_s3_uri" "$model_in" --only-show-errors
  else
    echo "Model file not found: $model_in" >&2
    echo "Set NSRL_MODEL_S3_URI, NSRL_RESUME_FROM_S3_URI, or NSRL_RESUME_CHECKPOINT to download it on the instance." >&2
    exit 2
  fi
fi

aws s3 cp "${s3_uri}/dashboard/runs.json" "${dashboard_dir}/runs.json" >/dev/null 2>&1 || true

cargo build --release -p nsrl-train

cmd=(
  cargo run --release -p nsrl-train --
  --mode mini-transformer-mlp
  --tokens "$tokens"
  --seq-len "${NSRL_SEQ_LEN:-4}"
  --stride "${NSRL_STRIDE:-1}"
  --window-offset "${NSRL_WINDOW_OFFSET:-0}"
  --batch-windows "${NSRL_BATCH_WINDOWS:-2}"
  --max-windows "${NSRL_MAX_WINDOWS:-32768}"
  --epochs "${NSRL_EPOCHS:-1}"
  --lr-shift "${NSRL_OUT_SHIFT:-18}"
  --mlp-lr-shift "${NSRL_MLP_SHIFT:-17}"
  --embed-lr-shift "${NSRL_EMBED_SHIFT:-13}"
  --attention-lr-shift "${NSRL_ATTENTION_SHIFT:-22}"
  --attention-q-lr-shift "${NSRL_ATTENTION_Q_SHIFT:-18}"
  --attention-qk-lr-shift "${NSRL_ATTENTION_QK_SHIFT:-16}"
  --mini-transformer-attention "${NSRL_ATTENTION:-linear}"
  --mini-transformer-position "${NSRL_POSITION:-nope}"
  --model-out "$model_out"
  --trace "$trace_out"
  --progress-out "$progress_out"
  --progress-interval-batches "${NSRL_PROGRESS_INTERVAL_BATCHES:-128}"
)

if [[ -n "$model_in" ]]; then
  cmd+=(--resume-from "$model_in")
fi

if [[ "${NSRL_ADAPTIVE_RULE_SHIFTS:-1}" != "0" ]]; then
  cmd+=(--adaptive-rule-shifts)
  cmd+=(--adaptive-rule-interval-batches "${NSRL_ADAPTIVE_RULE_INTERVAL_BATCHES:-128}")
fi

if [[ "${NSRL_ADAPTIVE_HOLOGRAPHIC_SHIFTS:-0}" != "0" ]]; then
  cmd+=(--adaptive-holographic-shifts)
fi

if [[ "${NSRL_REJECT_LOSS_REGRESSION:-0}" != "0" ]]; then
  cmd+=(--reject-loss-regression)
fi

printf '%q ' "${cmd[@]}" > "$command_out"
printf '\n' >> "$command_out"

render_dashboard() {
  local status="$1"
  local exit_code="${2:-}"
  local updated_at
  updated_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local args=(
    scripts/aws/render-dashboard.py
    --run-dir "$run_dir"
    --dashboard-dir "$dashboard_dir"
    --run-name "$run_name"
    --s3-uri "$s3_uri"
    --status "$status"
    --started-at "$started_at"
    --updated-at "$updated_at"
    --repo-rev "$repo_rev"
    --tokens "$tokens"
    --command-file "$command_out"
    --log-file "$log_out"
    --progress-file "$progress_out"
    --trace-file "$trace_out"
    --model-file "$model_out"
  )
  if [[ -n "$exit_code" ]]; then
    args+=(--exit-code "$exit_code" --finished-at "$updated_at")
  fi
  python3 "${args[@]}"
}

sync_dashboard() {
  aws s3 sync "$run_dir" "${s3_uri}/runs/${run_name}" --only-show-errors
  aws s3 sync "$dashboard_dir" "${s3_uri}/dashboard" --only-show-errors
}

render_dashboard running
sync_dashboard

set +e
"${cmd[@]}" > "$log_out" 2>&1 &
pid=$!
echo "$pid" > "${run_dir}/train.pid"

while kill -0 "$pid" >/dev/null 2>&1; do
  sleep "$sync_seconds"
  render_dashboard running
  sync_dashboard
done

wait "$pid"
exit_code=$?
set -e

if [[ "$exit_code" -eq 0 ]]; then
  render_dashboard succeeded "$exit_code"
  if [[ -n "${NSRL_PUBLISH_CHECKPOINT:-}" ]]; then
    checkpoint_name="$NSRL_PUBLISH_CHECKPOINT"
    checkpoint_uri="${s3_uri}/checkpoints/${checkpoint_name}"
    checkpoint_json="${run_dir}/checkpoint.json"
    CHECKPOINT_JSON="$checkpoint_json" \
    CHECKPOINT_NAME="$checkpoint_name" \
    CHECKPOINT_URI="$checkpoint_uri" \
    RUN_NAME="$run_name" \
    RUN_S3_URI="${s3_uri}/runs/${run_name}" \
    MODEL_S3_URI="${s3_uri}/runs/${run_name}/$(basename "$model_out")" \
    TRACE_S3_URI="${s3_uri}/runs/${run_name}/$(basename "$trace_out")" \
    PROGRESS_S3_URI="${s3_uri}/runs/${run_name}/$(basename "$progress_out")" \
    RUN_JSON_S3_URI="${s3_uri}/runs/${run_name}/run.json" \
    FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    python3 - <<'PY'
import json, os, pathlib
path = pathlib.Path(os.environ["CHECKPOINT_JSON"])
path.write_text(json.dumps({
    "schema": "nsrl.aws_checkpoint_pointer.v1",
    "checkpoint": os.environ["CHECKPOINT_NAME"],
    "checkpoint_s3_uri": os.environ["CHECKPOINT_URI"],
    "run_name": os.environ["RUN_NAME"],
    "run_s3_uri": os.environ["RUN_S3_URI"],
    "model_s3_uri": os.environ["MODEL_S3_URI"],
    "trace_s3_uri": os.environ["TRACE_S3_URI"],
    "progress_s3_uri": os.environ["PROGRESS_S3_URI"],
    "run_json_s3_uri": os.environ["RUN_JSON_S3_URI"],
    "finished_at": os.environ["FINISHED_AT"],
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    aws s3 cp "$model_out" "${checkpoint_uri}/latest.nsrlmt" --only-show-errors
    aws s3 cp "$trace_out" "${checkpoint_uri}/latest.trace.jsonl" --only-show-errors
    aws s3 cp "$progress_out" "${checkpoint_uri}/latest.progress.jsonl" --only-show-errors
    aws s3 cp "${run_dir}/run.json" "${checkpoint_uri}/latest.run.json" --only-show-errors
    aws s3 cp "$checkpoint_json" "${checkpoint_uri}/latest.checkpoint.json" --only-show-errors
  fi
else
  render_dashboard failed "$exit_code"
fi
sync_dashboard

exit "$exit_code"
