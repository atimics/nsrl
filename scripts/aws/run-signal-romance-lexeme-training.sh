#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'USAGE'
Run Signal romance lexeme training on an EC2 instance and publish the S3 dashboard.

Required:
  NSRL_S3_URI=s3://bucket/prefix
  NSRL_LEXEME_TOKENS_S3_URI=s3://bucket/path/v512.tokens.u16
  NSRL_LEXEME_VOCAB_S3_URI=s3://bucket/path/v512.vocab.tsv
  NSRL_LEXEME_PROMPTS_S3_URI=s3://bucket/path/eval-prompts.jsonl

Common knobs:
  NSRL_RUN_NAME=signal-romance-sim-lexeme-001
  NSRL_RUN_ROOT=/mnt/nsrl/aws-runs
  NSRL_EMBED_DIM=16
  NSRL_EMBED_WINDOWS=65536
  NSRL_SOFTMAX_SEQ_LEN=16
  NSRL_SOFTMAX_WINDOWS=250000
  NSRL_SOFTMAX_EPOCHS=3
  NSRL_PUBLISH_CHECKPOINT=signal-romance-sim-lexeme
  NSRL_SYNC_SECONDS=30
  NSRL_TERMINATE_ON_EXIT=1
USAGE
  exit 0
fi

if [[ -z "${NSRL_S3_URI:-}" ]]; then
  echo "NSRL_S3_URI is required" >&2
  exit 2
fi

for tool in aws python3 cargo node; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "$tool is required" >&2
    exit 2
  fi
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
run_name="${NSRL_RUN_NAME:-signal-romance-lexeme-${timestamp}}"
run_root="${NSRL_RUN_ROOT:-/mnt/nsrl/aws-runs}"
run_dir="${run_root}/${run_name}"
dashboard_dir="${run_root}/dashboard"
input_dir="${run_dir}/input"
eval_dir="${run_dir}/lexeme-eval"
mkdir -p "$run_dir" "$dashboard_dir" "$input_dir" "$eval_dir"

s3_uri="${NSRL_S3_URI%/}"
tokens="${NSRL_LEXEME_TOKENS:-${run_dir}/tokens.u16}"
vocab="${NSRL_LEXEME_VOCAB:-${run_dir}/vocab.tsv}"
prompts="${NSRL_LEXEME_PROMPTS:-${run_dir}/eval-prompts.jsonl}"
corpus="${NSRL_LEXEME_CORPUS:-${run_dir}/corpus.txt}"
manifest="${NSRL_LEXEME_MANIFEST:-${run_dir}/manifest.json}"
frames="${NSRL_LEXEME_FRAMES:-${run_dir}/frames.jsonl}"
embedding="${run_dir}/${run_name}.nsrllex"
embedding_trace="${run_dir}/${run_name}.embedding.trace.jsonl"
model_out="${run_dir}/${run_name}.nsrllm"
softmax_trace="${run_dir}/${run_name}.softmax.trace.jsonl"
eval_report="${run_dir}/eval-report.json"
eval_summary="${run_dir}/eval-summary.tsv"
log_out="${run_dir}/train.log"
command_out="${run_dir}/command.txt"
repo_rev="$(git rev-parse HEAD 2>/dev/null || true)"
sync_seconds="${NSRL_SYNC_SECONDS:-30}"
run_stage="preparing"

aws_instance_id="${NSRL_AWS_INSTANCE_ID:-}"
aws_instance_type="${NSRL_AWS_INSTANCE_TYPE:-}"
aws_instance_region="${AWS_REGION:-${AWS_DEFAULT_REGION:-}}"
aws_instance_az="${NSRL_AWS_AVAILABILITY_ZONE:-}"
aws_instance_launch_time="${NSRL_AWS_INSTANCE_LAUNCH_TIME:-}"
cost_hourly_usd="${NSRL_INSTANCE_HOURLY_USD:-}"
cost_currency="${NSRL_COST_CURRENCY:-USD}"

load_ec2_metadata() {
  if ! command -v curl >/dev/null 2>&1; then
    return 0
  fi
  local token identity_doc
  token="$(
    curl -fsS --connect-timeout 2 --max-time 5 \
      -X PUT \
      -H "X-aws-ec2-metadata-token-ttl-seconds: 60" \
      http://169.254.169.254/latest/api/token 2>/dev/null
  )" || return 0
  aws_instance_id="${aws_instance_id:-$(
    curl -fsS --connect-timeout 2 --max-time 5 \
      -H "X-aws-ec2-metadata-token: ${token}" \
      http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || true
  )}"
  aws_instance_type="${aws_instance_type:-$(
    curl -fsS --connect-timeout 2 --max-time 5 \
      -H "X-aws-ec2-metadata-token: ${token}" \
      http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || true
  )}"
  aws_instance_az="${aws_instance_az:-$(
    curl -fsS --connect-timeout 2 --max-time 5 \
      -H "X-aws-ec2-metadata-token: ${token}" \
      http://169.254.169.254/latest/meta-data/placement/availability-zone 2>/dev/null || true
  )}"
  identity_doc="$(
    curl -fsS --connect-timeout 2 --max-time 5 \
      -H "X-aws-ec2-metadata-token: ${token}" \
      http://169.254.169.254/latest/dynamic/instance-identity/document 2>/dev/null || true
  )"
  if [[ -n "$identity_doc" ]]; then
    read -r parsed_region parsed_launch_time < <(
      IDENTITY_DOC="$identity_doc" python3 - <<'PY'
import json
import os
doc = json.loads(os.environ["IDENTITY_DOC"])
print(doc.get("region", ""), doc.get("pendingTime", ""))
PY
    )
    aws_instance_region="${aws_instance_region:-$parsed_region}"
    aws_instance_launch_time="${aws_instance_launch_time:-$parsed_launch_time}"
  fi
}

download_if_needed() {
  local label="$1"
  local local_path="$2"
  local s3_path="$3"
  if [[ -s "$local_path" ]]; then
    return 0
  fi
  if [[ -z "$s3_path" ]]; then
    echo "$label missing locally and no S3 URI was provided: $local_path" >&2
    exit 2
  fi
  mkdir -p "$(dirname "$local_path")"
  aws s3 cp "$s3_path" "$local_path" --only-show-errors
}

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
    --stage "$run_stage"
    --started-at "$started_at"
    --updated-at "$updated_at"
    --repo-rev "$repo_rev"
    --tokens "$tokens"
    --instance-id "$aws_instance_id"
    --instance-type "$aws_instance_type"
    --instance-region "$aws_instance_region"
    --instance-availability-zone "$aws_instance_az"
    --instance-launch-time "$aws_instance_launch_time"
    --cost-hourly-usd "$cost_hourly_usd"
    --cost-currency "$cost_currency"
    --command-file "$command_out"
    --log-file "$log_out"
    --progress-file "$embedding_trace"
    --trace-file "$softmax_trace"
    --model-file "$model_out"
    --vocab-file "$vocab"
    --eval-report-file "$eval_report"
    --eval-summary-file "$eval_summary"
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

run_logged_step() {
  run_stage="$1"
  shift
  render_dashboard running
  sync_dashboard
  printf '\n[%s] stage=%s command=' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$run_stage" >> "$log_out"
  printf '%q ' "$@" >> "$log_out"
  printf '\n' >> "$log_out"
  set +e
  "$@" >> "$log_out" 2>&1 &
  local pid=$!
  while kill -0 "$pid" >/dev/null 2>&1; do
    sleep "$sync_seconds"
    render_dashboard running
    sync_dashboard
  done
  wait "$pid"
  local exit_code=$?
  set -e
  if [[ "$exit_code" -ne 0 ]]; then
    run_stage="failed:${run_stage}"
    render_dashboard failed "$exit_code"
    sync_dashboard
    terminate_instance_if_requested
    exit "$exit_code"
  fi
}

terminate_instance_if_requested() {
  if [[ "${NSRL_TERMINATE_ON_EXIT:-0}" == "0" ]]; then
    return 0
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo "NSRL_TERMINATE_ON_EXIT requested, but curl is unavailable" >&2
    return 0
  fi

  local token instance_id region identity_doc
  token="$(
    curl -fsS --connect-timeout 2 --max-time 5 \
      -X PUT \
      -H "X-aws-ec2-metadata-token-ttl-seconds: 60" \
      http://169.254.169.254/latest/api/token 2>/dev/null
  )" || {
    echo "NSRL_TERMINATE_ON_EXIT requested, but EC2 metadata is unavailable" >&2
    return 0
  }
  instance_id="$(
    curl -fsS --connect-timeout 2 --max-time 5 \
      -H "X-aws-ec2-metadata-token: ${token}" \
      http://169.254.169.254/latest/meta-data/instance-id
  )" || return 0
  region="${AWS_REGION:-${AWS_DEFAULT_REGION:-}}"
  if [[ -z "$region" ]]; then
    identity_doc="$(
      curl -fsS --connect-timeout 2 --max-time 5 \
        -H "X-aws-ec2-metadata-token: ${token}" \
        http://169.254.169.254/latest/dynamic/instance-identity/document
    )" || return 0
    region="$(
      IDENTITY_DOC="$identity_doc" python3 - <<'PY'
import json
import os
print(json.loads(os.environ["IDENTITY_DOC"])["region"])
PY
    )" || return 0
  fi

  echo "NSRL_TERMINATE_ON_EXIT terminating ${instance_id} in ${region}" >&2
  if ! aws ec2 terminate-instances \
    --region "$region" \
    --instance-ids "$instance_id" \
    --query 'TerminatingInstances[0].{InstanceId:InstanceId,Current:CurrentState.Name}' \
    --output json >&2; then
    echo "EC2 terminate API failed; falling back to instance-initiated shutdown" >&2
    shutdown -h now || true
  fi
}

load_ec2_metadata
aws s3 cp "${s3_uri}/dashboard/runs.json" "${dashboard_dir}/runs.json" >/dev/null 2>&1 || true

download_if_needed "tokens" "$tokens" "${NSRL_LEXEME_TOKENS_S3_URI:-}"
download_if_needed "vocab" "$vocab" "${NSRL_LEXEME_VOCAB_S3_URI:-}"
download_if_needed "eval prompts" "$prompts" "${NSRL_LEXEME_PROMPTS_S3_URI:-}"
if [[ -n "${NSRL_LEXEME_CORPUS_S3_URI:-}" ]]; then
  download_if_needed "corpus" "$corpus" "$NSRL_LEXEME_CORPUS_S3_URI"
fi
if [[ -n "${NSRL_LEXEME_MANIFEST_S3_URI:-}" ]]; then
  download_if_needed "manifest" "$manifest" "$NSRL_LEXEME_MANIFEST_S3_URI"
fi
if [[ -n "${NSRL_LEXEME_FRAMES_S3_URI:-}" ]]; then
  download_if_needed "frames" "$frames" "$NSRL_LEXEME_FRAMES_S3_URI"
fi

vocab_size="$(
  awk -F '\t' 'NR > 1 && $1 ~ /^[0-9]+$/ { id = $1 } END { print id ? id + 1 : 256 }' "$vocab"
)"

embed_cmd=(
  target/release/nsrl-train
  --mode lexeme-embedding
  --tokens "$tokens"
  --vocab "$vocab"
  --model-out "$embedding"
  --trace "$embedding_trace"
  --vocab-size "$vocab_size"
  --embedding-dim "${NSRL_EMBED_DIM:-16}"
  --context-radius "${NSRL_EMBED_CONTEXT_RADIUS:-2}"
  --stride "${NSRL_EMBED_STRIDE:-1}"
  --max-windows "${NSRL_EMBED_WINDOWS:-65536}"
  --epochs "${NSRL_EMBED_EPOCHS:-1}"
  --lr-shift "${NSRL_EMBED_LR_SHIFT:-8}"
  --concept-frequency-cap "${NSRL_FREQ_CAP:-4096}"
  --frequency-weight-min-q15 "${NSRL_FREQUENCY_WEIGHT_MIN_Q15:-4096}"
  --quality-weight-profile "${NSRL_QUALITY_WEIGHT_PROFILE:-cruft-aware}"
)

softmax_cmd=(
  target/release/nsrl-train
  --mode lexeme-softmax
  --tokens "$tokens"
  --vocab "$vocab"
  --model "$embedding"
  --model-out "$model_out"
  --trace "$softmax_trace"
  --seq-len "${NSRL_SOFTMAX_SEQ_LEN:-16}"
  --lexeme-context-features "${NSRL_LEXEME_CONTEXT_FEATURES:-ordered}"
  --stride "${NSRL_SOFTMAX_STRIDE:-1}"
  --max-windows "${NSRL_SOFTMAX_WINDOWS:-250000}"
  --epochs "${NSRL_SOFTMAX_EPOCHS:-3}"
  --lr-shift "${NSRL_SOFTMAX_LR_SHIFT:-18}"
  --lr-shift-decay-windows "${NSRL_SOFTMAX_LR_DECAY_WINDOWS:-$(( ${NSRL_SOFTMAX_WINDOWS:-250000} / 2 ))}"
  --lr-shift-decay-step "${NSRL_SOFTMAX_LR_DECAY_STEP:-1}"
  --max-lr-shift "${NSRL_SOFTMAX_MAX_LR_SHIFT:-23}"
  --max-weight-delta "${NSRL_SOFTMAX_MAX_WEIGHT_DELTA:-1}"
  --target-frequency-cap "${NSRL_FREQ_CAP:-4096}"
  --frequency-weight-min-q15 "${NSRL_FREQUENCY_WEIGHT_MIN_Q15:-4096}"
  --quality-weight-profile "${NSRL_QUALITY_WEIGHT_PROFILE:-cruft-aware}"
)

if [[ "${NSRL_TRAIN_LEXEME_EMBEDDINGS:-0}" != "0" ]]; then
  softmax_cmd+=(--train-lexeme-embeddings)
fi
if [[ -n "${NSRL_LEXEME_HIDDEN_DIM:-}" ]]; then
  softmax_cmd+=(--lexeme-hidden-dim "$NSRL_LEXEME_HIDDEN_DIM")
fi
if [[ -n "${NSRL_LEXEME_HIDDEN_LR_SHIFT:-}" ]]; then
  softmax_cmd+=(--lexeme-hidden-lr-shift "$NSRL_LEXEME_HIDDEN_LR_SHIFT")
fi
if [[ -n "${NSRL_LEXEME_ADAPTER_LOGIT_SHIFT:-}" ]]; then
  softmax_cmd+=(--lexeme-adapter-logit-shift "$NSRL_LEXEME_ADAPTER_LOGIT_SHIFT")
fi

eval_cmd=(
  node
  scripts/eval-signal-romance.mjs
  --backend lexeme
  --run-dir "$run_dir"
  --model "$model_out"
  --vocab "$vocab"
  --prompts "$prompts"
  --out-dir "$eval_dir"
  --count "${NSRL_LEXEME_EVAL_COUNT:-40}"
  --max-new-tokens "${NSRL_LEXEME_MAX_NEW_TOKENS:-20}"
)

{
  printf 'vocab_size=%s\n' "$vocab_size"
  printf 'embedding=%s\n' "$embedding"
  printf 'model=%s\n' "$model_out"
  printf 'tokens=%s\n' "$tokens"
  printf 'vocab=%s\n' "$vocab"
  printf 'prompts=%s\n' "$prompts"
  printf 'embed_command='
  printf '%q ' "${embed_cmd[@]}"
  printf '\nsoftmax_command='
  printf '%q ' "${softmax_cmd[@]}"
  printf '\neval_command='
  printf '%q ' "${eval_cmd[@]}"
  printf '\n'
} > "$command_out"

: > "$log_out"
render_dashboard running
sync_dashboard

if [[ -n "${NSRL_RUSTFLAGS:-}" ]]; then
  export RUSTFLAGS="$NSRL_RUSTFLAGS"
fi
export NSRL_TRAIN_BIN=target/release/nsrl-train

run_logged_step building cargo build --release -p nsrl-train
run_logged_step embedding "${embed_cmd[@]}"
run_logged_step softmax "${softmax_cmd[@]}"
if ! command -v node >/dev/null 2>&1; then
  if command -v dnf >/dev/null 2>&1; then
    run_logged_step installing-node dnf install -y nodejs
  else
    echo "node is required for eval; install node or set a runner image with nodejs" >&2
    exit 2
  fi
fi
run_logged_step evaluating "${eval_cmd[@]}"
cp "${eval_dir}/eval-report.json" "$eval_report"
cp "${eval_dir}/eval-summary.tsv" "$eval_summary"

run_stage="completed"
render_dashboard succeeded 0

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
  VOCAB_S3_URI="${s3_uri}/runs/${run_name}/$(basename "$vocab")" \
  TOKENS_S3_URI="${s3_uri}/runs/${run_name}/$(basename "$tokens")" \
  TRACE_S3_URI="${s3_uri}/runs/${run_name}/$(basename "$softmax_trace")" \
  EVAL_REPORT_S3_URI="${s3_uri}/runs/${run_name}/$(basename "$eval_report")" \
  EVAL_SUMMARY_S3_URI="${s3_uri}/runs/${run_name}/$(basename "$eval_summary")" \
  RUN_JSON_S3_URI="${s3_uri}/runs/${run_name}/run.json" \
  FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  python3 - <<'PY'
import json
import os
import pathlib
path = pathlib.Path(os.environ["CHECKPOINT_JSON"])
path.write_text(json.dumps({
    "schema": "nsrl.aws_lexeme_checkpoint_pointer.v1",
    "checkpoint": os.environ["CHECKPOINT_NAME"],
    "checkpoint_s3_uri": os.environ["CHECKPOINT_URI"],
    "run_name": os.environ["RUN_NAME"],
    "run_s3_uri": os.environ["RUN_S3_URI"],
    "model_s3_uri": os.environ["MODEL_S3_URI"],
    "vocab_s3_uri": os.environ["VOCAB_S3_URI"],
    "tokens_s3_uri": os.environ["TOKENS_S3_URI"],
    "trace_s3_uri": os.environ["TRACE_S3_URI"],
    "eval_report_s3_uri": os.environ["EVAL_REPORT_S3_URI"],
    "eval_summary_s3_uri": os.environ["EVAL_SUMMARY_S3_URI"],
    "run_json_s3_uri": os.environ["RUN_JSON_S3_URI"],
    "finished_at": os.environ["FINISHED_AT"],
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  aws s3 cp "$model_out" "${checkpoint_uri}/latest.nsrllm" --only-show-errors
  aws s3 cp "$vocab" "${checkpoint_uri}/latest.vocab.tsv" --only-show-errors
  aws s3 cp "$tokens" "${checkpoint_uri}/latest.tokens.u16" --only-show-errors
  aws s3 cp "$softmax_trace" "${checkpoint_uri}/latest.trace.jsonl" --only-show-errors
  aws s3 cp "$eval_report" "${checkpoint_uri}/latest.eval-report.json" --only-show-errors
  aws s3 cp "$eval_summary" "${checkpoint_uri}/latest.eval-summary.tsv" --only-show-errors
  aws s3 cp "${run_dir}/run.json" "${checkpoint_uri}/latest.run.json" --only-show-errors
  aws s3 cp "$checkpoint_json" "${checkpoint_uri}/latest.checkpoint.json" --only-show-errors
fi

sync_dashboard
terminate_instance_if_requested
