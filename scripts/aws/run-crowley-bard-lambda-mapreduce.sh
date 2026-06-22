#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

STAMP="${NSRL_RUN_STAMP:-$(date -u +"%Y%m%dT%H%M%SZ")}"
PROFILE="${NSRL_AWS_PROFILE:-${AWS_PROFILE:-staging}}"
REGION="${NSRL_AWS_REGION:-${AWS_REGION:-us-east-1}}"
S3_URI="${NSRL_S3_URI:-s3://nsrl-training-022118847419-us-east-1/wikibard}"
TOKENS_S3_URI="${NSRL_TOKENS_S3_URI:-s3://nsrl-training-022118847419-us-east-1/wikibard/corpus/datasets/crowley-bard-focused-v1/20260622T053753Z/tokens/crowley-bard-focused-v1.tokens.u8}"
RUN_NAME="${NSRL_RUN_NAME:-crowley-bard-lambda-mapreduce-64k-${STAMP}}"

WORKERS="${NSRL_SWARM_WORKERS:-${WORKERS:-4}}"
MAX_WINDOWS="${NSRL_MAX_WINDOWS:-${MAX_WINDOWS:-65536}}"
SEQ_LEN="${NSRL_SEQ_LEN:-${SEQ_LEN:-8}}"
STRIDE="${NSRL_STRIDE:-${STRIDE:-1}}"
BATCH_WINDOWS="${NSRL_BATCH_WINDOWS:-${BATCH_WINDOWS:-2}}"
MAP_REDUCE_WORKERS="${NSRL_MAP_REDUCE_WORKERS:-${MAP_REDUCE_WORKERS:-2}}"
MEMORY_MB="${NSRL_LAMBDA_MEMORY_MB:-${MEMORY_MB:-2048}}"
TIMEOUT_SECONDS="${NSRL_LAMBDA_TIMEOUT_SECONDS:-${TIMEOUT_SECONDS:-900}}"
PROGRESS_INTERVAL_BATCHES="${NSRL_PROGRESS_INTERVAL_BATCHES:-${PROGRESS_INTERVAL_BATCHES:-1024}}"

if [[ "${BUILD:-0}" == "1" ]]; then
  scripts/aws/build-lambda-swarm-worker.sh
fi

args=(
  node scripts/aws/run-lambda-swarm-comparison.mjs
  --run
  --profile "$PROFILE"
  --region "$REGION"
  --s3-uri "$S3_URI"
  --tokens-s3-uri "$TOKENS_S3_URI"
  --run-name "$RUN_NAME"
  --workers "$WORKERS"
  --max-windows "$MAX_WINDOWS"
  --seq-len "$SEQ_LEN"
  --stride "$STRIDE"
  --batch-windows "$BATCH_WINDOWS"
  --batch-mode map-reduce
  --map-reduce-workers "$MAP_REDUCE_WORKERS"
  --tokenizer ascii-lower
  --adaptive-rule-shifts 1
  --adaptive-rule-interval-batches 128
  --adaptive-holographic-shifts 0
  --progress-interval-batches "$PROGRESS_INTERVAL_BATCHES"
  --trace-detail none
  --memory-mb "$MEMORY_MB"
  --timeout-seconds "$TIMEOUT_SECONDS"
)

if [[ "${DEPLOY:-0}" == "1" ]]; then
  args+=(--deploy)
fi

if [[ "${ASSEMBLE:-1}" == "0" ]]; then
  args+=(--no-assemble)
fi

exec "${args[@]}" "$@"
