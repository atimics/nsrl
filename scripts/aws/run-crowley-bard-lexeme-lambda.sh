#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

STAMP="${NSRL_RUN_STAMP:-$(date -u +"%Y%m%dT%H%M%SZ")}"
PROFILE="${NSRL_AWS_PROFILE:-${AWS_PROFILE:-staging}}"
REGION="${NSRL_AWS_REGION:-${AWS_REGION:-us-east-1}}"
FUNCTION_NAME="${NSRL_FUNCTION_NAME:-nsrl-mini-transformer-swarm-worker}"
S3_URI="${NSRL_S3_URI:-s3://nsrl-training-022118847419-us-east-1/wikibard}"
RUN_NAME="${NSRL_RUN_NAME:-crowley-bard-lexeme-lambda-${STAMP}}"
RUN_ROOT="${NSRL_RUN_ROOT:-data/aws-lambda-lexeme/runs}"
RUN_DIR="$RUN_ROOT/$RUN_NAME"
OUTPUT_S3_PREFIX="${NSRL_OUTPUT_S3_PREFIX:-$S3_URI/lambda-runs/$RUN_NAME}"

BASE_RUN_DIR="${NSRL_LEXEME_BASE_RUN_DIR:-data/processed/visionary-twitter-bot-demo}"
TOKENS_LOCAL="${NSRL_LEXEME_TOKENS:-$BASE_RUN_DIR/v4096.tokens.u16}"
VOCAB_LOCAL="${NSRL_LEXEME_VOCAB:-$BASE_RUN_DIR/v4096.vocab.tsv}"
INPUT_S3_PREFIX="${NSRL_LEXEME_INPUT_S3_PREFIX:-$S3_URI/corpus/datasets/visionary-twitter-bot-demo/lexeme-v4096}"
TOKENS_S3_URI="${NSRL_TOKENS_S3_URI:-$INPUT_S3_PREFIX/v4096.tokens.u16}"
VOCAB_S3_URI="${NSRL_VOCAB_S3_URI:-$INPUT_S3_PREFIX/v4096.vocab.tsv}"
BASE_MODEL_LOCAL="${NSRL_LEXEME_BASE_MODEL:-}"
BASE_MODEL_S3_URI="${NSRL_BASE_MODEL_S3_URI:-}"

EMBEDDING_WINDOWS="${NSRL_EMBEDDING_WINDOWS:-131072}"
EMBEDDING_EPOCHS="${NSRL_EMBEDDING_EPOCHS:-1}"
SOFTMAX_WINDOWS="${NSRL_SOFTMAX_WINDOWS:-131072}"
SOFTMAX_EPOCHS="${NSRL_SOFTMAX_EPOCHS:-1}"
SOFTMAX_SEQ_LEN="${NSRL_SOFTMAX_SEQ_LEN:-8}"
SOFTMAX_LR_SHIFT="${NSRL_SOFTMAX_LR_SHIFT:-21}"
SOFTMAX_MAX_LR_SHIFT="${NSRL_SOFTMAX_MAX_LR_SHIFT:-23}"
SOFTMAX_LR_DECAY_WINDOWS="${NSRL_SOFTMAX_LR_DECAY_WINDOWS:-$((SOFTMAX_WINDOWS * SOFTMAX_EPOCHS / 2))}"
LEXEME_WORKERS="${NSRL_LEXEME_WORKERS:-1}"
WINDOW_OFFSET_BASE="${NSRL_WINDOW_OFFSET_BASE:-0}"
SAMPLE_MAX_NEW_TOKENS="${NSRL_SAMPLE_MAX_NEW_TOKENS:-96}"
POLL_SECONDS="${NSRL_POLL_SECONDS:-5}"
POLL_TIMEOUT_SECONDS="${NSRL_POLL_TIMEOUT_SECONDS:-900}"

mkdir -p "$RUN_DIR"

if ! [[ "$LEXEME_WORKERS" =~ ^[0-9]+$ ]] || (( LEXEME_WORKERS < 1 )); then
  echo "NSRL_LEXEME_WORKERS must be a positive integer" >&2
  exit 1
fi

if [[ "${BUILD:-0}" == "1" ]]; then
  scripts/aws/build-lambda-swarm-worker.sh
fi

if [[ "${DEPLOY:-0}" == "1" ]]; then
  node scripts/aws/run-lambda-swarm-comparison.mjs \
    --deploy \
    --profile "$PROFILE" \
    --region "$REGION" \
    --function-name "$FUNCTION_NAME" \
    --s3-uri "$S3_URI" \
    --tokens-s3-uri "$TOKENS_S3_URI" \
    --run-name "$RUN_NAME-deploy-only"
fi

if [[ "${UPLOAD_INPUTS:-1}" == "1" ]]; then
  aws --profile "$PROFILE" --region "$REGION" s3 cp "$TOKENS_LOCAL" "$TOKENS_S3_URI" --only-show-errors
  aws --profile "$PROFILE" --region "$REGION" s3 cp "$VOCAB_LOCAL" "$VOCAB_S3_URI" --only-show-errors
fi

if [[ -n "$BASE_MODEL_LOCAL" && -z "$BASE_MODEL_S3_URI" ]]; then
  BASE_MODEL_S3_URI="$OUTPUT_S3_PREFIX/inputs/base.nsrllm"
fi
if [[ -n "$BASE_MODEL_LOCAL" ]]; then
  aws --profile "$PROFILE" --region "$REGION" s3 cp "$BASE_MODEL_LOCAL" "$BASE_MODEL_S3_URI" --only-show-errors
fi

export PROFILE REGION FUNCTION_NAME RUN_NAME RUN_DIR OUTPUT_S3_PREFIX TOKENS_S3_URI VOCAB_S3_URI BASE_MODEL_S3_URI
export EMBEDDING_WINDOWS EMBEDDING_EPOCHS SOFTMAX_WINDOWS SOFTMAX_EPOCHS SOFTMAX_SEQ_LEN SOFTMAX_LR_SHIFT SOFTMAX_MAX_LR_SHIFT SOFTMAX_LR_DECAY_WINDOWS
export LEXEME_WORKERS WINDOW_OFFSET_BASE SAMPLE_MAX_NEW_TOKENS

node <<'NODE'
const fs = require("fs");
const path = require("path");

const runDir = process.env.RUN_DIR || "";
const workerCount = Number(process.env.LEXEME_WORKERS || 1);
if (!Number.isInteger(workerCount) || workerCount < 1) {
  throw new Error("LEXEME_WORKERS must be a positive integer");
}
fs.mkdirSync(runDir, { recursive: true });
const payloads = [];
for (let workerIndex = 0; workerIndex < workerCount; workerIndex += 1) {
  const workerId = `worker-${String(workerIndex).padStart(3, "0")}`;
  const windowOffsetBase = Number(process.env.WINDOW_OFFSET_BASE || 0);
  const payload = {
    job_kind: "lexeme-crowley",
    run_name: process.env.RUN_NAME,
    worker_index: workerIndex,
    worker_count: workerCount,
    output_s3_prefix: process.env.OUTPUT_S3_PREFIX,
    tokens_s3_uri: process.env.TOKENS_S3_URI,
    vocab_s3_uri: process.env.VOCAB_S3_URI,
    base_model_s3_uri: process.env.BASE_MODEL_S3_URI || "",
    config: {
      vocab_size: 4096,
      embedding_dim: 16,
      frequency_cap: 4096,
      embedding_windows: Number(process.env.EMBEDDING_WINDOWS),
      embedding_epochs: Number(process.env.EMBEDDING_EPOCHS),
      softmax_windows: Number(process.env.SOFTMAX_WINDOWS),
      softmax_epochs: Number(process.env.SOFTMAX_EPOCHS),
      softmax_lr_shift: Number(process.env.SOFTMAX_LR_SHIFT),
      softmax_max_lr_shift: Number(process.env.SOFTMAX_MAX_LR_SHIFT),
      softmax_lr_decay_windows: Number(process.env.SOFTMAX_LR_DECAY_WINDOWS),
      softmax_seq_len: Number(process.env.SOFTMAX_SEQ_LEN),
      stride: workerCount,
      window_offset: windowOffsetBase + workerIndex,
      lexeme_context_features: "mean",
      quality_weight_profile: "cruft-aware",
      corpus_prior_order: 2,
      corpus_prior_logit_shift: 7,
      repeat_window: 80,
      repeat_penalty_shift: 3,
      max_repeat_run: 2,
      no_repeat_ngram: 3,
      sample_max_new_tokens: Number(process.env.SAMPLE_MAX_NEW_TOKENS),
    },
  };
  const payloadPath = path.join(runDir, `payload-${workerId}.json`);
  fs.writeFileSync(payloadPath, `${JSON.stringify(payload, null, 2)}\n`);
  if (workerIndex === 0) {
    fs.writeFileSync(path.join(runDir, "payload.json"), `${JSON.stringify(payload, null, 2)}\n`);
  }
  payloads.push({ workerId, payloadPath, payload });
}
fs.writeFileSync(
  path.join(runDir, "run-options.json"),
  `${JSON.stringify({
    profile: process.env.PROFILE,
    region: process.env.REGION,
    functionName: process.env.FUNCTION_NAME,
    runName: process.env.RUN_NAME,
    outputS3Prefix: process.env.OUTPUT_S3_PREFIX,
    tokensS3Uri: process.env.TOKENS_S3_URI,
    vocabS3Uri: process.env.VOCAB_S3_URI,
    baseModelS3Uri: process.env.BASE_MODEL_S3_URI || "",
    windowOffsetBase: Number(process.env.WINDOW_OFFSET_BASE || 0),
    workerCount,
    payloads,
  }, null, 2)}\n`,
);
NODE

aws --profile "$PROFILE" --region "$REGION" s3 cp "$RUN_DIR/run-options.json" "$OUTPUT_S3_PREFIX/run-options.json" --only-show-errors

mkdir -p "$RUN_DIR/workers" "$RUN_DIR/samples"
for ((worker_index = 0; worker_index < LEXEME_WORKERS; worker_index++)); do
  printf -v worker_id "worker-%03d" "$worker_index"
  aws --profile "$PROFILE" --region "$REGION" lambda invoke \
    --function-name "$FUNCTION_NAME" \
    --invocation-type Event \
    --cli-binary-format raw-in-base64-out \
    --payload "file://$RUN_DIR/payload-$worker_id.json" \
    "$RUN_DIR/invoke-$worker_id.json" >/dev/null
done

deadline=$((SECONDS + POLL_TIMEOUT_SECONDS))
for ((worker_index = 0; worker_index < LEXEME_WORKERS; worker_index++)); do
  printf -v worker_id "worker-%03d" "$worker_index"
  summary_uri="$OUTPUT_S3_PREFIX/workers/$worker_id.summary.json"
  summary_path="$RUN_DIR/workers/$worker_id.summary.json"
  until aws --profile "$PROFILE" --region "$REGION" s3 cp "$summary_uri" "$summary_path" --only-show-errors 2>/dev/null; do
    if (( SECONDS >= deadline )); then
      echo "timed out waiting for $summary_uri" >&2
      exit 1
    fi
    sleep "$POLL_SECONDS"
  done

  if ! jq -e '.ok == true' "$summary_path" >/dev/null; then
    cat "$summary_path" >&2
    exit 1
  fi

  suffixes=(nsrllm softmax.trace.jsonl stdout.txt)
  if [[ -z "$BASE_MODEL_S3_URI" ]]; then
    suffixes+=(nsrllex embedding.trace.jsonl)
  fi
  for suffix in "${suffixes[@]}"; do
    key="$worker_id.$suffix"
    aws --profile "$PROFILE" --region "$REGION" s3 cp "$OUTPUT_S3_PREFIX/workers/$key" "$RUN_DIR/workers/$key" --only-show-errors
  done

  jq -r '.samples[]? | .text_s3_uri, .trace_s3_uri' "$summary_path" | while IFS= read -r uri; do
    [[ -n "$uri" ]] || continue
    aws --profile "$PROFILE" --region "$REGION" s3 cp "$uri" "$RUN_DIR/samples/$(basename "$uri")" --only-show-errors
  done
done

final_model="$RUN_DIR/workers/worker-000.nsrllm"
if (( LEXEME_WORKERS > 1 )); then
  reduced_model="$RUN_DIR/reduced.nsrllm"
  reduced_trace="$RUN_DIR/reduced.trace.jsonl"
  reduce_args=(
    --mode lexeme-reduce
    --model "$RUN_DIR/workers/worker-000.nsrllm"
    --model-out "$reduced_model"
    --trace "$reduced_trace"
  )
  for ((worker_index = 1; worker_index < LEXEME_WORKERS; worker_index++)); do
    printf -v worker_id "worker-%03d" "$worker_index"
    reduce_args+=(--expert "$RUN_DIR/workers/$worker_id.nsrllm")
  done
  cargo run --release -q -p nsrl-train -- "${reduce_args[@]}"
  aws --profile "$PROFILE" --region "$REGION" s3 cp "$reduced_model" "$OUTPUT_S3_PREFIX/reduced.nsrllm" --only-show-errors
  aws --profile "$PROFILE" --region "$REGION" s3 cp "$reduced_trace" "$OUTPUT_S3_PREFIX/reduced.trace.jsonl" --only-show-errors
  final_model="$reduced_model"
fi

echo "run_name=$RUN_NAME"
echo "worker_count=$LEXEME_WORKERS"
echo "local_run_dir=$RUN_DIR"
echo "output_s3_prefix=$OUTPUT_S3_PREFIX"
echo "model=$final_model"
echo "summaries=$RUN_DIR/workers"
