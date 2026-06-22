#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_MODEL_ROOT="$REPO_ROOT/data/processed/crowley-bard-aphorism-v2"
DEFAULT_MODEL_PATH="$DEFAULT_MODEL_ROOT/experiments/v4096.seq8-mean-reduce-base15-lr25-o98304.nsrllm"
DEFAULT_VOCAB_PATH="$DEFAULT_MODEL_ROOT/v4096.vocab.tsv"
DEFAULT_TOKENS_PATH="$DEFAULT_MODEL_ROOT/v4096.tokens.u16"
MODEL_DIR="${X_BOT_MODEL_DIR:-}"
MODEL_PATH="${X_BOT_MODEL_PATH:-${MODEL_DIR:+$MODEL_DIR/v4096.nsrllm}}"
VOCAB_PATH="${X_BOT_VOCAB_PATH:-${MODEL_DIR:+$MODEL_DIR/v4096.vocab.tsv}}"
TOKENS_PATH="${X_BOT_TOKENS_PATH:-${MODEL_DIR:+$MODEL_DIR/v4096.tokens.u16}}"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
VOCAB_PATH="${VOCAB_PATH:-$DEFAULT_VOCAB_PATH}"
TOKENS_PATH="${TOKENS_PATH:-$DEFAULT_TOKENS_PATH}"
MODEL_S3_URI="${X_BOT_MODEL_S3_URI:-}"

if [[ -z "$MODEL_S3_URI" ]]; then
  echo "Set X_BOT_MODEL_S3_URI, for example s3://bucket/prefix/crowley-bard/model" >&2
  exit 1
fi

for artifact in "$MODEL_PATH" "$VOCAB_PATH" "$TOKENS_PATH"; do
  if [[ ! -f "$artifact" ]]; then
    echo "missing model artifact: $artifact" >&2
    exit 1
  fi
done

aws s3 cp "$MODEL_PATH" "$MODEL_S3_URI/v4096.nsrllm"
aws s3 cp "$VOCAB_PATH" "$MODEL_S3_URI/v4096.vocab.tsv"
aws s3 cp "$TOKENS_PATH" "$MODEL_S3_URI/v4096.tokens.u16"

printf 'synced Crowley model bundle to %s\n' "$MODEL_S3_URI"
