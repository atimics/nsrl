#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_DIR="${X_BOT_MODEL_DIR:-$REPO_ROOT/data/processed/visionary-twitter-bot-demo}"
MODEL_S3_URI="${X_BOT_MODEL_S3_URI:-}"

if [[ -z "$MODEL_S3_URI" ]]; then
  echo "Set X_BOT_MODEL_S3_URI, for example s3://bucket/prefix/crowley-bard/model" >&2
  exit 1
fi

for artifact in v4096.nsrllm v4096.vocab.tsv v4096.tokens.u16; do
  if [[ ! -f "$MODEL_DIR/$artifact" ]]; then
    echo "missing model artifact: $MODEL_DIR/$artifact" >&2
    exit 1
  fi
done

aws s3 cp "$MODEL_DIR/v4096.nsrllm" "$MODEL_S3_URI/v4096.nsrllm"
aws s3 cp "$MODEL_DIR/v4096.vocab.tsv" "$MODEL_S3_URI/v4096.vocab.tsv"
aws s3 cp "$MODEL_DIR/v4096.tokens.u16" "$MODEL_S3_URI/v4096.tokens.u16"

printf 'synced Crowley model bundle to %s\n' "$MODEL_S3_URI"
