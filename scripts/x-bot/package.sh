#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${1:-$SCRIPT_DIR/build/crowley-bard-mention-lambda.zip}"
BIN="${X_BOT_LAMBDA_BIN:-$SCRIPT_DIR/build/bin/nsrl-train}"
CORPUS_BIN="${X_BOT_LAMBDA_CORPUS_BIN:-$SCRIPT_DIR/build/bin/nsrl-corpus}"
MODEL_DIR="${X_BOT_MODEL_DIR:-$REPO_ROOT/data/processed/visionary-twitter-bot-demo}"

mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"
rm -rf "$SCRIPT_DIR/build/package"
mkdir -p "$SCRIPT_DIR/build/package/bin" "$SCRIPT_DIR/build/package/model"

cp "$SCRIPT_DIR/lambda_function.py" "$SCRIPT_DIR/build/package/lambda_function.py"

if [[ ! -x "$BIN" ]]; then
  echo "missing executable Lambda nsrl-train binary: $BIN" >&2
  echo "run scripts/x-bot/build-lambda-binary.sh first" >&2
  exit 1
fi
if [[ ! -x "$CORPUS_BIN" ]]; then
  echo "missing executable Lambda nsrl-corpus binary: $CORPUS_BIN" >&2
  echo "run scripts/x-bot/build-lambda-binary.sh first" >&2
  exit 1
fi

for artifact in v4096.nsrllm v4096.vocab.tsv v4096.tokens.u16; do
  if [[ ! -f "$MODEL_DIR/$artifact" ]]; then
    echo "missing model artifact: $MODEL_DIR/$artifact" >&2
    exit 1
  fi
done

cp "$BIN" "$SCRIPT_DIR/build/package/bin/nsrl-train"
cp "$CORPUS_BIN" "$SCRIPT_DIR/build/package/bin/nsrl-corpus"
chmod +x "$SCRIPT_DIR/build/package/bin/nsrl-train"
chmod +x "$SCRIPT_DIR/build/package/bin/nsrl-corpus"
cp "$MODEL_DIR/v4096.nsrllm" "$SCRIPT_DIR/build/package/model/v4096.nsrllm"
cp "$MODEL_DIR/v4096.vocab.tsv" "$SCRIPT_DIR/build/package/model/v4096.vocab.tsv"
cp "$MODEL_DIR/v4096.tokens.u16" "$SCRIPT_DIR/build/package/model/v4096.tokens.u16"

(
  cd "$SCRIPT_DIR/build/package"
  zip -q -r "$OUT" .
)

printf '%s\n' "$OUT"
