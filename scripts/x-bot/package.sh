#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="${1:-$SCRIPT_DIR/build/crowley-bard-mention-lambda.zip}"
BIN="${X_BOT_LAMBDA_BIN:-$SCRIPT_DIR/build/bin/nsrl-train}"
CORPUS_BIN="${X_BOT_LAMBDA_CORPUS_BIN:-$SCRIPT_DIR/build/bin/nsrl-corpus}"
SIGIL_BIN="${X_BOT_LAMBDA_SIGIL_BIN:-$SCRIPT_DIR/build/bin/nsrl-bitmap-sample}"
DEFAULT_MODEL_ROOT="$REPO_ROOT/data/processed/crowley-bard-aphorism-v2"
DEFAULT_MODEL_PATH="$DEFAULT_MODEL_ROOT/experiments/v4096.seq8-mean-reduce-base15-lr25-o98304.nsrllm"
DEFAULT_VOCAB_PATH="$DEFAULT_MODEL_ROOT/v4096.vocab.tsv"
DEFAULT_TOKENS_PATH="$DEFAULT_MODEL_ROOT/v4096.tokens.u16"
DEFAULT_SIGIL_MODEL_PATH="$REPO_ROOT/web/assets/solomon-model.nsrltch"
DEFAULT_SIGIL_TEXT_INDEX_PATH="$REPO_ROOT/web/assets/solomon-spirit-text-signatures.tsv"
DEFAULT_SIGIL_LATENT_MODEL_PATH="$REPO_ROOT/data/processed/key-solomon-goetia-latent-v1/scaling-curve/n576-ld32-tf512-e12/model.nsrllat"
MODEL_DIR="${X_BOT_MODEL_DIR:-}"
MODEL_PATH="${X_BOT_MODEL_PATH:-${MODEL_DIR:+$MODEL_DIR/v4096.nsrllm}}"
VOCAB_PATH="${X_BOT_VOCAB_PATH:-${MODEL_DIR:+$MODEL_DIR/v4096.vocab.tsv}}"
TOKENS_PATH="${X_BOT_TOKENS_PATH:-${MODEL_DIR:+$MODEL_DIR/v4096.tokens.u16}}"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
VOCAB_PATH="${VOCAB_PATH:-$DEFAULT_VOCAB_PATH}"
TOKENS_PATH="${TOKENS_PATH:-$DEFAULT_TOKENS_PATH}"
SIGIL_MODEL_PATH="${X_BOT_SIGIL_MODEL_PATH:-$DEFAULT_SIGIL_MODEL_PATH}"
SIGIL_TEXT_INDEX_PATH="${X_BOT_SIGIL_TEXT_INDEX_PATH:-$DEFAULT_SIGIL_TEXT_INDEX_PATH}"
SIGIL_LATENT_MODEL_PATH="${X_BOT_SIGIL_LATENT_MODEL_PATH:-$DEFAULT_SIGIL_LATENT_MODEL_PATH}"

mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"
rm -rf "$SCRIPT_DIR/build/package"
mkdir -p "$SCRIPT_DIR/build/package/bin" "$SCRIPT_DIR/build/package/model" "$SCRIPT_DIR/build/package/solomon"

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
if [[ ! -x "$SIGIL_BIN" ]]; then
  echo "missing executable Lambda nsrl-bitmap-sample binary: $SIGIL_BIN" >&2
  echo "run scripts/x-bot/build-lambda-binary.sh first" >&2
  exit 1
fi

for artifact in "$MODEL_PATH" "$VOCAB_PATH" "$TOKENS_PATH"; do
  if [[ ! -f "$artifact" ]]; then
    echo "missing model artifact: $artifact" >&2
    exit 1
  fi
done
for artifact in "$SIGIL_MODEL_PATH" "$SIGIL_TEXT_INDEX_PATH"; do
  if [[ ! -f "$artifact" ]]; then
    echo "missing Solomon artifact: $artifact" >&2
    exit 1
  fi
done

cp "$BIN" "$SCRIPT_DIR/build/package/bin/nsrl-train"
cp "$CORPUS_BIN" "$SCRIPT_DIR/build/package/bin/nsrl-corpus"
cp "$SIGIL_BIN" "$SCRIPT_DIR/build/package/bin/nsrl-bitmap-sample"
chmod +x "$SCRIPT_DIR/build/package/bin/nsrl-train"
chmod +x "$SCRIPT_DIR/build/package/bin/nsrl-corpus"
chmod +x "$SCRIPT_DIR/build/package/bin/nsrl-bitmap-sample"
cp "$MODEL_PATH" "$SCRIPT_DIR/build/package/model/v4096.nsrllm"
cp "$VOCAB_PATH" "$SCRIPT_DIR/build/package/model/v4096.vocab.tsv"
cp "$TOKENS_PATH" "$SCRIPT_DIR/build/package/model/v4096.tokens.u16"
cp "$SIGIL_MODEL_PATH" "$SCRIPT_DIR/build/package/solomon/model.nsrltch"
cp "$SIGIL_TEXT_INDEX_PATH" "$SCRIPT_DIR/build/package/solomon/solomon-spirit-text-signatures.tsv"
if [[ -n "$SIGIL_LATENT_MODEL_PATH" ]]; then
  if [[ -f "$SIGIL_LATENT_MODEL_PATH" ]]; then
    cp "$SIGIL_LATENT_MODEL_PATH" "$SCRIPT_DIR/build/package/solomon/current-best.nsrllat"
  elif [[ -n "${X_BOT_SIGIL_LATENT_MODEL_PATH:-}" ]]; then
    echo "missing Solomon latent model artifact: $SIGIL_LATENT_MODEL_PATH" >&2
    exit 1
  else
    echo "warning: Solomon latent model not found; Lambda will fall back to the text index: $SIGIL_LATENT_MODEL_PATH" >&2
  fi
fi

(
  cd "$SCRIPT_DIR/build/package"
  zip -q -r "$OUT" .
)

printf '%s\n' "$OUT"
