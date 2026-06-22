#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p web/assets web/pkg

MODEL_ROOT="${WEB_CHAT_MODEL_ROOT:-data/processed/crowley-bard-aphorism-v2}"
MODEL_PATH="${WEB_CHAT_MODEL_PATH:-$MODEL_ROOT/experiments/v4096.seq8-mean-reduce-base15-lr25-o98304.nsrllm}"
VOCAB_PATH="${WEB_CHAT_VOCAB_PATH:-$MODEL_ROOT/v4096.vocab.tsv}"
TOKENS_PATH="${WEB_CHAT_TOKENS_PATH:-$MODEL_ROOT/v4096.tokens.u16}"
SOLOMON_MODEL_PATH="${WEB_CHAT_SOLOMON_MODEL_PATH:-data/processed/key-solomon-goetia-denoise-v1/text-multichannel-conv/model.nsrltch}"
SOLOMON_TEXT_INDEX_PATH="${WEB_CHAT_SOLOMON_TEXT_INDEX_PATH:-data/processed/key-solomon-goetia-text-index-pg72679/solomon-spirit-text-signatures.tsv}"

cp "$MODEL_PATH" web/assets/model.nsrllm
cp "$VOCAB_PATH" web/assets/v4096.vocab.tsv
cp "$TOKENS_PATH" web/assets/v4096.tokens.u16
cp "$SOLOMON_MODEL_PATH" web/assets/solomon-model.nsrltch
cp "$SOLOMON_TEXT_INDEX_PATH" web/assets/solomon-spirit-text-signatures.tsv
cp data/processed/visionary-twitter-bot-demo/social-assets/crowley-bard-banner-1500x500.png \
  web/assets/crowley-bard-banner.png
cp data/processed/visionary-twitter-bot-demo/social-assets/crowley-bard-pfp-400.png \
  web/assets/crowley-bard-pfp.png

wasm-pack build crates/nsrl-web-wasm --release --target web --out-dir ../../web/pkg
rm -f web/pkg/.gitignore
